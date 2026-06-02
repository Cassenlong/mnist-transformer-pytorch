import os
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# 1. 固定随机种子，方便复现实验
# =========================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 2. Patch Embedding
# 把 28x28 手写数字图像切成若干 patch
# 类似简化版 Vision Transformer
# =========================
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=28, patch_size=4, in_channels=1, embed_dim=96):
        super().__init__()

        assert image_size % patch_size == 0, "image_size 必须能被 patch_size 整除"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.proj(x)
        # x: [B, embed_dim, 7, 7]

        x = x.flatten(2)
        # x: [B, embed_dim, num_patches]

        x = x.transpose(1, 2)
        # x: [B, num_patches, embed_dim]

        return x


# =========================
# 3. 自己实现多头自注意力
# =========================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=96, num_heads=4, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        qkv = self.qkv(x)
        # qkv: [B, N, 3C]

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # qkv: [B, N, 3, heads, head_dim]

        qkv = qkv.permute(2, 0, 3, 1, 4)
        # qkv: [3, B, heads, N, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = q @ k.transpose(-2, -1)
        # attn_scores: [B, heads, N, N]

        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = attn_weights @ v
        # out: [B, heads, N, head_dim]

        out = out.transpose(1, 2)
        # out: [B, N, heads, head_dim]

        out = out.reshape(B, N, C)
        # out: [B, N, C]

        out = self.out_proj(out)
        out = self.out_drop(out)

        return out


# =========================
# 4. Transformer Encoder Block
# =========================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=96, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LN Transformer
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =========================
# 5. MNIST Transformer 分类模型
# =========================
class MNISTVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=96,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        B = x.shape[0]

        x = self.patch_embed(x)
        # x: [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        # cls_tokens: [B, 1, embed_dim]

        x = torch.cat((cls_tokens, x), dim=1)
        # x: [B, num_patches + 1, embed_dim]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_out = x[:, 0]
        # cls_out: [B, embed_dim]

        logits = self.head(cls_out)
        # logits: [B, 10]

        return logits


# =========================
# 6. 训练一个 epoch
# =========================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples * 100

        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc:.2f}%"
        })

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples * 100

    return epoch_loss, epoch_acc


# =========================
# 7. 测试模型
# =========================
def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples * 100

    return epoch_loss, epoch_acc


# =========================
# 8. 绘制训练损失曲线和测试准确率曲线
# =========================
def plot_training_curves(history, save_dir="."):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # =========================
    # 8.1 训练损失曲线
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()

    loss_curve_path = save_dir / "training_loss_curve.png"
    plt.savefig(loss_curve_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Training loss curve saved to: {loss_curve_path.resolve()}")

    # =========================
    # 8.2 测试准确率曲线
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["test_acc"], marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Curve")
    plt.grid(True)
    plt.tight_layout()

    acc_curve_path = save_dir / "test_accuracy_curve.png"
    plt.savefig(acc_curve_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Test accuracy curve saved to: {acc_curve_path.resolve()}")


# =========================
# 9. 可视化预测结果
# =========================
def visualize_predictions(model, test_loader, device, save_path="mnist_predictions.png", num_images=8):
    model.eval()

    images, labels = next(iter(test_loader))
    images_device = images.to(device)

    with torch.no_grad():
        logits = model(images_device)
        preds = logits.argmax(dim=1).cpu()

    plt.figure(figsize=(14, 4))

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Prediction visualization saved to: {Path(save_path).resolve()}")


# =========================
# 10. 主函数
# =========================
def main():
    set_seed(42)

    data_dir = "./data"
    batch_size = 128
    epochs = 10
    learning_rate = 3e-4
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )

    # Windows + PyCharm 初学环境下，num_workers=0 最稳
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = MNISTVisionTransformer(
        image_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=96,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_acc = 0.0
    best_model_path = "best_mnist_transformer.pth"

    # 用于保存每个 epoch 的训练和测试指标
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()

        # 记录当前 epoch 的指标
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved. Best Test Acc: {best_acc:.2f}%")

    print("\n" + "=" * 60)
    print(f"Training finished. Best Test Accuracy: {best_acc:.2f}%")
    print("=" * 60)

    if best_acc >= 95:
        print("Requirement satisfied: accuracy is above 95%.")
    else:
        print("Accuracy is below 95%. Try increasing epochs to 15 or 20.")

    # 打印 history，方便确认曲线数据确实被记录了
    print("\nTraining history:")
    print(history)

    # 绘制训练损失曲线和测试准确率曲线
    plot_training_curves(
        history=history,
        save_dir="."
    )

    # 加载最佳模型并可视化预测结果
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    visualize_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        save_path="mnist_predictions.png",
        num_images=8
    )


if __name__ == "__main__":
    main()
