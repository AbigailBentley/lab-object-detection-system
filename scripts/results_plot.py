import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("runs/detect/train2/results.csv")

# Set seaborn theme
sns.set(style="whitegrid")
epochs = df["epoch"]

# 1. Training vs Validation Loss Curves
plt.figure(figsize=(10, 6))
plt.plot(epochs, df["train/box_loss"], label="Train Box Loss")
plt.plot(epochs, df["val/box_loss"], label="Val Box Loss")
plt.plot(epochs, df["train/cls_loss"], label="Train Cls Loss")
plt.plot(epochs, df["val/cls_loss"], label="Val Cls Loss")
plt.plot(epochs, df["train/dfl_loss"], label="Train DFL Loss")
plt.plot(epochs, df["val/dfl_loss"], label="Val DFL Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=300)
plt.show()


# 1b. then the separate loss graphs
loss_pairs = [
    ("box_loss", "Box Loss"),
    ("cls_loss", "Classification Loss"),
    ("dfl_loss", "Distribution Focal Loss")
]

# Plot each loss type separately
for loss_name, display_name in loss_pairs:
    train_col = f"train/{loss_name}"
    val_col = f"val/{loss_name}"
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df[train_col], label=f"Train {display_name}", linewidth=2)
    plt.plot(epochs, df[val_col], label=f"Val {display_name}", linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{display_name} Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{loss_name}_curve.png", dpi=300)
    plt.show()


# 2. mAP@50 and mAP@50–95 vs Epoch
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["metrics/mAP50(B)"], label="mAP@50")
plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@50–95")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("Mean Average Precision over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("map_curves.png", dpi=300)
plt.show()

# 3. Precision and Recall vs Epoch
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["metrics/precision(B)"], label="Precision")
plt.plot(epochs, df["metrics/recall(B)"], label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision and Recall over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall.png", dpi=300)
plt.show()

# 4. Learning Rate Schedule
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["lr/pg0"], label="lr/pg0")
plt.plot(epochs, df["lr/pg1"], label="lr/pg1")
plt.plot(epochs, df["lr/pg2"], label="lr/pg2")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.tight_layout()
plt.savefig("lr_schedule.png", dpi=300)
plt.show()

# 5. Time per Epoch
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["time"], label="Time per Epoch", color="teal")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.title("Training Time per Epoch")
plt.tight_layout()
plt.savefig("epoch_time.png", dpi=300)
plt.show()
