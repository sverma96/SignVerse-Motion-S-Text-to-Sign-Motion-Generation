import pandas as pd
import matplotlib.pyplot as plt

# Load training history
df = pd.read_csv("motion_s_outputs/training_history.csv")

plt.figure()

# Always plot loss
if "train_loss" in df.columns:
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")

if "val_loss" in df.columns:
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")

# Only plot contrastive if exists
if "train_contrastive" in df.columns:
    plt.plot(df["epoch"], df["train_contrastive"], label="Train Contrastive")

if "val_contrastive" in df.columns:
    plt.plot(df["epoch"], df["val_contrastive"], label="Val Contrastive")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.legend()
plt.grid()

plt.savefig("motion_s_outputs/loss_curve.png")
plt.show()