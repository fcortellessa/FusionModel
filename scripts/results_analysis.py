import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV (replace with your file path)
df = pd.read_csv("logs/train_20260415_122920_metrics.csv")

# Ensure numeric columns
numeric_cols = [
    "epoch", "train_loss", "val_abs_trans", "val_abs_rot_x",
    "loss_q", "loss_y", "loss_z", "train_abs_trans", "train_abs_rot_x"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# =========================
# 1. TRAIN LOSS TREND
# Check if it decreases and if there are plateaus
# =========================
plt.figure()
plt.plot(df["epoch"], df["train_loss"])
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.grid()

# Annotate best point
best_idx = df["train_loss"].idxmin()
plt.annotate(
    f"Best: {df['train_loss'][best_idx]:.2f}",
    (df["epoch"][best_idx], df["train_loss"][best_idx])
)

plt.show()


# =========================
# 2. VALIDATION METRICS
# Check if they decrease and if they are noisy (dataset or batch size issue)
# =========================

# two plots in one figure for translation and rotation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Translation Plot
ax1.plot(df["epoch"], df["val_abs_trans"], color='#1f77b4', label="Val Translation Error")
ax1.set_ylabel("Translation Error (m)")
ax1.set_title("Validation Performance Over Epochs")
ax1.legend()
ax1.grid(True)

# Rotation Plot
ax2.plot(df["epoch"], df["val_abs_rot_x"], color='orange', label="Val Rotation Error")
ax2.set_ylabel("Rotation Error (deg)")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# =========================
# 3. TRAIN vs VAL (OVERFITTING CHECK)
# If train ↓ but val ↑ → overfitting
# If both ↓ but val is noisy → dataset or batch size is small
# =========================

plt.figure()
plt.plot(df["epoch"], df["train_abs_trans"], label="Train Translation")
plt.plot(df["epoch"], df["val_abs_trans"], label="Val Translation")

plt.title("Train vs Validation Translation")
plt.xlabel("Epoch")
plt.ylabel("Translation Error")
plt.legend()
plt.grid()

plt.show()

plt.figure()
plt.plot(df["epoch"], df["train_abs_rot_x"], label="Train Rotation")
plt.plot(df["epoch"], df["val_abs_rot_x"], label="Val Rotation")

plt.title("Train vs Validation Rotation")
plt.xlabel("Epoch")
plt.ylabel("Rotation Error")
plt.legend()
plt.grid()

plt.show()


# =========================
# 4. LOSS COMPONENTS
# Check if one dominates or if they plateau together
# =========================
plt.figure()
plt.plot(df["epoch"], df["loss_q"], label="loss_q (main)")
plt.plot(df["epoch"], df["loss_y"], label="loss_y")
plt.plot(df["epoch"], df["loss_z"], label="loss_z")

plt.title("Loss Components")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
plt.grid()

plt.show()


# =========================
# 5. ROTATION vs TRANSLATION (TRAIN)
# Check which is harder to learn and if they plateau together
# Model might focus on one and neglect the other if they have very different scales or difficulties
# =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(df["epoch"], df["train_abs_trans"], color='#1f77b4', label="Train Translation Error")
ax1.set_ylabel("Translation Error (m)")
ax1.set_title("Training Performance Over Epochs")
ax1.legend()
ax1.grid(True)

ax2.plot(df["epoch"], df["train_abs_rot_x"], color='orange', label="Train Rotation Error")
ax2.set_ylabel("Rotation Error (deg)")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.grid(True)

# plt.figure()
# plt.plot(df["epoch"], df["train_abs_trans"], label="Train Translation")
# plt.plot(df["epoch"], df["train_abs_rot_x"], label="Train Rotation")

# plt.title("Training Errors")
# plt.xlabel("Epoch")
# plt.ylabel("Error")
# plt.legend()
# plt.grid()

plt.show()


# =========================
# 6. PLATEAU DETECTION
# =========================
# df["loss_diff"] = df["train_loss"].diff()

# plt.figure()
# plt.plot(df["epoch"], df["loss_diff"])
# plt.axhline(0)

# plt.title("Loss Change Per Epoch\n(Plateau ≈ values near 0)")
# plt.xlabel("Epoch")
# plt.ylabel("Δ Loss")
# plt.grid()

# # Highlight plateau region (small changes)
# plateau = df[df["loss_diff"].abs() < 0.05]
# if not plateau.empty:
#     plt.scatter(plateau["epoch"], plateau["loss_diff"])
#     plt.annotate("Plateau region", (plateau["epoch"].iloc[0], plateau["loss_diff"].iloc[0]))

# plt.show()


# =========================
# 7. SUMMARY PRINT
# =========================
print("\n===== TRAINING SUMMARY =====")
print(f"Initial Loss: {df['train_loss'].iloc[0]:.3f}")
print(f"Final Loss: {df['train_loss'].iloc[-1]:.3f}")
print(f"Best Loss: {df['train_loss'].min():.3f} (epoch {df.loc[df['train_loss'].idxmin(), 'epoch']})")

print(f"\nFinal Val Translation: {df['val_abs_trans'].iloc[-1]:.6f}")
print(f"Final Val Rotation: {df['val_abs_rot_x'].iloc[-1]:.6f}")
