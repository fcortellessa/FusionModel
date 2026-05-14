import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

# Load your CSV (replace with your file path)
df = pd.read_csv(r"runs\20260511_182856_bs16_concat\metrics.csv")

# Ensure numeric columns
numeric_cols = [
    "epoch", "train_loss", "loss_q", "loss_y", "loss_z",
    "train_abs_y", "train_abs_z", "train_abs_trans", "train_abs_rot_x",
    "val_abs_y", "val_abs_z", "val_abs_trans", "val_abs_rot_x",
    "lr", "lr_base", "lr_head_rot", "lr_head_y", "lr_head_z",
]
existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
df[existing_numeric_cols] = df[existing_numeric_cols].apply(pd.to_numeric, errors='coerce')


def format_metadata_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def get_run_metadata_text(frame):
    metadata_fields = ["fusion", "head_type", "batch_size"]
    parts = []
    for field in metadata_fields:
        if field not in frame.columns:
            continue
        series = frame[field].dropna()
        if series.empty:
            continue
        value = format_metadata_value(series.iloc[0])
        if value is not None:
            parts.append(f"{field}={value}")
    return " | ".join(parts) if parts else None


def add_run_metadata(fig, frame):
    metadata_text = get_run_metadata_text(frame)
    if metadata_text:
        fig.text(
            0.01,
            0.01,
            metadata_text,
            ha="left",
            va="bottom",
            fontsize=8,
            alpha=0.75,
        )


def plot_train_val(ax, epoch_col, train_col, val_col, title, ylabel, train_label=None, val_label=None, train_color="#1f77b4", val_color="#ff7f0e"):
    if train_col in df.columns:
        ax.plot(df[epoch_col], df[train_col], color=train_color, label=train_label or f"Train {train_col}")
    if val_col in df.columns:
        ax.plot(df[epoch_col], df[val_col], color=val_color, label=val_label or f"Val {val_col}")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)


def maybe_parse_fusion_weights(value):
    if pd.isna(value) or value == "":
        return None
    try:
        parsed = literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        return parsed
    return None

# =========================
# 1. TRAIN VS VAL LOSS
# =========================
fig, ax = plt.subplots(figsize=(10, 4))
plot_train_val(
    ax,
    "epoch",
    "train_loss",
    "val_loss",
    "Train vs Validation Loss",
    "Loss",
    train_label="Train Loss",
    val_label="Val Loss",
)
if "train_loss" in df.columns:
    best_idx = df["train_loss"].idxmin()
    ax.annotate(
        f"Best train: {df.loc[best_idx, 'train_loss']:.2f}",
        (df.loc[best_idx, "epoch"], df.loc[best_idx, "train_loss"]),
    )
ax.set_xlabel("Epoch")
add_run_metadata(fig, df)
fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()


# =========================
# 2. TRANSLATION AND ROTATION ERRORS
# =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
plot_train_val(
    ax1,
    "epoch",
    "train_abs_trans",
    "val_abs_trans",
    "Translation Error",
    "Translation Error (mm)",
    train_label="Train Translation",
    val_label="Val Translation",
)
plot_train_val(
    ax2,
    "epoch",
    "train_abs_rot_x",
    "val_abs_rot_x",
    "Rotation Error",
    "Rotation Error (deg)",
    train_label="Train Rotation",
    val_label="Val Rotation",
    train_color="orange",
    val_color="#d62728",
)
ax2.set_xlabel("Epoch")

add_run_metadata(fig, df)
fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()


# =========================
# 3. PER-AXIS TRANSLATION (IF AVAILABLE)
# =========================
if "train_abs_y" in df.columns or "val_abs_y" in df.columns or "train_abs_z" in df.columns or "val_abs_z" in df.columns:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plot_train_val(ax1, "epoch", "train_abs_y", "val_abs_y", "Y-axis Error", "Abs Error (m)", train_label="Train Y", val_label="Val Y")
    plot_train_val(ax2, "epoch", "train_abs_z", "val_abs_z", "Z-axis Error", "Abs Error (m)", train_label="Train Z", val_label="Val Z", train_color="#2ca02c", val_color="#9467bd")
    ax2.set_xlabel("Epoch")
    add_run_metadata(fig, df)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

# =========================
# 4. LOSS COMPONENTS
# =========================
fig, ax = plt.subplots(figsize=(10, 4))
if "loss_q" in df.columns:
    ax.plot(df["epoch"], df["loss_q"], label="loss_q")
if "loss_y" in df.columns:
    ax.plot(df["epoch"], df["loss_y"], label="loss_y")
if "loss_z" in df.columns:
    ax.plot(df["epoch"], df["loss_z"], label="loss_z")
ax.set_title("Loss Components")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss Value")
ax.legend()
ax.grid(True)
add_run_metadata(fig, df)
fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()


# =========================
# 5. FUSION WEIGHTS (IF AVAILABLE)
# =========================
if "fusion_weights" in df.columns:
    parsed_weights = df["fusion_weights"].apply(maybe_parse_fusion_weights)
    valid_mask = parsed_weights.notna()
    if valid_mask.any():
        first_valid = parsed_weights[valid_mask].iloc[0]
        if len(first_valid) == 2:
            fig, ax = plt.subplots(figsize=(10, 4))
            weight_0 = parsed_weights.apply(lambda x: x[0] if x is not None else None)
            weight_1 = parsed_weights.apply(lambda x: x[1] if x is not None else None)
            ax.plot(df["epoch"], weight_0, label="Fusion Weight Stream 0")
            ax.plot(df["epoch"], weight_1, label="Fusion Weight Stream 1")
            ax.set_title("Fusion Weights Over Epochs")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight")
            ax.legend()
            ax.grid(True)
            add_run_metadata(fig, df)
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            plt.show()


# =========================
# 6. LEARNING RATE (IF AVAILABLE)
# =========================
lr_columns = [col for col in ["lr_base", "lr_head_rot", "lr_head_y", "lr_head_z"] if col in df.columns]
if lr_columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in lr_columns:
        ax.plot(df["epoch"], df[col], label=col)
    ax.set_title("Learning Rates Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.legend()
    ax.grid(True)
    add_run_metadata(fig, df)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


# =========================
# 7. SUMMARY PRINT
# =========================
print("\n===== TRAINING SUMMARY =====")
print(f"Initial Loss: {df['train_loss'].iloc[0]:.3f}")
print(f"Final Loss: {df['train_loss'].iloc[-1]:.3f}")
print(f"Best Loss: {df['train_loss'].min():.3f} (epoch {df.loc[df['train_loss'].idxmin(), 'epoch']})")

if "val_abs_y" in df.columns:
    print(f"\nFinal Val Translation in y: {df['val_abs_y'].iloc[-1]:.6f}")
if "val_abs_z" in df.columns:
    print(f"Final Val Translation in z: {df['val_abs_z'].iloc[-1]:.6f}")
elif "val_abs_trans" in df.columns:
    print(f"Final Val Translation: {df['val_abs_trans'].iloc[-1]:.6f}")
if "val_abs_rot_x" in df.columns:
    print(f"Final Val Rotation around x: {df['val_abs_rot_x'].iloc[-1]:.6f}")
