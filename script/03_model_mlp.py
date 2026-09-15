import os
import numpy as np
import pandas as pd
import joblib
import warnings
import importlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

# load module co ten bat dau bang so
processing = importlib.import_module("02_processing")
run_preprocessing = processing.run_preprocessing

# cau hinh chung
SEED       = 42
OUTPUT_DIR = "./output"
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# cau hinh gpu
def configure_gpu():
    """
    Bat memory growth de TF khong chiem toan bo VRAM luc khoi dong.
    Phu hop voi GPU co VRAM han che (4-8 GB).
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Phat hien {len(gpus)} GPU: {[g.name for g in gpus]}")
    else:
        print("[GPU] Khong tim thay GPU, su dung CPU.")
    return len(gpus) > 0


# buoc 1: kiem tra data leakage
def check_and_remove_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.9,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Tinh tuong quan Pearson giua tung feature va TARGET.
    Loai bo cot co |corr| > threshold de tranh AUC ao.

    Args:
        X         : Feature matrix
        y         : Nhan TARGET
        threshold : Nguong tuong quan (mac dinh 0.9)
        verbose   : In ket qua chi tiet

    Returns:
        X da loai bo cot leaky
    """
    print(f"[DATA LEAKAGE CHECK] Nguong |correlation| > {threshold}")
    corr_series = X.corrwith(y).abs().sort_values(ascending=False)

    if verbose:
        print("  Top-10 features tuong quan voi TARGET:")
        for feat, val in corr_series.head(10).items():
            flag = " LEAKY" if val > threshold else ""
            print(f"     {feat:45s}  corr = {val:.4f}{flag}")

    leaky = corr_series[corr_series > threshold].index.tolist()
    if leaky:
        print(f"[WARN] Loai bo {len(leaky)} cot co data leakage: {leaky}")
        X = X.drop(columns=leaky)
    else:
        print("[INFO] Khong phat hien data leakage.")

    return X


# buoc 2: kien truc mlp
def build_mlp(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    """
    Xay dung MLP voi BatchNormalization va Dropout.

    Kien truc:
        Input(input_dim)
        -> Dense(256, relu) -> BatchNorm -> Dropout(0.4)
        -> Dense(128, relu) -> BatchNorm -> Dropout(0.3)
        -> Dense(64,  relu) -> BatchNorm -> Dropout(0.2)
        -> Dense(1, sigmoid)

    BatchNormalization: on dinh qua trinh train, giam overfitting, tang toc hoi tu.
    Dropout: ngau nhien tat neuron moi buoc train, buoc model hoc feature doc lap.
    L2 Regularization: phat trong so lon, khuyen khich model don gian hon.
    """
    inputs = keras.Input(shape=(input_dim,), name="input_layer")

    # block 1: 256 neurons
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_256")(inputs)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Dropout(0.4, name="dropout_256")(x)

    # block 2: 128 neurons
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_128")(x)
    x = layers.BatchNormalization(name="bn_128")(x)
    x = layers.Dropout(0.3, name="dropout_128")(x)

    # block 3: 64 neurons
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_64")(x)
    x = layers.BatchNormalization(name="bn_64")(x)
    x = layers.Dropout(0.2, name="dropout_64")(x)

    # output layer
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_HomCredit")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# buoc 3: callbacks chong overfitting
def get_callbacks(model_path: str) -> list:
    """
    - EarlyStopping    : Dung khi val_auc khong cai thien sau 10 epoch
    - ReduceLROnPlateau: Giam learning rate khi val_auc dung cai thien
    - ModelCheckpoint  : Luu model tot nhat theo val_auc
    """
    early_stop = EarlyStopping(
        monitor="val_auc", patience=10,
        restore_best_weights=True, mode="max", verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc", factor=0.5, patience=5,
        min_lr=1e-6, mode="max", verbose=1,
    )
    checkpoint = ModelCheckpoint(
        filepath=model_path, monitor="val_auc",
        save_best_only=True, mode="max", verbose=0,
    )
    return [early_stop, reduce_lr, checkpoint]


# buoc 4: visualize ket qua
def plot_training_history(history, save_path: str):
    """Ve bieu do AUC va Loss theo epoch (train vs validation)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Lich su Huan luyen MLP - Home Credit Default Risk", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["auc"],     label="Train AUC",  color="#2196F3", lw=2)
    axes[0].plot(history.history["val_auc"], label="Val AUC",    color="#F44336", lw=2, linestyle="--")
    axes[0].set_title("ROC-AUC qua tung Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("AUC")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss", color="#4CAF50", lw=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#FF9800", lw=2, linestyle="--")
    axes[1].set_title("Binary Cross-Entropy Loss qua tung Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, auc_score: float, save_path: str):
    """Ve ROC Curve voi AUC score va nguong toi uu (Youden's J)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores    = tpr - fpr
    best_idx    = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_fpr    = fpr[best_idx]
    best_tpr    = tpr[best_idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"MLP (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Classifier")
    ax.scatter(best_fpr, best_tpr, color="#F44336", zorder=5,
            label=f"Nguong toi uu = {best_thresh:.3f}\n(FPR={best_fpr:.3f}, TPR={best_tpr:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve - Home Credit Default Risk", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return best_thresh


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """Ve Confusion Matrix co chu thich ti le phan tram."""
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Khong vo no (0)", "Vo no (1)"])

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix - Tap Validation", fontsize=13, fontweight="bold")

    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            ax.text(j, i + 0.35, f"({pct:.1f}%)", ha="center", va="center", fontsize=10, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names: list, top_n: int = 20, save_path: str = None):
    """
    Uoc luong feature importance bang trong so lop dau tien.
    Chi la proxy - dung SHAP de chinh xac hon.
    """
    first_layer_weights = model.get_layer("dense_256").get_weights()[0]  # shape: (n_feat, 256)
    importance = np.abs(first_layer_weights).mean(axis=1)

    feat_imp  = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    top_feats = feat_imp.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    top_feats[::-1].plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Top-{top_n} Feature Importance (proxy tu trong so lop 1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |Weight|"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return feat_imp


# pipeline training chinh
def train_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    test_size: float = 0.2,
    epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    corr_threshold: float = 0.9,
):
    """
    Pipeline huan luyen MLP hoan chinh.

    Args:
        X, y           : Du lieu da tien xu ly
        feature_names  : Danh sach ten feature
        test_size      : Ti le validation
        epochs         : So epoch toi da
        batch_size     : Kich thuoc batch
        learning_rate  : Learning rate khoi dau
        corr_threshold : Nguong loai bo data leakage

    Returns:
        model, history, val_auc
    """
    # kiem tra data leakage
    X_clean      = check_and_remove_leakage(X, y, threshold=corr_threshold)
    feature_names = X_clean.columns.tolist()

    # chia train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y, test_size=test_size, random_state=SEED, stratify=y,
    )
    print(f"[DATA SPLIT] Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"[INFO] Train TARGET=1: {y_train.mean():.4f} | Val TARGET=1: {y_val.mean():.4f}")

    # chuyen sang numpy
    X_train_np = X_train.values.astype(np.float32)
    X_val_np   = X_val.values.astype(np.float32)
    y_train_np = y_train.values.astype(np.float32)
    y_val_np   = y_val.values.astype(np.float32)

    # tinh class weight (bo sung cho truong hop khong dung SMOTE)
    neg, pos     = (y_train_np == 0).sum(), (y_train_np == 1).sum()
    class_weight = {0: 1.0, 1: neg / pos}
    print(f"[CLASS WEIGHT] w(0)={class_weight[0]:.2f}, w(1)={class_weight[1]:.2f}")

    # xay dung va huan luyen model
    input_dim  = X_train_np.shape[1]
    model      = build_mlp(input_dim, learning_rate=learning_rate)
    model.summary()

    model_path = os.path.join(MODEL_DIR, "mlp_best.keras")
    callbacks  = get_callbacks(model_path)

    print("[TRAINING] Bat dau huan luyen...")
    history = model.fit(
        X_train_np, y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # danh gia
    y_prob  = model.predict(X_val_np, batch_size=batch_size, verbose=0).flatten()
    val_auc = roc_auc_score(y_val_np, y_prob)
    print(f"[EVALUATION] Validation AUC: {val_auc:.4f}")

    # tim nguong tot nhat (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_val_np, y_prob)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred      = (y_prob >= best_thresh).astype(int)

    print(f"[CLASSIFICATION REPORT] (nguong = {best_thresh:.3f})")
    print(classification_report(y_val_np, y_pred, target_names=["Khong vo no (0)", "Vo no (1)"]))

    # ve bieu do
    plot_training_history(history, os.path.join(PLOT_DIR, "training_history.png"))
    plot_roc_curve(y_val_np, y_prob, val_auc, os.path.join(PLOT_DIR, "roc_curve.png"))
    plot_confusion_matrix(y_val_np, y_pred, os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plot_feature_importance(model, feature_names, save_path=os.path.join(PLOT_DIR, "feature_importance.png"))

    # luu model
    final_path = os.path.join(MODEL_DIR, "mlp_final.keras")
    model.save(final_path)
    print(f"[SAVE] Model da luu tai: {final_path}")

    # danh gia muc tieu
    if 0.75 <= val_auc <= 0.82:
        print(f"[RESULT] AUC = {val_auc:.4f} - DAT MUC TIEU (0.75 - 0.82)")
    elif val_auc > 0.82:
        print(f"[RESULT] AUC = {val_auc:.4f} - CAO HON MUC TIEU, kiem tra Data Leakage!")
    else:
        print(f"[RESULT] AUC = {val_auc:.4f} - Can toi uu them features hoac hyperparameter")

    return model, history, val_auc


# entry point
if __name__ == "__main__":
    has_gpu    = configure_gpu()
    batch_size = 1024 if has_gpu else 512

    processed_X = os.path.join("./output", "X_train_processed.parquet")
    processed_y = os.path.join("./output", "y_train_processed.parquet")

    if os.path.exists(processed_X) and os.path.exists(processed_y):
        print("[INFO] Tai du lieu da xu ly tu file Parquet...")
        X            = pd.read_parquet(processed_X)
        y            = pd.read_parquet(processed_y).squeeze()
        feature_names = X.columns.tolist()
        print(f"[INFO] X shape: {X.shape} | y distribution:\n{y.value_counts()}")
    else:
        print("[INFO] File Parquet chua ton tai - chay tien xu ly...")
        X, y, feature_names = run_preprocessing(
            balance_strategy="combined",
            drop_leakage=True,
            corr_threshold=0.9,
        )

    model, history, val_auc = train_mlp(
        X=X, y=y, feature_names=feature_names,
        test_size=0.2, epochs=100,
        batch_size=batch_size, learning_rate=1e-3,
        corr_threshold=0.9,
    )

    # in thong so tong ket
    print("=" * 50)
    print(f"[SUMMARY] AUC Score: {val_auc:.4f}")
    print(f"[SUMMARY] Tong so tham so: {model.count_params():,}")
    print("=" * 50)
