import pandas as pd
import numpy as np

# -----------------------------
# 1) CONFIG
# -----------------------------
CSV_PATH = "D:\\DATA_RIZAL\\KAMPUS\\SEMESTER_5\\MATHEMATICS_FOR_AI\\PROJECT_FINAL\\DATASET_ICAL.csv"  # ganti kalau nama file berbeda
X_COL = "studytime"
Y_COL = "Y"

ALPHA = 0.10
W0 = 0.10
B0 = 0.10
N_ITER = 5
THRESH = 0.5

np.set_printoptions(precision=10, suppress=True)

# -----------------------------
# 2) HELPER FUNCTIONS
# -----------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation.
    Untuk mencegah overflow di exp, z di-clip.
    """
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Binary Cross-Entropy (Log Loss).
    Tambah epsilon supaya tidak log(0).
    """
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def accuracy_from_prob(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> float:
    """
    Akurasi klasifikasi dengan threshold.
    """
    y_pred = (y_prob >= thresh).astype(int)
    return float(np.mean(y_pred == y_true))


# -------------
# 3) LOAD DATA
# --------------
df = pd.read_csv(CSV_PATH)

# Pastikan kolom yang dibutuhkan ada
missing_cols = [c for c in [X_COL, Y_COL] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Kolom tidak ditemukan di CSV: {missing_cols}. "
                     f"Kolom yang ada: {list(df.columns)}")

x = df[X_COL].astype(float).to_numpy()
y = df[Y_COL].astype(int).to_numpy()

n = len(x)
if n < 50:
    print(f"WARNING: data kamu hanya {n} baris (WAJIB MINIMAL 50 BARIS TUGASNYA).")

print(f"Loaded dataset: n={n}, x_col='{X_COL}', y_col='{Y_COL}'")
print("x sample:", x[:10])
print("y sample:", y[:10])

# -----------------------------
# 4) GRADIENT DESCENT TRAINING
# -----------------------------
w = float(W0)
b = float(B0)

history = []

for t in range(1, N_ITER + 1):
    z = w * x + b
    yhat = sigmoid(z)

    # NOTE: error = yhat - y 
    err = yhat - y

    grad_w = float(np.mean(err * x))
    grad_b = float(np.mean(err))

    loss = binary_cross_entropy(y, yhat)
    acc = accuracy_from_prob(y, yhat, THRESH)

    history.append({
        "iter": t,
        "w_before": w,
        "b_before": b,
        "grad_w": grad_w,
        "grad_b": grad_b,
        "loss": loss,
        "acc": acc
    })

    w = w - ALPHA * grad_w
    b = b - ALPHA * grad_b

    print("\n" + "=" * 60)
    print(f"ITERASI {t}")
    print(f"w_before = {history[-1]['w_before']:.10f}, b_before = {history[-1]['b_before']:.10f}")
    print(f"grad_w   = {grad_w:.10f}, grad_b   = {grad_b:.10f}")
    print(f"loss     = {loss:.10f}, acc       = {acc:.4f}")
    print(f"w_after  = {w:.10f}, b_after  = {b:.10f}")

# ------------
# 5) EVALUASI
# ------------
z_final = w * x + b
p_final = sigmoid(z_final)
loss_final = binary_cross_entropy(y, p_final)
acc_final = accuracy_from_prob(y, p_final, THRESH)

print("\n" + "#" * 60)
print("FINAL MODEL (setelah semua iterasi)")
print(f"w_final = {w:.10f}")
print(f"b_final = {b:.10f}")
print(f"final_loss = {loss_final:.10f}")
print(f"final_acc  = {acc_final:.4f}")

# ----------
# 6) EXPORT
# ---------
hist_df = pd.DataFrame(history)
hist_df["w_after"] = hist_df["w_before"] - ALPHA * hist_df["grad_w"]
hist_df["b_after"] = hist_df["b_before"] - ALPHA * hist_df["grad_b"]

hist_df.to_csv("RESULT_history_ical.csv", index=False)
print("\nSaved: RESULT_history_ical.csv (ringkasan iterasi)")

out_df = df.copy()
out_df["z_final"] = z_final
out_df["p_final"] = p_final
out_df["pred_final"] = (p_final >= THRESH).astype(int)
out_df.to_csv("RESULT_predictions_ical.csv", index=False)
print("Saved: RESULT_predictions_ical.csv (prediksi final per baris)")

accuracy_report = f"FINAL ACCURACY: {acc_final:.4f}\n"
