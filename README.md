# Analisis Data Penjualan Produk dengan Teorema Bayes dan Regresi Linear

📌 Proyek ini merupakan tugas **Mathematics for AI** yang membahas analisis data menggunakan pendekatan statistik dan machine learning sederhana. Fokus utama eksperimen adalah klasifikasi biner menggunakan **Logistic Regression satu fitur** yang dioptimasi dengan **Gradient Descent**, kemudian diverifikasi dengan **perhitungan manual (Excel)** dan **perhitungan otomatis (Python)**.

---

## 👤 Identitas

- **Nama**: Muhammad Rizal Haris  
- **NIM**: 105841103223  
- **Program Studi**: Informatika  
- **Fakultas**: Teknik  
- **Universitas**: Universitas Muhammadiyah Makassar  
- **Tahun**: 2025  

---

## 🎯 Tujuan Proyek

1. Menganalisis dataset sederhana yang berisi variabel pembelajaran.
2. Menerapkan model **Logistic Regression** untuk klasifikasi target.
3. Menghitung proses optimasi menggunakan **Gradient Descent**.
4. Membandingkan hasil **perhitungan manual Excel** vs **hasil Python**.
5. Mengukur performa model menggunakan **akurasi**.

---

## 📂 Dataset

Dataset terdiri dari 3 kolom utama:

| Kolom | Deskripsi |
|------|----------|
| `studytime` | Waktu belajar (fitur utama / X) |
| `G3` | Nilai akhir (variabel tambahan) |
| `Y` | Label target biner (0/1) |

Contoh data:

| studytime | G3 | Y |
|----------|----|---|
| 2 | 6  | 0 |
| 2 | 10 | 1 |
| 3 | 15 | 1 |
| 1 | 5  | 0 |
| 4 | 20 | 1 |

Total data yang digunakan: **50 data**.

---

## 🧠 Metode yang Digunakan

### 1) Logistic Regression (1 fitur)

Model probabilitas:

\[
z = w \cdot x + b
\]

\[
p = \sigma(z) = \frac{1}{1+e^{-z}}
\]

Prediksi kelas:

- Jika `p >= 0.5` → prediksi = **1**
- Jika `p < 0.5` → prediksi = **0**

---

### 2) Gradient Descent

Update parameter:

\[
w := w - \alpha \cdot \frac{\partial L}{\partial w}
\]

\[
b := b - \alpha \cdot \frac{\partial L}{\partial b}
\]

Parameter awal:

- `w = 0.1`
- `b = 0.1`
- `alpha = 0.1`

Iterasi dilakukan sampai 5 kali (sesuai perhitungan di dokumen tugas).

---

## 🧪 Implementasi Python

Output Python yang ditampilkan meliputi:

✅ perhitungan nilai:
- `z_final`
- `p_final`
- `pred_final`

✅ hasil iterasi Gradient Descent:
- `w_before`, `b_before`
- `grad_w`, `grad_b`
- `loss`
- `acc`
- `w_after`, `b_after`

---

## 🧾 Perhitungan Manual (Excel)

Perhitungan manual dilakukan untuk memvalidasi proses Gradient Descent.

Pada Excel ditunjukkan:

- perhitungan `z`
- perhitungan `p = sigmoid(z)`
- error
- `grad_w` dan `grad_b`
- update nilai `w` dan `b`

Dilakukan dari:

✅ Iterasi 1 sampai Iterasi 5

---

## 📊 Hasil

### ✅ Akurasi

Akurasi yang diperoleh:

> **0.8 (80%)**

Menariknya, baik hasil Python maupun hasil Excel manual menunjukkan **akurasi identik**, yang berarti implementasi dan validasi berjalan benar.

---

## ✅ Kesimpulan

Perhitungan manual menggunakan Excel dan hasil komputasi menggunakan Python menunjukkan nilai akurasi yang sama yaitu **80%**. Hal ini membuktikan implementasi model Logistic Regression satu fitur dengan optimasi Gradient Descent sudah dilakukan secara benar serta konsisten, dan perhitungan manual mampu memvalidasi hasil komputasi secara menyeluruh.

---
