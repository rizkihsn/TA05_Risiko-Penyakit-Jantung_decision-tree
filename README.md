# CardioTree: Klasifikasi Risiko Penyakit Jantung dengan Decision Tree (TA-05)

Tugas Akhir 05 - Praktikum Pembelajaran Mesin. Aplikasi web Flask yang memprediksi ada/tidaknya penyakit jantung
dari 11 data klinis pasien memakai model Decision Tree yang sudah di-*pruning* (`max_depth=3`).
Selain persentase kemungkinan, aplikasi menampilkan **alur keputusan** pohon (aturan jika-maka yang dilalui)
dan ilustrasi segmen ST yang berubah sesuai isian.

Dataset: Soriano, F. (2021). *Heart Failure Prediction Dataset*. Kaggle.
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

> Aplikasi ini hanya untuk pembelajaran dan bukan alat diagnosis medis.

## Hasil singkat
| | Model A (`max_depth=None`) | Model B (`max_depth=3`) |
|---|---|---|
| Akurasi data latih | 100,0% | 85,0% |
| Akurasi data uji | 79,3% | 85,5% |
| AUC data uji | 0,796 | 0,915 |
| Jumlah daun | 100 | 8 |

## Struktur
```
app.py                            aplikasi Flask (entrypoint, variabel `app`)
templates/index.html              tampilan (HTML, CSS, JS dalam satu berkas)
model/model_decision_tree.joblib  Model B (dipakai aplikasi)
model/model_A_overfit.joblib      Model A (pembanding, tidak dipakai aplikasi)
data/heart.csv                    dataset
TA05_Decision_Tree_Heart.ipynb    notebook analisis (EDA, Model A vs B, interpretasi)
figures/                          gambar hasil analisis dan tangkapan layar
requirements.txt  .python-version  .vercelignore
```

## Menjalankan di komputer
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                   # buka http://127.0.0.1:5000
```
Versi paket di `requirements.txt` sengaja dikunci sama dengan yang dipakai saat model disimpan.
Jangan ubah versi `scikit-learn`, karena file `.joblib` hanya dijamin dapat dimuat pada versi yang sama.

## Mengunggah ke GitHub
1. Buat repositori kosong di https://github.com/new (mis. `ta05-decision-tree-heart`).
2. Di folder proyek ini:
```bash
git init
git add .
git commit -m "TA-05: CardioTree, Decision Tree klasifikasi penyakit jantung"
git branch -M main
git remote add origin https://github.com/USERNAME/ta05-decision-tree-heart.git
git push -u origin main
```

## Deploy ke Vercel
1. Login ke https://vercel.com dengan akun GitHub, lalu **Add New > Project** dan impor repositori di atas.
2. Vercel mengenali Flask otomatis dari `app.py` dan `requirements.txt`, jadi build command tidak perlu diisi.
   Versi Python dibaca dari `.python-version` (3.13).
3. Klik **Deploy**. Setelah selesai, URL aplikasi (`https://nama-proyek.vercel.app`) muncul di dashboard.
4. `.vercelignore` mengecualikan `figures/`, `data/`, dan notebook dari paket deployment karena tidak dipakai saat runtime.
   Jika build gagal dengan pesan ukuran function melebihi batas, cek log build dan kurangi berkas yang ikut terunggah.
5. Setelah deploy, buka halaman utama dan coba tombol **Coba contoh berisiko**. Jika muncul `TemplateNotFound`
   atau model gagal dimuat, periksa log function di dashboard Vercel.
