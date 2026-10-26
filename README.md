# Segmentasi Nasabah Bank berbasis Perilaku Transaksi

Mengelompokkan nasabah bank berdasarkan **perilaku bertransaksi**, lalu menguji secara jujur apakah kelompok itu benar-benar ada di data.
Proyek ini mencakup pembersihan data, feature engineering level nasabah, clustering dengan validasi berlapis, model pendamping yang dapat dijelaskan, dan dashboard interaktif.

**Demo:** `<tempel tautan Streamlit Community Cloud di sini>`

<!-- Taruh 2-3 tangkapan layar dashboard di folder docs/ lalu aktifkan baris di bawah
![Ringkasan](docs/ringkasan.png)
![Simulasi](docs/simulasi.png)
-->

## Ringkasan hasil

| Metrik | Nilai | Arti |
|---|---|---|
| Jumlah nasabah | 494 | Satu baris = satu akun (hasil agregasi dari transaksi) |
| Jumlah segmen (k) | 3 | Dipilih lewat silhouette pada rentang 3–6 |
| Silhouette | 0,137 | Pada data yang kolomnya diacak: 0,088 |
| Stabilitas bootstrap (ARI) | 0,66 | Seberapa konsisten segmen jika data diambil ulang |
| Model pendamping | Logistic Regression, macro-F1 0,99 | Menunjukkan segmen bisa ditiru model, **bukan** bukti segmen valid |
| Kekuatan pengelompokan | **Lemah** | Segmen bersifat deskriptif, bukan kelompok alami |

**Temuan utama:** nasabah di data ini tidak membentuk kelompok yang terpisah jelas. Silhouette hanya sedikit di atas data acak, dan tidak ada kombinasi algoritma (KMeans, GMM, Ward) yang mengubahnya.
Walau begitu, tiga segmen tetap berguna untuk **membaca perbedaan perilaku**:

- **Segmen 1 (44%)** paling sering bertransaksi (z = +0,86 dibanding rata-rata).
- **Segmen 0 (29%)** lebih banyak memakai kanal Online (z = +0,75) dan lebih jarang bertransaksi (z = −0,58).
- **Segmen 2 (27%)** paling jarang bertransaksi (z = −0,78) dan paling jarang memakai kanal Online (z = −0,76).

*(Angka z diambil dari tab Ringkasan dashboard. Lengkapnya ada di sana.)*

## Pertanyaan bisnis

Kelompok nasabah seperti apa yang ada, dan tindakan apa yang masuk akal untuk tiap kelompok (misalnya mendorong kanal digital, retensi, atau pemantauan risiko)?

## Data

Dataset transaksi bank (±2.500 transaksi, 16 kolom) dari materi Dicoding, periode Januari 2023 hingga Januari 2024. Satu nasabah memiliki beberapa transaksi, sehingga analisis dilakukan pada level **akun**.

> Isi sebelum dipublikasikan: sumber asli dataset dan lisensinya.

## Pendekatan

1. **Audit data.** Memeriksa arti satu baris, pola missing value, dan kolom tanpa variasi.
2. **Pembersihan.** Membuang duplikat dan baris tanpa kunci. Sisanya diimputasi (median/modus per akun). Outlier **tidak dibuang** karena login berulang adalah sinyal risiko.
3. **Feature engineering level nasabah.** Frekuensi, nominal dan saldo, rasio nominal/saldo, bauran kanal, proporsi login berulang, jumlah lokasi. Fitur tanpa variasi dibuang otomatis (misalnya `avg_gap_days`, yang bernilai 0 untuk semua akun).
4. **Clustering KMeans** dalam satu `Pipeline` (log-transform lalu `StandardScaler`). Pemilihan k memakai silhouette, Davies–Bouldin, Calinski–Harabasz, dan stabilitas bootstrap.
5. **Validasi.** Silhouette dibandingkan dengan data yang kolomnya diacak, dan KMeans dibandingkan dengan GMM dan Agglomerative.
6. **Model pendamping.** Beberapa klasifier dengan baseline, stratified CV, dan tuning. Test set dipakai sekali. Ada juga uji apakah usia dan pekerjaan saja cukup untuk menebak segmen.
7. **Dashboard** Streamlit di atas artefak model.

## Yang saya pelajari: kesalahan di versi pertama

Versi awal proyek ini terlihat bagus di angka (silhouette 0,67) tetapi keliru. Saya memeriksa ulang dan menemukan:

- `LocationCoords` (jarak ke satu kota, skala ribuan km) tidak diskalakan sehingga menguasai KMeans. **Ketiga cluster ternyata hanya tiga rentang jarak** yang tidak tumpang tindih.
- PCA dengan ambang 95% hanya menghasilkan 1 komponen, tanda satu fitur menguasai varians.
- Klasifikasi tanpa fitur itu hanya mencapai akurasi 0,38, di bawah tebakan kelas mayoritas (0,40). Dengan fitur itu akurasinya 1,00. Jadi masalahnya ada di target, bukan di model.
- Outlier yang dibuang menghilangkan semua login ganda, sehingga `LoginAttempts` menjadi konstan.
- File "model" ternyata hanya menyimpan data, bukan model.

Perbaikannya: unit analisis diganti ke nasabah, semua fitur diskalakan dalam pipeline, outlier dijadikan fitur, validitas diuji terhadap data acak, dan yang disimpan adalah pipeline utuh.

## Limitasi

- Pola data menunjukkan ciri data yang dibangkitkan acak (korelasi antar fitur perilaku mendekati nol), sehingga struktur kelompok alami lemah. Ini perlu dikonfirmasi dengan sumber data.
- Tidak ada label fraud atau hasil bisnis, jadi dampak rekomendasi belum teruji (perlu A/B test).
- Model pendamping adalah *surrogate* dari clustering. Akurasi tinggi tidak membuktikan validitas segmen.
- Segmen ditetapkan dari seluruh periode dan tidak diperbarui otomatis. Grafik bulanan di dashboard menunjukkan komposisi transaksi, bukan pergeseran segmen.

## Struktur proyek

```
.
├── Customer_Segmentation_Improved.ipynb   # analisis end-to-end
├── app.py                                 # dashboard Streamlit
├── requirements.txt
├── .streamlit/config.toml                 # tema dashboard
└── artifacts/                             # keluaran notebook
    ├── segmentation_pipeline.joblib
    ├── segment_classifier.joblib
    ├── customers_with_segment.csv
    ├── transactions_with_segment.csv
    ├── segment_profile.csv
    ├── segment_profile_zscore.csv
    └── metadata.json
```

## Menjalankan

1. Jalankan `Customer_Segmentation_Improved.ipynb` (lokal atau Colab). Sel terakhir membuat `artifacts.zip`. Ekstrak sehingga folder `artifacts/` ada di sebelah `app.py`.
2. Pasang pustaka dengan **versi scikit-learn yang sama** seperti saat model dilatih (tercatat di `artifacts/metadata.json`):
   ```bash
   pip install -r requirements.txt
   pip install scikit-learn==1.6.1
   ```
3. Jalankan dashboard:
   ```bash
   streamlit run app.py
   ```

## Rencana lanjutan

- Menguji pipeline pada data dengan kelompok yang sengaja ditanam, untuk membuktikan metode mampu mendeteksi cluster bila ada.
- Menerapkan pendekatan yang sama pada dataset ritel/RFM yang memiliki struktur nyata.
- Mengubah pertanyaan ke skor risiko (login berulang, rasio nominal/saldo) yang lebih sesuai dengan sifat data.
