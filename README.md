=== README - PREDIKSI NILAI UJIAN MAHASISWA ===
Berdasarkan Jam Belajar | Simple Linear Regression

Nama : Alea Farrel 
NIM  : 25.11.6401
Kelas: IF03 / S1-Informatika
Mata Kuliah : Pengantar Sains Data

DESKRIPSI PROYEK
-----------------
Program ini adalah Final Project Mini Machine Learning untuk 
mata kuliah Pengantar Sains Data. Program memprediksi nilai 
ujian mahasiswa berdasarkan jumlah jam belajar per hari 
menggunakan algoritma Simple Linear Regression yang 
diimplementasikan secara manual (tanpa library sklearn).

DATASET
--------
Nama    : Student Performance Dataset (Hours vs Scores)
Sumber  : github.com/AdiPersonalWorks/Random
File    : Dataset/dataset.csv
Kolom   : Jam_Belajar (jam/hari), Nilai (skor ujian 0-100)
Ukuran  : 10.000 baris data

CARA MENJALANKAN PROGRAM
-------------------------
1. Install library yang dibutuhkan:
   pip install pandas numpy matplotlib

2. Jalankan program utama:
   python main.py

3. Ikuti instruksi di terminal:
   - Tekan ENTER untuk memuat data
   - Tunggu proses EDA, Training, dan Evaluasi selesai
   - Dua grafik output akan otomatis terbuka
   - Masukkan angka jam belajar pada mode interaktif

STRUKTUR FOLDER
----------------
prediksi_nilai_berdasarkan_jam_belajar/
├─ utils.py                   <- Modul fungsi (load, EDA, train, simpan, dll)
├─ main.py                    <- File eksekusi utama
├─ Readme.txt                 <- Dokumentasi proyek (file ini)
├─ Dataset/
│  └─ dataset.csv             <- Data mentah (jam belajar & nilai)
└─ output/
   ├─ hasil_analisis.txt      <- Ringkasan statistik & evaluasi model
   ├─ grafik_output.png       <- Grafik 1: Scatter Plot + Garis Regresi
   └─ grafik_distribusi.png   <- Grafik 2: Bar Chart rata-rata nilai per kategori

ALUR PROGRAM
-------------
1. Memuat dataset (dataset.csv) menggunakan Pandas
2. Exploratory Data Analysis (EDA): cek missing value & korelasi
3. Persiapan Data: Split Training & Testing (80:20)
4. Melatih Model Linear Regression (metode OLS manual)
5. Evaluasi Model: MAE, MSE, RMSE, R-squared
6. Menyimpan Output:
   - hasil_analisis.txt  (statistik deskriptif + evaluasi model)
   - grafik_output.png   (Scatter Plot + Garis Regresi)
   - grafik_distribusi.png (Bar Chart rata-rata nilai per kategori jam belajar)
7. Mode Prediksi Interaktif (input manual oleh user)

DESKRIPSI MODUL
----------------
utils.py  : Berisi seluruh fungsi pendukung:
            - load_dataset()              : Membaca & menyiapkan data CSV
            - tampilkan_eda()             : Menampilkan info EDA ringkas
            - split_data()                : Membagi data train/test secara manual
            - latih_linear_regression()   : Menghitung koefisien OLS manual
            - prediksi()                  : Menghitung prediksi nilai
            - evaluasi_model()            : Menghitung MAE, MSE, RMSE, R²
            - simpan_hasil()              : Menyimpan statistik lengkap ke .txt
            - simpan_grafik()             : Membuat & menyimpan Scatter Plot
            - simpan_grafik_kedua()       : Membuat & menyimpan Bar Chart
            - buka_gambar()               : Membuka file gambar otomatis

main.py   : File eksekusi yang mengimpor dan memanggil fungsi dari utils.py

PANDUAN SINGKAT PENGGUNAAN
--------------------------
1. Buka terminal atau command prompt di folder proyek.
2. Jalankan perintah: python main.py
3. Tekan ENTER saat diminta untuk memuat dataset.
4. Dua grafik akan otomatis terbuka setelah proses selesai.
5. Gunakan menu interaktif: [1] untuk prediksi, [2] untuk keluar.
