import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import platform
import subprocess

# === WARNA ANSI UNTUK TERMINAL ===
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# === FUNGSI LOAD DATASET ===
def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    kolom_jam = ['hours', 'hours_studied', 'hours studied', 'jam_belajar']
    kolom_nilai = ['scores', 'score', 'performance_index', 'performance index', 'exam_score', 'nilai']

    nama_jam   = next((k for k in kolom_jam   if k in df.columns), None)
    nama_nilai = next((k for k in kolom_nilai if k in df.columns), None)

    df = df.rename(columns={nama_jam: 'Jam_Belajar', nama_nilai: 'Nilai'})
    df = df[['Jam_Belajar', 'Nilai']].dropna()

    print(f"{C.GREEN}[OK] Dataset siap: {len(df)} baris{C.RESET}")
    return df

# === FUNGSI EDA COMPACT ===
def tampilkan_eda(df):
    korelasi = df['Jam_Belajar'].corr(df['Nilai'])
    print(f"{C.BLUE}[ EDA ]{C.RESET} Total: {len(df)} | Missing: {df.isnull().sum().sum()} | Korelasi: {korelasi:.2f}")

# === FUNGSI SPLIT DATA MANUAL ===
def split_data(df, test_size=0.2, random_state=42):
    df_acak = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df_acak) * (1 - test_size))

    df_train = df_acak.iloc[:split_idx]
    df_test  = df_acak.iloc[split_idx:]

    return df_train['Jam_Belajar'].values, df_test['Jam_Belajar'].values, df_train['Nilai'].values, df_test['Nilai'].values

# === FUNGSI PELATIHAN REGRESI LINEAR MANUAL (OLS) ===
def latih_linear_regression(X_train, y_train):
    mean_x = np.mean(X_train)
    mean_y = np.mean(y_train)
    pembilang = np.sum((X_train - mean_x) * (y_train - mean_y))
    penyebut = np.sum((X_train - mean_x) ** 2)
    slope = pembilang / penyebut
    intercept = mean_y - (slope * mean_x)
    return slope, intercept

# === FUNGSI PREDIKSI MANUAL ===
def prediksi(X, slope, intercept):
    return (slope * X) + intercept

# === FUNGSI EVALUASI MODEL MANUAL ===
def evaluasi_model(y_asli, y_prediksi):
    error = y_asli - y_prediksi
    n = len(y_asli)

    mae = np.sum(np.abs(error)) / n
    mse = np.sum(error ** 2) / n
    rmse = np.sqrt(mse)

    ss_total = np.sum((y_asli - np.mean(y_asli)) ** 2)
    r2 = 1 - (np.sum(error ** 2) / ss_total)

    print(f"{C.BLUE}[ METRIK ]{C.RESET} MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R2: {r2:.2f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# === FUNGSI SIMPAN HASIL (DIPERKAYA DENGAN STATISTIK DESKRIPTIF) ===
def simpan_hasil(df, hasil, koef, intercept, n_train, n_test, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Hitung statistik deskriptif menggunakan Pandas
    rata_jam   = df['Jam_Belajar'].mean()
    max_jam    = df['Jam_Belajar'].max()
    min_jam    = df['Jam_Belajar'].min()
    rata_nilai = df['Nilai'].mean()
    max_nilai  = df['Nilai'].max()
    min_nilai  = df['Nilai'].min()
    missing    = df.isnull().sum().sum()
    korelasi   = df['Jam_Belajar'].corr(df['Nilai'])

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 45 + "\n")
        f.write("   HASIL ANALISIS - PREDIKSI NILAI MAHASISWA\n")
        f.write("=" * 45 + "\n")
        f.write(f"Judul Proyek : Prediksi Nilai Ujian Berdasarkan Jam Belajar\n")
        f.write(f"Nama         : Alea Farrel\n")
        f.write(f"NIM          : 25.11.6401\n\n")

        f.write("--- STATISTIK DATASET ---\n")
        f.write(f"Total Data              : {len(df)} baris\n")
        f.write(f"Missing Values          : {missing}\n")
        f.write(f"Korelasi (Jam vs Nilai) : {korelasi:.4f}\n\n")

        f.write("Statistik Jam Belajar (per hari):\n")
        f.write(f"  Rata-rata  : {rata_jam:.2f} jam\n")
        f.write(f"  Tertinggi  : {max_jam:.2f} jam\n")
        f.write(f"  Terendah   : {min_jam:.2f} jam\n\n")

        f.write("Statistik Nilai Ujian:\n")
        f.write(f"  Rata-rata  : {rata_nilai:.2f}\n")
        f.write(f"  Tertinggi  : {max_nilai:.2f}\n")
        f.write(f"  Terendah   : {min_nilai:.2f}\n\n")

        f.write("--- MODEL REGRESI LINEAR ---\n")
        f.write(f"Data Split   : {n_train} Train | {n_test} Test (80:20)\n")
        f.write(f"Persamaan    : Nilai = {koef:.2f} * Jam_Belajar/Hari + {intercept:.2f}\n\n")

        f.write("Evaluasi Model:\n")
        f.write(f"  MAE   : {hasil['MAE']:.4f}  (Mean Absolute Error)\n")
        f.write(f"  MSE   : {hasil['MSE']:.4f}  (Mean Squared Error)\n")
        f.write(f"  RMSE  : {hasil['RMSE']:.4f}  (Root Mean Squared Error)\n")
        f.write(f"  R2    : {hasil['R2']:.4f}  (Koefisien Determinasi)\n\n")

        f.write("--- RATA-RATA NILAI PER KATEGORI JAM BELAJAR ---\n")
        bins   = [0, 2, 4, 6, 8, float('inf')]
        labels = ['< 2 jam', '2-4 jam', '4-6 jam', '6-8 jam', '> 8 jam']
        df_temp = df.copy()
        df_temp['Kategori'] = pd.cut(df_temp['Jam_Belajar'], bins=bins, labels=labels, right=True)
        ringkasan = df_temp.groupby('Kategori', observed=True)['Nilai'].agg(['mean', 'count'])
        for kat, row in ringkasan.iterrows():
            f.write(f"  {str(kat):10s}: Rata-rata Nilai = {row['mean']:.2f}  (n={int(row['count'])})\n")

        f.write("\n" + "=" * 45 + "\n")

    print(f"{C.GREEN}[OK] Statistik lengkap disimpan ke {path}{C.RESET}")

# === FUNGSI BUAT DAN SIMPAN GRAFIK 1: SCATTER PLOT + GARIS REGRESI ===
def simpan_grafik(X_train, y_train, slope, intercept, rmse, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    x_line = np.linspace(X_train.min(), X_train.max(), 100)
    y_line = prediksi(x_line, slope, intercept)

    plt.figure(figsize=(7, 4))

    # Grafik Dot (Scatter)
    plt.scatter(X_train, y_train, color='steelblue', alpha=0.3, s=15, label='Data Aktual')

    # Garis Regresi
    plt.plot(x_line, y_line, color='red', linewidth=2, label='Garis Regresi')

    # Batas Toleransi (Regression Line +/- RMSE)
    plt.plot(x_line, y_line + rmse, color='orange', linestyle='--', linewidth=1.5, label='Batas Toleransi (+RMSE)')
    plt.plot(x_line, y_line - rmse, color='orange', linestyle='--', linewidth=1.5, label='Batas Toleransi (-RMSE)')

    # Styling tambahan
    plt.fill_between(x_line, y_line - rmse, y_line + rmse, color='orange', alpha=0.1)

    plt.title('Prediksi Nilai Berdasarkan Jam Belajar (Per Hari)', fontsize=12, fontweight='bold')
    plt.xlabel('Jam Belajar (Per Hari)', fontsize=10)
    plt.ylabel('Nilai Ujian', fontsize=10)
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

# === FUNGSI BUAT DAN SIMPAN GRAFIK 2: BAR CHART RATA-RATA NILAI PER KATEGORI ===
def simpan_grafik_kedua(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Kategorisasi jam belajar menggunakan pd.cut (groupby aggregation)
    bins   = [0, 2, 4, 6, 8, float('inf')]
    labels = ['< 2 jam', '2-4 jam', '4-6 jam', '6-8 jam', '> 8 jam']
    df_temp = df.copy()
    df_temp['Kategori'] = pd.cut(df_temp['Jam_Belajar'], bins=bins, labels=labels, right=True)

    # Agregasi rata-rata nilai per kategori menggunakan groupby
    ringkasan = df_temp.groupby('Kategori', observed=True)['Nilai'].mean()

    # Warna berbeda untuk tiap bar
    warna = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#76b7b2']

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(ringkasan.index, ringkasan.values, color=warna, edgecolor='white', linewidth=1.2, width=0.6)

    # Tambahkan label nilai di atas tiap bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.8,
            f'{yval:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333'
        )

    ax.set_title('Rata-rata Nilai Ujian per Kategori Jam Belajar', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Kategori Jam Belajar (per hari)', fontsize=11)
    ax.set_ylabel('Rata-rata Nilai Ujian', fontsize=11)
    ax.set_ylim(0, ringkasan.max() * 1.15)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

    print(f"{C.GREEN}[OK] Grafik 2 (Bar Chart distribusi) disimpan ke {path}{C.RESET}")

# === FUNGSI BUKA GAMBAR OTOMATIS ===
def buka_gambar(path):
    sistem = platform.system()
    if sistem == 'Windows':
        os.startfile(path)
    elif sistem == 'Darwin':
        subprocess.run(['open', path])
    else:
        subprocess.run(['xdg-open', path])
