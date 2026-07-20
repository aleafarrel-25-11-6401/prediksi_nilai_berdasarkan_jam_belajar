import os
import sys

# Mencegah Python membuat folder __pycache__
sys.dont_write_bytecode = True

import utils

# Gunakan kode ANSI dari utils
C = utils.C

# --- SETUP PATH ---
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset', 'dataset.csv')
OUTPUT_TXT   = os.path.join(BASE_DIR, 'output', 'hasil_analisis.txt')
OUTPUT_PNG   = os.path.join(BASE_DIR, 'output', 'grafik_output.png')
OUTPUT_PNG2  = os.path.join(BASE_DIR, 'output', 'grafik_distribusi.png')

# === MAIN PROGRAM ===
def main():
    # Aktifkan warna ANSI di Windows terminal
    os.system('color')
    
    # === TAMPILAN AWAL (START SCREEN) ===
    print(f"\n{C.HEADER}{C.BOLD}======================================={C.RESET}")
    print(f"{C.HEADER}{C.BOLD}  SISTEM PREDIKSI NILAI MAHASISWA{C.RESET}")
    print(f"{C.HEADER}{C.BOLD}======================================={C.RESET}")
    print(f"{C.BLUE}Algoritma : Simple Linear Regression (Manual){C.RESET}")
    print(f"{C.BLUE}Variabel  : Jam Belajar -> Nilai Ujian{C.RESET}\n")
    
    input(f"{C.GREEN}{C.BOLD}Tekan [ENTER] untuk mulai memuat data...{C.RESET}")
    print("-" * 40)

    # 1. CEK DATASET
    if not os.path.exists(DATASET_PATH):
        print(f"{C.RED}Error: File dataset tidak ditemukan di '{DATASET_PATH}'!{C.RESET}")
        print(f"{C.WARNING}Pastikan Anda telah memasukkan 'dataset.csv' ke dalam folder 'Dataset'.{C.RESET}")
        sys.exit(1)

    df = utils.load_dataset(DATASET_PATH)

    # 2. EDA & DATA PREP
    utils.tampilkan_eda(df)
    X_train, X_test, y_train, y_test = utils.split_data(df, test_size=0.2)

    # 3. TRAIN & EVALUATE
    print(f"{C.BLUE}[ TRAINING ]{C.RESET} Melatih model dengan {len(X_train)} data...")
    koef, inter = utils.latih_linear_regression(X_train, y_train)
    hasil = utils.evaluasi_model(y_test, utils.prediksi(X_test, koef, inter))

    # 4. SIMPAN OUTPUT & BUKA GRAFIK
    os.makedirs(os.path.join(BASE_DIR, 'output'), exist_ok=True)

    # Simpan statistik lengkap ke .txt (menyertakan statistik deskriptif dataset)
    utils.simpan_hasil(df, hasil, koef, inter, len(X_train), len(X_test), OUTPUT_TXT)

    # Grafik 1: Scatter Plot + Garis Regresi Linear
    utils.simpan_grafik(X_train, y_train, koef, inter, hasil['RMSE'], OUTPUT_PNG)
    print(f"{C.GREEN}[OK] Grafik 1 (Scatter Plot + Garis Regresi) disimpan.{C.RESET}")

    # Grafik 2: Bar Chart rata-rata nilai per kategori jam belajar
    utils.simpan_grafik_kedua(df, OUTPUT_PNG2)

    print(f"\n{C.GREEN}{C.BOLD}Semua output berhasil disimpan di folder 'output/'.{C.RESET}")
    utils.buka_gambar(OUTPUT_PNG)
    utils.buka_gambar(OUTPUT_PNG2)

    # 5. PREDIKSI INTERAKTIF
    print(f"\n{C.HEADER}--- MODE INTERAKTIF ---{C.RESET}")
    print(f"{C.WARNING}Ketik {C.BOLD}angka{C.RESET}{C.WARNING} untuk jam belajar {C.BOLD}PER HARI{C.RESET}{C.WARNING}.{C.RESET}")
    print(f"{C.WARNING}Pilih menu: [1] Prediksi  [2] Keluar{C.RESET}")

    while True:
        menu = input(f"\n{C.BOLD}Menu (1/2): {C.RESET}").strip()
        
        if menu == '2':
            print(f"{C.GREEN}Program dihentikan. Terima Kasih...{C.RESET}")
            break
        elif menu == '1':
            inp = input("Jam belajar (per hari): ")
            try:
                jam = float(inp)
                pred = utils.prediksi(jam, koef, inter)
                
                # Gunakan fungsi bawaan Python min/max (tanpa numpy)
                batas_bawah = max(0, pred - hasil['RMSE'])
                batas_atas = min(100, pred + hasil['RMSE'])
                pred_c = min(100, max(0, pred))
                
                print(f"{C.GREEN}➜ Nilai: {pred_c:.1f}{C.RESET} {C.WARNING}(Toleransi: {batas_bawah:.1f} - {batas_atas:.1f}){C.RESET}")
            except ValueError:
                print(f"{C.RED}Error: Masukkan angka yang benar!{C.RESET}")
        else:
            print(f"{C.RED}Pilihan tidak valid.{C.RESET}")

if __name__ == '__main__':
    main()
