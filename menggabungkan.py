import os
import glob
import pandas as pd

# Mengatur direktori kerja
os.chdir("./csv/")

# Mencari semua file csv di jalur
extension = "csv"
all_filenames = [i for i in glob.glob("*.{}".format(extension))]

# Menggabungkan semua file dalam daftar
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

# Ekspor ke csv
combined_csv.to_csv("data_training.csv", index=False, encoding="utf-8-sig")
