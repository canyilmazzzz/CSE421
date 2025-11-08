from google.colab import files
uploaded = files.upload()

import pandas as pd

def read_data(file_path):
    column_names = ["user", "activity", "timestamp", "x-accel", "y-accel", "z-accel"]

    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
        sep=',',                     # Virgülle ayrılmış
        skip_blank_lines=True,
        on_bad_lines='skip'          # Hatalı satırları atla ✅
    )

    # 'z-accel' sütunundaki ';' karakterini temizle ve float'a çevir
    df["z-accel"] = df["z-accel"].astype(str).str.replace(";", "", regex=False).astype(float)

    # Eksik değerleri kaldır
    df.dropna(inplace=True)

    # Kullanıcı numarasını integer'a dönüştür
    df["user"] = df["user"].astype(str).str.extract(r'(\d+)').astype(int)

    print(f" Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df

    DATA_PATH = "WISDM_ar_v1.1_raw.txt"
data_df = read_data(DATA_PATH)
data_df.head()

