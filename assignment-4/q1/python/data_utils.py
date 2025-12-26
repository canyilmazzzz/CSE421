import pandas as pd
import numpy as np

def read_data(file_path):
    column_names = [
        "user",
        "activity",
        "timestamp",
        "x-accel",
        "y-accel",
        "z-accel"
    ]

    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
        engine="python",
        on_bad_lines="skip"
    )

    df["z-accel"] = (
        df["z-accel"]
        .astype(str)
        .str.replace(";", "", regex=False)
        .replace("None", np.nan) # Added this line to handle 'None' strings
        .astype(float)
    )

    df.dropna(inplace=True)

    print("Number of columns:", df.shape[1])
    print("Number of rows:", df.shape[0])

    return df

# Original content of MSFp2GSirEh5 follows
from data_utils import read_data

df = read_data("WISDM_ar_v1.1_raw.txt")
