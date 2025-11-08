import numpy as np
import pandas as pd

def create_features(df, time_steps, step_size):
    x_segments, y_segments, z_segments, labels = [], [], [], []

    for i in range(0, len(df) - time_steps, step_size):
        xs = df["x-accel"].values[i:i+time_steps]
        ys = df["y-accel"].values[i:i+time_steps]
        zs = df["z-accel"].values[i:i+time_steps]
        activity = df["activity"].iloc[i:i+time_steps]
        
        # Segment boyunca aynı aktivite varsa etiketle
        if activity.nunique() == 1:
            x_segments.append(xs)
            y_segments.append(ys)
            z_segments.append(zs)
            labels.append(activity.iloc[0])

    # Özellik çıkarımı
    segments_df = pd.DataFrame({
        "x_segments": x_segments,
        "y_segments": y_segments,
        "z_segments": z_segments
    })

    feature_df = pd.DataFrame({
        "x_mean": segments_df["x_segments"].apply(np.mean),
        "y_mean": segments_df["y_segments"].apply(np.mean),
        "z_mean": segments_df["z_segments"].apply(np.mean),
        "x_std": segments_df["x_segments"].apply(np.std),
        "y_std": segments_df["y_segments"].apply(np.std),
        "z_std": segments_df["z_segments"].apply(np.std)
    })

    labels = np.asarray(labels)
    return feature_df, labels

