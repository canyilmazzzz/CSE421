import numpy as np
import pandas as pd

def create_features(df, time_steps, step_size):

    x_segments = []
    y_segments = []
    z_segments = []
    labels = []

    for i in range(0, len(df) - time_steps, step_size):

        xs = df["x-accel"].values[i:i + time_steps]
        ys = df["y-accel"].values[i:i + time_steps]
        zs = df["z-accel"].values[i:i + time_steps]

        label_counts = df["activity"][i:i + time_steps].value_counts()

        if label_counts.iloc[0] == time_steps:
            label = label_counts.index[0]
            x_segments.append(xs)
            y_segments.append(ys)
            z_segments.append(zs)
            labels.append(label)

    segments_df = pd.DataFrame({
        "x_segments": x_segments,
        "y_segments": y_segments,
        "z_segments": z_segments
    })

    feature_df = pd.DataFrame()

    # mean
    feature_df["x_mean"] = segments_df["x_segments"].apply(lambda x: x.mean())
    feature_df["y_mean"] = segments_df["y_segments"].apply(lambda x: x.mean())
    feature_df["z_mean"] = segments_df["z_segments"].apply(lambda x: x.mean())

    # positive count
    feature_df["x_pos_count"] = segments_df["x_segments"].apply(lambda x: np.sum(x > 0))
    feature_df["y_pos_count"] = segments_df["y_segments"].apply(lambda x: np.sum(x > 0))
    feature_df["z_pos_count"] = segments_df["z_segments"].apply(lambda x: np.sum(x > 0))

    # FFT features
    FFT_SIZE = time_steps // 2 + 1

    x_fft = segments_df["x_segments"].apply(lambda x: np.abs(np.fft.fft(x))[1:FFT_SIZE])
    y_fft = segments_df["y_segments"].apply(lambda x: np.abs(np.fft.fft(x))[1:FFT_SIZE])
    z_fft = segments_df["z_segments"].apply(lambda x: np.abs(np.fft.fft(x))[1:FFT_SIZE])

    feature_df["x_std_fft"] = x_fft.apply(lambda x: x.std())
    feature_df["y_std_fft"] = y_fft.apply(lambda x: x.std())
    feature_df["z_std_fft"] = z_fft.apply(lambda x: x.std())

    feature_df["sma_fft"] = (
        x_fft.apply(lambda x: np.sum(np.abs(x) / 50)) +
        y_fft.apply(lambda x: np.sum(np.abs(x) / 50)) +
        z_fft.apply(lambda x: np.sum(np.abs(x) / 50))
    )

    labels = np.array(labels)

    return feature_df, labels

