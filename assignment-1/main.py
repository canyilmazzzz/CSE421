from data_utils import read_data
from feature_utils import create_features


TIME_PERIODS = 80
STEP_DISTANCE = 40

df_train = data_df[data_df["user"] <= 28]
df_test = data_df[data_df["user"] > 28]

train_features, train_labels = create_features(df_train, TIME_PERIODS, STEP_DISTANCE)
test_features, test_labels = create_features(df_test, TIME_PERIODS, STEP_DISTANCE)

print("Train samples:", len(train_features))
print("Test samples:", len(test_features))
