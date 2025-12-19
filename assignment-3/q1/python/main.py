import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from Data.paths import TEMPERATURE_DATA_PATH
from Models.paths import REGRESSION_MODEL_DIR

df = pd.read_csv(TEMPERATURE_DATA_PATH, sep=r"\s+", comment="#", header=None)

df.columns = [
    "Date",
    "Time",
    "Temperature_Comedor",
    "Temperature_Habitacion",
    "Weather_Temperature",
    "CO2_Comedor",
    "CO2_Habitacion",
    "Humedad_Comedor",
    "Humedad_Habitacion",
    "Lighting_Comedor",
    "Lighting_Habitacion",
    "Precipitacion",
    "Meteo_Crepusculo",
    "Meteo_Viento",
    "Sol_Oest",
    "Sol_Est",
    "Sol_Sud",
    "Piranometro",
    "Entalpic_1",
    "Entalpic_2",
    "Entalpic_Turbo",
    "Temperature_Exterior",
    "Humedad_Exterior",
    "Day_Of_Week",
]

y = df["Temperature_Habitacion"][::4]

prev_values_count = 5

X = pd.DataFrame()
for i in range(prev_values_count, 0, -1):
    X["t-" + str(i)] = y.shift(i)

X = X[prev_values_count:]
y = y[prev_values_count:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_predicted = linear_model.predict(X_train)
y_test_predict = linear_model.predict(X_test)

fig, ax = plt.subplots(1, 1)
ax.plot(y_test.to_numpy(), label="Actual values")
ax.plot(y_test_predict, label="Predicted values")
plt.legend()
plt.show()

mae_train = np.sqrt(mean_absolute_error(y_train, y_train_predicted))
mae_test = np.sqrt(mean_absolute_error(y_test, y_test_predict))

print(f"Training set MAE: {mae_train}\n")
print(f"Test set MAE:{mae_test}")

dump(linear_model, os.path.join(REGRESSION_MODEL_DIR, "temperature_pred_linreg.joblib"))
