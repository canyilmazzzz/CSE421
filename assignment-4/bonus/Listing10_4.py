import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

w = tf.keras.initializers.Constant([[0.5], [-0.5]])
b = tf.keras.initializers.Constant([0.0])

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            units=1,
            input_shape=(2,),
            kernel_initializer=w,
            bias_initializer=b,
            activation="sigmoid",
        )
    ]
)

x_grid, y_grid = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))

Z = model(np.c_[x_grid.ravel(), y_grid.ravel()], training=False).numpy()

Z = Z.reshape(x_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(x_grid, y_grid, Z, cmap=cm.coolwarm)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("σ(0.5x − 0.5y)")
plt.colorbar(surf)
plt.show()
