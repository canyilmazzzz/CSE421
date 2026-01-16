import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================
# PARAMETERS
# =====================
IMG_SIZE = 96
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 5

# =====================
# DATA GENERATORS
# =====================
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    "raw-img",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)

val_gen = datagen.flow_from_directory(
    "raw-img",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

# =====================
# MODEL DEFINITION
# =====================
base_model = MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.25,  # VERY IMPORTANT
    include_top=False,
    weights=None,  # smaller & STM32-friendly
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# =====================
# COMPILE & TRAIN
# =====================
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# =====================
# INT8 TFLITE EXPORT
# =====================


def representative_data_gen():
    for _ in range(100):
        images, _ = next(train_gen)
        yield [images.astype("float32")]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("mobilenet_animals_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ INT8 TFLite model saved as mobilenet_animals_int8.tflite")
