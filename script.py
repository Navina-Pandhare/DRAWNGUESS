import keras

model = keras.models.load_model("doodle_model.h5")
print("Model Input Shape:", model.input_shape)