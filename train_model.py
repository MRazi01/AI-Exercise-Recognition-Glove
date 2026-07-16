import numpy as np
import tensorflow as tf

inputs = np.load("inputs.npy")
outputs = np.load("outputs.npy")

num_inputs = len(inputs)

randomize = np.arange(num_inputs)
np.random.shuffle(randomize)

inputs = inputs[randomize]
outputs = outputs[randomize]

TRAIN_SPLIT = int(0.6 * num_inputs)
TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

inputs_train, inputs_test, inputs_validate = np.split(
    inputs,
    [TRAIN_SPLIT, TEST_SPLIT]
)

outputs_train, outputs_test, outputs_validate = np.split(
    outputs,
    [TRAIN_SPLIT, TEST_SPLIT]
)

NUM_GESTURES = outputs.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(15, activation="relu"),
    tf.keras.layers.Dense(15, activation="relu"),
    tf.keras.layers.Dense(NUM_GESTURES, activation="softmax")
])

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    inputs_train,
    outputs_train,
    epochs=1000,
    batch_size=1,
    validation_data=(inputs_validate, outputs_validate)
)

model.save("gesture_model.h5")
