import numpy as np
import pandas as pd

GESTURES = [
    "punch",
    "down",
    "left_hook",
    "right_hook",
    "uppercut"
]

SAMPLES_PER_GESTURE = 119

NUM_GESTURES = len(GESTURES)
ONE_HOT = np.eye(NUM_GESTURES)

inputs = []
outputs = []

for gesture_index in range(NUM_GESTURES):

    gesture = GESTURES[gesture_index]
    output = ONE_HOT[gesture_index]

    df = pd.read_csv(f"{gesture}.csv")

    num_recordings = int(df.shape[0] / SAMPLES_PER_GESTURE)

    for i in range(num_recordings):

        tensor = []

        for j in range(SAMPLES_PER_GESTURE):

            index = i * SAMPLES_PER_GESTURE + j

            tensor += [
                (df['aX'][index] + 4) / 8,
                (df['aY'][index] + 4) / 8,
                (df['aZ'][index] + 4) / 8,
                (df['gX'][index] + 2000) / 4000,
                (df['gY'][index] + 2000) / 4000,
                (df['gZ'][index] + 2000) / 4000
            ]

        inputs.append(tensor)
        outputs.append(output)

inputs = np.array(inputs)
outputs = np.array(outputs)

np.save("inputs.npy", inputs)
np.save("outputs.npy", outputs)

print("Preprocessing Complete")
