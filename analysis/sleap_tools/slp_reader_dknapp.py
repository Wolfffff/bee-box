import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

with h5py.File(
    "d:\\sleap-tigergpu\\20210505_run003_00000000.avi.predictions.slp", "r"
) as f:
    keys = list(f.keys())
    print(keys)

    for key in keys:
        sleap_data = np.array(f[key])
        print(key)
        print(sleap_data.shape)
        if len(sleap_data) >= 3:
            print(sleap_data[0:3])
            print(np.asarray(sleap_data[0]).dtype)
        print("\n")

    sleap_predictions = np.array(f["pred_points"])
    sleap_instances = np.array(f["instances"])

    frames_key = []
    for c in sleap_instances:
        frames_key.append(int(c[2]))

    frames_key = np.array(frames_key)


def get_nth_instance_coords(n, sleap_predictions, sleap_instances):
    nth_inst_tuple = sleap_instances[n]
    start_index = int(nth_inst_tuple[7])
    end_index = int(nth_inst_tuple[8])

    nth_coords = np.zeros((2, end_index - start_index), dtype=float)
    for idx in range(start_index, end_index):
        prediction_tuple = sleap_predictions[idx]
        nth_coords[0, idx - start_index] = float(prediction_tuple[0])
        nth_coords[1, idx - start_index] = float(prediction_tuple[1])

    return nth_coords


def plot_all_instances_in_same_frame(
    frame, frames_key, sleap_predictions, sleap_instances, video_path
):
    frame_coords = []
    video_data = cv2.VideoCapture(video_path)
    fps = video_data.get(cv2.CAP_PROP_FPS)
    font = cv2.FONT_HERSHEY_SIMPLEX

    idcs = np.where(frames_key == frame)

    video_data.set(1, frame)
    success, image = video_data.read()

    bee_parts = {0: "tg", 1: "a", 2: "tx", 3: "h"}
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx in idcs[0]:
        print("idx: ", idx)
        nth_inst_tuple = sleap_instances[idx]
        start_index = int(nth_inst_tuple[7])
        end_index = int(nth_inst_tuple[8])
        print("start_index: ", start_index)
        print("end_index:   ", end_index)

        nth_coords = np.zeros((2, end_index - start_index), dtype=float)
        for j in range(start_index, end_index):
            prediction_tuple = sleap_predictions[j]
            nth_coords[0, j - start_index] = float(prediction_tuple[0])
            nth_coords[1, j - start_index] = float(prediction_tuple[1])
            image = cv2.circle(
                image,
                (
                    int(round(float(prediction_tuple[0]))),
                    int(round(float(prediction_tuple[1]))),
                ),
                5,
                (0, 255, 0),
                2,
            )
            # cv2.putText(image, bee_parts[j - start_index], (int(round(float(prediction_tuple[0]))) + 7, int(round(float(prediction_tuple[1]))) + 7), font, 1, (255, 255, 0), 2)
            print(
                "Drew circle! ("
                + str(prediction_tuple[0])
                + ", "
                + str(prediction_tuple[1])
                + ")"
            )

        frame_coords.append(nth_coords)

    image = cv2.resize(image, (800, 800))
    cv2.imshow("", image)

    return frame_coords


inst_coords = plot_all_instances_in_same_frame(
    700,
    frames_key,
    sleap_predictions,
    sleap_instances,
    "d:\\sleap-tigergpu\\20210505_run003_00000000.mp4",
)
cv2.waitKey(0)

print(inst_coords)
# plt.scatter(inst_coords[0], inst_coords[1])
# plt.show()

cv2.destroyAllWindows()
