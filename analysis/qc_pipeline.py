import glob
from natsort import natsorted
from bee_tracking import Tracking

base_dir = "/Genomics/ayroleslab2/scott/bees/data/"

experiment_dict = {"20210909_run000_00000000": {}}

for key, value in experiment_dict.items():
    experiment_dict[key]["result_files"] = natsorted(
        glob.glob(base_dir + key + "/*/*_aruco_data_with_track_numbers.csv")
    )
track = Tracking.fromListOfArucoFiles(experiment_dict[key]["result_files"])
