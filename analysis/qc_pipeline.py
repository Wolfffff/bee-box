import glob
from natsort import natsorted
from bee_tracking import Tracking
import os
from utils.logger import logger
import pickle

base_dir = "/Genomics/ayroleslab2/scott/bees/data/"
experiment_dict = { d: {} for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,d))}


for key, value in experiment_dict.items():
    experiment_dict[key]["result_files"] = natsorted(
        glob.glob(base_dir + key + "/*/*_aruco_data_with_track_numbers.csv")
    )
    if len(experiment_dict[key]["result_files"]) == 0:
            experiment_dict[key]["result_files"] = natsorted(
        glob.glob(base_dir + key + "/*_aruco_data_with_track_numbers.csv")
    )
    if len(experiment_dict[key]["result_files"]) == 0:
        logger.info("No result files found for experiment: " + key)
    experiment_dict[key]["track"] = Tracking.fromListOfArucoFiles(experiment_dict[key]["result_files"])
    experiment_dict[key]["np_array"] = experiment_dict[key]["track"].getNumpyArray()
    
with open('experiment_data.pkl', 'wb') as file:
    pickle.dump(experiment_dict, file)