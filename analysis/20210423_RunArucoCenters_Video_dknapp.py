# USAGE
# python detect_aruco_video.py

# import the necessary packages

from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
import numpy as np
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
import pandas as pd

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--type", type=str,
# 	default="DICT_ARUCO_ORIGINAL",
# 	help="type of ArUCo tag to detect")
# args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
# if ARUCO_DICT.get(args["type"], None) is None:
# 	print("[INFO] ArUCo tag of '{}' is not supported".format(
# 		args["type"]))
# 	sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format("DICT_5X5_100"))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParams.adaptiveThreshWinSizeStep = 1
arucoParams.adaptiveThreshWinSizeMin = 3
arucoParams.adaptiveThreshWinSizeMax = 30

arucoParams.adaptiveThreshConstant = 12
print("[INFO] starting video stream...")
vs = cv2.VideoCapture("20210505_run003_00000000.avi")

results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY'])

# loop over the frames from the video stream
i = 0
while vs.isOpened():
    ret, frame = vs.read()
    print("Frame Number: ", i)
    # detect ArUco markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        frame, arucoDict, parameters=arucoParams)
    
    if len(corners) > 0:
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (topRight[0], topRight[1])
            bottomRight = (bottomRight[0], bottomRight[1])
            bottomLeft = (bottomLeft[0], bottomLeft[1])
            topLeft = (topLeft[0], topLeft[1])
            
            # Calculate centroid x a
            cX = (topLeft[0] + bottomRight[0]) / 2.0
            cY = (topLeft[1] + bottomRight[1]) / 2.0

            # 'Frame', 'Tag', 'cX','cY'
            results_df.loc[len(results_df)] = [int(i), int(markerID[0]), cX, cY]
    i = i + 1

    #Quit after a short time

    if i == 60 * 10 * 40:
        break



results_df = results_df.astype({'Frame': int,'Tag': int, 'cX': np.float64,'cY': np.float64})
results_df.to_csv("results.csv", index=False)

# do a bit of cleanup
# cv2.destroyAllWindows()
