# USAGE
# python detect_aruco_video.py

# import the necessary packages

import cv2
import numpy as np
import pandas as pd
from datetime import datetime

import ArUco_with_pandas_dknapp as awpd

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
#arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
#arucoParams.adaptiveThreshWinSizeStep = 1
#arucoParams.adaptiveThreshWinSizeMin = 3
#arucoParams.adaptiveThreshWinSizeMax = 30

#arucoParams.adaptiveThreshConstant = 12

#dknapp constants
# arucoParams.maxMarkerPerimeterRate = 0.06
# arucoParams.minMarkerPerimeterRate = 0.03

#arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

arucoParams.errorCorrectionRate = 1.

print("[INFO] starting video stream...")
vs = cv2.VideoCapture("d:\\sleap-tigergpu\\20210505_run003_00000000.avi")
fps = vs.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('ArUco_Labelled_20210619_2400_Test_ER10.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (800,800))

results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY'])

# loop over the frames from the video stream

start_time = datetime.now()

font = cv2.FONT_HERSHEY_SIMPLEX

frames_to_watch = 2400
bee_population = 16

tlx = 134
tly = 1050

dimension = 1750

i = 0
detected = 0
while vs.isOpened():
	ret, frame = vs.read()
	frame = frame[tlx:tlx + dimension, tly:tly + dimension]
	# detect ArUco markers in the input frame
	(corners, ids, rejected) = cv2.aruco.detectMarkers(
		frame, arucoDict, parameters=arucoParams)

	detected += len(corners)
	if len(corners) > 0:
		len(corners)
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

			frame = cv2.circle(frame, (int(round(cX)), int(round(cY))), 50, (255, 0, 0), 2)
			cv2.putText(frame, str(int(markerID[0])), (int(round(cX)), int(round(cY)) - 50), font, 2, (255, 0, 0), 2)

	i = i + 1

	frame = cv2.resize(frame, (800,800))
	out.write(frame)
	cv2.imshow('current frame', frame)
	cv2.waitKey(1)

	#Quit after a short time
	if i == frames_to_watch:
		break

	print("Frame Number: " +  str(i) + ', Total Detected Tags: ' + str(detected))

end_time = datetime.now()
delta = end_time - start_time
print('\nExecution time: ' + str(round(delta.total_seconds() * 1000)) + 'ms')
print('FPS: ' + str(round(float(delta.total_seconds()) / float(frames_to_watch), 2)) + '\n')

print('Detected total of ' + str(detected) + ' tags.')
print('This is a data density of ' + str(round(float(detected) * 100. / float(bee_population * frames_to_watch), 2)) + '%')

results_df = results_df.astype({'Frame': int,'Tag': int, 'cX': np.float64,'cY': np.float64})
results_df.to_csv("results.csv", index=False)

# do a bit of cleanup
vs.release()
out.release()
cv2.destroyAllWindows()

aruco_df = awpd.load_into_pd_dataframe('results.csv')
tags = awpd.find_tags_fast(aruco_df)
aruco_df_by_tag = awpd.sort_for_individual_tags(aruco_df, tags)

for tag in tags:
	print('\n' + '=' * 50)
	print(tag)
	print('\n' + '-' * 50)
	print(aruco_df_by_tag[tag])

print('\n\nFound the following tags (total of ' + str(len(tags)) + '):')
print(tags)