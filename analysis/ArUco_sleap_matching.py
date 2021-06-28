import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import ArUco_with_pandas_dknapp as awpd

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

def ArUco_cleaner(video_path: str, frame_array: list, tl_coords: tuple[int, int], square_dim: int, bee_population: int) -> 'DataFrame':
	"""Processes a video for ArUco tags, attempting to iteratively lower the error correction to flush out false positives and misreadings.	"""

	# load the ArUCo dictionary and grab the ArUCo parameters
	print("[INFO] detecting '{}' tags...".format("DICT_5X5_100"))
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])
	arucoParams = cv2.aruco.DetectorParameters_create()
	arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
	arucoParams.adaptiveThreshWinSizeStep = 1
	arucoParams.adaptiveThreshWinSizeMin = 3
	arucoParams.adaptiveThreshWinSizeMax = 30
	arucoParams.adaptiveThreshConstant = 12
	arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

	vs = cv2.VideoCapture(video_path)
	fps = vs.get(cv2.CAP_PROP_FPS)
	results_df = pd.DataFrame(columns = ['Frame', 'Tag', 'cX','cY'])

	detected = 0

	# Start by casting a wide net
	arucoParams.errorCorrectionRate = 1.

	for frame_idx in awpd.progressBar(frame_array):
		# Navigate to relevant frame in video
		vs.set(1, frame_idx)
		ret, frame = vs.read()

		# In-built cropping feature; cropping video is slow so best to just crop the frames we need, as needed
		frame = frame[tl_coords[0] : tl_coords[0] + square_dim, tl_coords[1] : tl_coords[1] + square_dim]


		# Detect ArUco markers in the input frame
		(corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters = arucoParams)

		# Sort out data into dataframe for easy access
		detected += len(corners)
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
				cX = (topLeft[0] + bottomRight[0]) / 2.
				cY = (topLeft[1] + bottomRight[1]) / 2.

				# 'Frame', 'Tag', 'cX','cY'
				results_df.loc[len(results_df)] = [int(frame_idx), int(markerID[0]), cX, cY]

	# Sort out which tags have been detected; sort by amount of data
	tags = awpd.find_tags_fast(results_df)
	aruco_df_by_tag = awpd.sort_for_individual_tags(results_df, tags)

	tag_lengths = {}
	for tag in tags:
		tag_lengths[int(tag)] = len(aruco_df_by_tag[tag].index)

	print(tag_lengths)
	print('Total of ' + str(len(tags)) + ' tags detected in first pass')

	pre_suspicion_data = len(results_df.index)

	
	suspicious_index = 0
	sorted_tag_lengths = sorted(tag_lengths.items(), key = lambda item: item[1])
	if len(tags) > bee_population:
		while suspicious_index < len(tags):
			print('\n\nToo many tags... ' + str(len(tags)) + '/' + str(bee_population))
			suspicious_tag = sorted_tag_lengths[suspicious_index][0]
			print("I'm suspicious of tag " + str(suspicious_tag))
			suspicious_frames = aruco_df_by_tag[suspicious_tag]['Frame'].tolist()
			print("Suspicious frames: " + str(suspicious_frames))

			# Start by getting rid of data from suspicious frames.
			for j in suspicious_frames:
				results_df = results_df[results_df.Frame != j]
			aruco_df_by_tag = awpd.sort_for_individual_tags(results_df, tags)
			suspicious_index += 1

			for frame_idx in suspicious_frames:
				arucoParams.errorCorrectionRate = 1.
				while arucoParams.errorCorrectionRate > 0.1:
					vs.set(1, frame_idx)
					ret, frame = vs.read()
					frame = frame[tl_coords[0] : tl_coords[0] + square_dim, tl_coords[1] : tl_coords[1] + square_dim]
					(corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters = arucoParams)

					if suspicious_tag not in ids:
						break
					else:
						arucoParams.errorCorrectionRate -= 0.2
				print('Refinished frame ' + str(frame_idx) + ' with errorCorrectionRate = ' + str(round(arucoParams.errorCorrectionRate, 1)))
				print('Tags identified after refinishing: ' + str(np.sort(ids.flatten())))

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
						cX = (topLeft[0] + bottomRight[0]) / 2.
						cY = (topLeft[1] + bottomRight[1]) / 2.

						# 'Frame', 'Tag', 'cX','cY'
						results_df.loc[len(results_df)] = [frame_idx, int(markerID[0]), cX, cY]

			tags = awpd.find_tags_fast(results_df)
			print(tags)
			if len(tags) <= bee_population:
				print('Bee population now matched... terminating!')
				print(results_df)
				break
	post_suspicion_data = len(results_df.index)
	print('Pre -suspicion data amount:', pre_suspicion_data)
	print('Post-suspicion data amount:', post_suspicion_data)
	print('Lost data amount: ', post_suspicion_data - pre_suspicion_data)

	return results_df


results_df = ArUco_cleaner('d:\\sleap-tigergpu\\20210505_run003_00000000.mp4', range(0, 1000), (0, 0), 2063, 16)
results_df.to_csv("results.csv", index = False)
tags = awpd.find_tags_fast(results_df)
aruco_df_by_tag = awpd.sort_for_individual_tags(results_df, tags)
awpd.print_by_tag_data(tags, aruco_df_by_tag)


#aruco_df_by_tag.to_csv("results_by_tag.csv", index = False)


#def Generate_ArUco_Preferences(aruco_df_by_tag, )