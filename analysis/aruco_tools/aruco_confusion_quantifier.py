# Code for tracking ArUco tags in raw videos
# Input is avi and output is labeled avi and tracks as csv

# Heavily based on: https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/

import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import aruco_utils_pd as awpd

def create_confusion_matrix(aruco_csv_path: str, output_csv_path: str, checking_window: int):
	aruco_df = awpd.load_into_pd_dataframe(aruco_csv_path)
	tags = np.sort(awpd.find_tags_fast(aruco_df))
	print(tags)

	index_dict = {}
	idx = 0
	for tag in tags:
		index_dict[tag] = idx
		idx += 1

	confusion_matrix = np.zeros((len(tags), len(tags)))

	frames_calculated = 0
	for frame in range(int(np.min(aruco_df.index)), int(np.max(aruco_df.index) - checking_window)):
		frames_calculated += 1
		window_iter = aruco_df.loc[frame:frame + checking_window].itertuples()

		for row1, row2 in [(a, b) for a in window_iter for b in window_iter]:
			if row2.Tag != row1.Tag:
				distance_squared = float(np.power(row1.cX - row2.cX, 2) + np.power(row1.cY - row2.cY, 2))
				if distance_squared != 0:
					confusion_matrix[index_dict[row1.Tag], index_dict[row2.Tag]] += 1. / distance_squared
					confusion_matrix[index_dict[row2.Tag], index_dict[row1.Tag]] += 1. / distance_squared

	confusion_matrix = confusion_matrix / frames_calculated

	np.savetxt(output_csv_path, confusion_matrix, delimiter = ',')
	# print(confusion_matrix)

	plt.imshow(confusion_matrix, norm = matplotlib.colors.LogNorm())
	plt.xticks(range(0, len(tags)), [str(int(tag)) for tag in tags])
	plt.yticks(range(0, len(tags)), [str(int(tag)) for tag in tags])
	plt.colorbar()
	plt.grid()
	plt.show()

create_confusion_matrix('d:\\20210706_run000_00000000_aruco_annotated_30p_threshold.csv', 'd:\\confusion_matrix.csv', 50)