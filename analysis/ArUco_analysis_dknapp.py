import numpy as np
import csv
import matplotlib.pyplot as plt
import datetime
import seaborn as sns          # conda install seaborn
import cv2                     # conda install opencv-python
from tabulate import tabulate  # conda install tabulate

def load_csv(csv_filename):
	# Load ArUco CSV file and convert to numpy array
	with open(csv_filename) as csv_file:
		csv_reader = csv.reader(csv_file)

		csv_list = []

		for row in csv_reader:
			csv_list.append(row)

	return np.transpose(np.array(csv_list))

def findtags(aruco_array):
	# Return int64 numpy array of detected tags
	return np.unique(aruco_array[2,1:-1].astype('int'))

def find_tag_path(aruco_array, tag_number):
	path = [[],[]]
	aruco_array = np.transpose(aruco_array)
	for row in aruco_array[1:-1]:
		if int(row[2]) == tag_number:
			path[0].append(row[3])
			path[1].append(row[4])

	path = np.array(path).astype('float')

	return path

# Find the path corresponding to a tag, but with frame gaps included
# rows 0,1 -> (x,y) coords
# row  2   -> frame number
# row  3   -> gap since last frame
def find_tag_path_wg(aruco_array, tag_number):
	path = [[],[],[],[]]
	aruco_array = np.transpose(aruco_array)
	for row in aruco_array[1:-1]:
		if int(row[2]) == tag_number:
			path[0].append(row[3])
			path[1].append(row[4])
			path[2].append(row[1])

	path[2] = np.array(path[2]).astype('float')

	path[3] = np.zeros(len(path[2]))
	path = np.array(path).astype('float')
	path[3, 0] = 1
	for j in range(1, len(path[2])):
		path[3, j] = path[2, j] - path[2, j - 1]

	return path

def plot_bee_path(aruco_array, tag_number):
	path = find_tag_path(aruco_array, tag_number)
	the_dpi = 100
	path_fig = plt.figure(figsize=(1080/the_dpi, 1080/the_dpi), dpi = the_dpi)
	plt.plot(path[0], path[1], linewidth = 0.5)
	plt.xticks(np.arange(np.floor(min(path[0]) / 100.) * 100, np.ceil(max(path[0]) / 100.) * 100, 100.))
	plt.yticks(np.arange(np.floor(min(path[1]) / 100.) * 100, np.ceil(max(path[1]) / 100.) * 100, 100.))
	plt.gca().set_aspect('equal', 'box')
	plt.title('Path plot of bee #' + str(tag_number))
	figure_name = 'Path_Plot_' + str(tag_number) + '.png'
	plt.savefig(figure_name)
	print('Saved the plot of bee #' + str(tag_number) + "'s path as " + figure_name)

def plot_path_with_red_gaps(aruco_array, tag_number, gap_threshold):
	path_wg = find_tag_path_wg(aruco_array, tag_number)
	the_dpi = 100
	path_fig = plt.figure(figsize=(1080/the_dpi, 1080/the_dpi), dpi = the_dpi)
	for j in range(1, len(path_wg[0])):
		if path_wg[3, j] > gap_threshold:
			plt.plot([path_wg[0, j - 1], path_wg[0, j]], [path_wg[1, j - 1], path_wg[1, j]], linewidth = 0.5, color = 'r')
		else:
			plt.plot([path_wg[0, j - 1], path_wg[0, j]], [path_wg[1, j - 1], path_wg[1, j]], linewidth = 0.5, color = 'b')

	plt.xticks(np.arange(np.floor(min(path_wg[0]) / 100.) * 100, np.ceil(max(path_wg[0]) / 100.) * 100, 100.))
	plt.yticks(np.arange(np.floor(min(path_wg[1]) / 100.) * 100, np.ceil(max(path_wg[1]) / 100.) * 100, 100.))
	plt.gca().set_aspect('equal', 'box')
	plt.title('Path plot of bee #' + str(tag_number) + ' with >' + str(gap_threshold) + ' frame gaps in red')
	figure_name = 'Path_Plot_Red_Gaps_' + str(tag_number) + '.png'
	plt.savefig(figure_name)
	print('Saved the plot of bee #' + str(tag_number) + "'s path as " + figure_name)

def plot_occupancy_map(aruco_array, tag_number):
	path = find_tag_path(aruco_array, tag_number)
	the_dpi = 100
	path_fig = plt.figure(figsize=(1080/the_dpi, 1080/the_dpi), dpi = the_dpi)
	ax = sns.kdeplot(x = path[0], y = path[1], cmap = 'Reds', shade = True, thresh = 0.001)
	plt.xticks(np.arange(np.floor(min(path[0]) / 100.) * 100, np.ceil(max(path[0]) / 100.) * 100, 100.))
	plt.yticks(np.arange(np.floor(min(path[1]) / 100.) * 100, np.ceil(max(path[1]) / 100.) * 100, 100.))
	plt.gca().set_aspect('equal', 'box')
	plt.title('Occupancy map bee #' + str(tag_number))
	figure_name = 'Occupancy_map_' + str(tag_number) + '.png'
	plt.savefig(figure_name)
	print('Saved the plot of bee #' + str(tag_number) + "'s occupancy map as " + figure_name)

def plot_error_map(aruco_array, tag_number, gap_threshold):
	path_wg = find_tag_path_wg(aruco_array, tag_number)
	the_dpi = 100
	path_fig = plt.figure(figsize=(1080/the_dpi, 1080/the_dpi), dpi = the_dpi)

	# Go through and divide greater-than-1-frame gaps into points along it
	gaps = [[], []]
	for j in range(1, len(path_wg[0])):
		if path_wg[3, j] > gap_threshold:
			gap_length = np.sqrt(np.power(path_wg[0, j] - path_wg[0, j - 1], 2) + np.power(path_wg[1, j] - path_wg[1, j - 1], 2))
			gap_increment = gap_length / path_wg[3, j]
			gap_angle = np.arctan2(path_wg[1, j] - path_wg[1, j - 1], path_wg[0, j] - path_wg[0, j - 1])

			for k in range(0, int(path_wg[3, j])):
				gaps[0].append(float(path_wg[0, j] + gap_increment * np.cos(gap_angle)))
				gaps[1].append(float(path_wg[1, j] + gap_increment * np.sin(gap_angle)))

	gaps = np.array(gaps)
	ax = sns.kdeplot(x = gaps[0], y = gaps[1], cmap = 'Reds', shade = True, thresh = 0.001)
	plt.xticks(np.arange(np.floor(min(gaps[0]) / 100.) * 100, np.ceil(max(gaps[0]) / 100.) * 100, 100.))
	plt.yticks(np.arange(np.floor(min(gaps[1]) / 100.) * 100, np.ceil(max(gaps[1]) / 100.) * 100, 100.))
	plt.gca().set_aspect('equal', 'box')
	plt.title('ArUco predicted missing frames map, bee #' + str(tag_number))
	figure_name = 'error_map_' + str(tag_number) + '.png'
	plt.savefig(figure_name)
	print('Saved the plot of bee #' + str(tag_number) + "'s distribution of errors as " + figure_name)

def show_gap_starts(aruco_array, video_path, tag_number, gap_threshold):
	video_data = cv2.VideoCapture(video_path)
	fps = video_data.get(cv2.CAP_PROP_FPS)
	font = cv2.FONT_HERSHEY_SIMPLEX
	success, image = video_data.read()
	h, w, c = image.shape

	path_wg = find_tag_path_wg(aruco_array, tag_number)
	for j in range(1, len(path_wg[0])):

		if path_wg[3, j] > gap_threshold:
			video_data.set(1, int(path_wg[2, j - 1]))
			success, image = video_data.read()
			
			if success == True:
				image = cv2.circle(image, (int(path_wg[0, j - 1]), int(path_wg[1, j - 1])), 120, (0, 0, 0), 2)
				image = cv2.resize(image, (800,800))
				timestamp = str(datetime.timedelta(seconds = np.round(video_data.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)))
				cv2.rectangle(image, (0, 800), (800, 780), (255, 255, 255), -1)
				cv2.putText(image, 'Timestamp: ' + timestamp + ', Frame: ' + str(int(path_wg[2, j - 1])) + ', Gap length: ' + str(int(path_wg[3, j])) \
					+ '  (~' + str(np.round(path_wg[3, j] * 10. / fps) / 10.) + ' seconds @' + str(fps) + ' fps)', (10, 794), font, 0.5, (0, 0, 0), 1)
				cv2.imshow('Bee #' + str(tag_number) + ', beginning frames of ArUco data gaps longer than ' + str(gap_threshold) \
				 + ' frames; press esc to leave', image)
				if cv2.waitKey(0) == 27:
				    break
				else:
				    continue

	cv2.destroyAllWindows()

def collect_all_bee_stats(aruco_array, video_path):
	video_data = cv2.VideoCapture(video_path)
	fps = video_data.get(cv2.CAP_PROP_FPS)
	total_frames = float(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
	tags = findtags(aruco_array)
	longest_skips = []
	g1s_skips = []
	g10s_skips = []
	g100s_skips = []
	captured_percents = []

	for tag_number in tags:
		path_wg = find_tag_path_wg(aruco_array, tag_number)
		skips_data = np.array(path_wg[3]).astype('int')

		longest_skips.append(np.max(skips_data))

		g1s_skips.append(len([skip for skip in skips_data if skip > fps]))
		g10s_skips.append(len([skip for skip in skips_data if skip > fps * 10]))
		g100s_skips.append(len([skip for skip in skips_data if skip > fps * 100]))

		captured_percents.append(float(np.sum(skips_data)) * 100. / total_frames)

	return tags, longest_skips, g1s_skips, g10s_skips, g100s_skips, captured_percents

def display_bee_stats_table(aruco_array, video_path):
	tags, longest_skips, g1s_skips, g10s_skips, g100s_skips, captured_percents = collect_all_bee_stats(aruco_array, video_path)

	longest_skips = [str(j) for j in longest_skips]
	longest_skips.insert(0, ' Longest gaps')

	g1s_skips = [str(j) for j in g1s_skips]
	g1s_skips.insert(0, ' >   1 second gaps')

	g10s_skips = [str(j) for j in g10s_skips]
	g10s_skips.insert(0, ' >  10 second gaps')

	g100s_skips = [str(j) for j in g100s_skips]
	g100s_skips.insert(0, ' > 100 second gaps')

	captured_percents = [str(round(j, 1)) for j in captured_percents]
	captured_percents.insert(0, '% frames with data')

	table = [captured_percents, longest_skips, g1s_skips, g10s_skips, g100s_skips]

	print(tabulate(table, headers = tags))



# 
aruco = load_csv('20210506_12h_Tracked_DefaultArUcoParams.csv')


#tags = findtags(aruco)
#print('I found the following tags:')
#print(tags)

#plot_bee_path(aruco, 45)
#plot_path_with_red_gaps(aruco, 45, 50)

#plot_occupancy_map(aruco, 45)
#plot_error_map(aruco, 45, 50)

#show_gap_starts(aruco, '20210505_run003_00000000.avi', 45, 2000)

display_bee_stats_table(aruco, '20210505_run003_00000000.avi')