import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from munkres import Munkres

def sleap_reader(slp_predictions_path: str) -> h5py.File:
	f = h5py.File(slp_predictions_path, 'r')
	keys = list(f.keys())
	print(keys)

	for key in keys:
		sleap_data = np.array(f[key])
		print(key)
		print(sleap_data.shape)
		if len(sleap_data) >= 3:
			print(sleap_data[0:3])
			print(np.asarray(sleap_data[0]).dtype)
		print('\n')

	return f


def import_sleap_instances_df(slp_predictions_path: str, fps: float) -> pd.DataFrame:
	sleap_file = sleap_reader(slp_predictions_path)
	sleap_predictions = np.array(sleap_file['pred_points'])
	sleap_instances   = np.array(sleap_file['instances'])
	sleap_frames = np.array(sleap_file['frames'])

	bee_movements_list = [] # List of tuples to append df rows later; appending to a list is (presumably) faster

	for frame in sleap_frames:
		frame_number = int(frame[0])

		inst_start_idx = int(frame[3])
		inst_end_idx = int(frame[4])
		for instance_idx in range(inst_start_idx, inst_end_idx):
			current_instance = sleap_instances[instance_idx]
			track_number = int(current_instance[4])
			score = float(current_instance[6])

			pred_start_idx = int(current_instance[7])
			pred_end_idx = int(current_instance[8])
			
			# Check if the tag is available otherwise, ignore this point
			tag_prediction = sleap_predictions[pred_start_idx]
			tagX = float(tag_prediction[0])
			tagY = float(tag_prediction[1])

			head_prediction = sleap_predictions[pred_start_idx + 3]
			headX = float(head_prediction[0])
			headY = float(head_prediction[1])

			abdomen_prediction = sleap_predictions[pred_start_idx + 2]
			abdomenX = float(abdomen_prediction[0])
			abdomenY = float(abdomen_prediction[1])

			headAbdomenAngle = np.arctan2(abdomenY - headY, abdomenX - headX)

			if ((not np.isnan(tagX) and not np.isnan(tagY)) and not np.isnan(headAbdomenAngle)) and not np.isnan(score):
				# For now the speeds are left empty; we'll fill them in later

				bee_movements_list.append((frame_number, track_number, score, tagX, tagY, headAbdomenAngle, 0., 0.))

	bee_movements_df = pd.DataFrame(bee_movements_list, columns = ['Frame', 'Track', 'Score', 'tagX', 'tagY', 'headAbdomenAngle', 'tagSpeed', 'headAbdomenRotVel'])
	print(bee_movements_df)

	tracks = np.sort(pd.unique(bee_movements_df['Track']))
	print('\n\nTags: ', tracks)
	print('\n\nCalculating speeds ...\n\n')

	for track in tracks:
		current_track_df = bee_movements_df.loc[bee_movements_df['Track'] == track]

		previous_frame = -2
		previous_row = current_track_df.iloc[0]
		for row in current_track_df.itertuples():
			if row.Frame - previous_frame == 1:
				tag_speed = fps * np.sqrt(np.power(row.tagX - previous_row.tagX, 2) + np.power(row.tagY - previous_row.tagY, 2))
				rot_speed = fps * (row.headAbdomenAngle - previous_row.headAbdomenAngle)

				bee_movements_df.iloc[row.Index, 6] = tag_speed
				bee_movements_df.iloc[row.Index, 7] = rot_speed

			previous_frame = row.Frame
			previous_row = row

	print(bee_movements_df)
	return bee_movements_df


def tag_speeds_histogram(bee_movements_df: pd.DataFrame) -> None:
	speeds = bee_movements_df['tagSpeed'].to_numpy()
	speeds = np.ma.masked_less(speeds, 0.001)
	# speeds = np.ma.masked_greater(speeds, 10000)
	speeds = speeds.compressed()
	# q25, q75 = np.percentile(speeds, [.25, .75])
	# bin_width = 2 * (q75 - q25) * np.power(float(len(speeds)), -(1./3.))
	# bins = round((np.max(speeds) - np.min(speeds)) / bin_width)
	# plt.hist(np.log10(speeds), bins = 100)
	plt.hist(speeds / 20., bins = 10000)
	plt.xlim([0, 50])
	plt.ylim([0, 150])
	# plt.yscale('log', nonpositive = 'clip')
	plt.show()

def real_tag_speed_confidence(bee_movements_df: pd.DataFrame, test_speed: float):
	speeds = bee_movements_df['tagSpeed'].to_numpy()
	total_speeds = float(len(speeds))
	speeds = np.ma.masked_less(speeds, test_speed)
	speeds = speeds.compressed()

	return float(len(speeds)) / total_speeds

def real_rotation_speed_confidence(bee_movements_df: pd.DataFrame, test_speed: float):
	speeds = np.abs(bee_movements_df['headAbdomenRotVel'].to_numpy())
	total_speeds = float(len(speeds))
	speeds = np.ma.masked_less(speeds, np.abs(test_speed))
	speeds = speeds.compressed()

	return float(len(speeds)) / total_speeds


def sigmoid(input_value: float, center: float, slope: float):
	'''
	sigmoid = 1 / (1 + exp[-slope * {input - centur}])
	'''
	return 1 / (1. + np.exp(-slope * (input_value - center)))


def merge_sleap_tracklets(slp_predictions_path: str, output_path: str, bee_movements_df: pd.DataFrame, pairings: list) -> list:
	m = Munkres()
	frame_dfs = []

	# Variables to avoid running cost analysis on first trigger
	first_trigger = True

	frame = 0
	for row in np.transpose(pairings):
		# Initialize variables relevant to this frame that might need to be used later on
		this_frame_tracks = row
		current_frame_df = bee_movements_df.loc[bee_movements_df['Frame'] == frame]
		frame_dfs.append(current_frame_df)

		# We don't care about consecutive frames with consistent pairings;
		# Only run matching when pairings change.
		# Obviously, we can't run tracklet merging on frame zero; there's nothing to start merging from.
		if frame > 0 and not np.array_equal(this_frame_tracks, last_frame_tracks):
			print('\n' + '=' * 80)
			print(f'Frame {frame}:\n')

			# Remove zeros from arrays; these correspond to unpaired tags
			trimmed_lft = np.ma.masked_equal(np.unique(last_frame_tracks), 0)
			trimmed_lft = trimmed_lft.compressed()
			trimmed_tft = np.ma.masked_equal(np.unique(this_frame_tracks), 0)
			trimmed_tft = trimmed_tft.compressed()
			tft_length = len(trimmed_tft)
			lft_length = len(trimmed_lft)

			print(f'({lft_length}, {tft_length})')
			if tft_length == lft_length:
				print('No tracks gained or lost!')
			elif tft_length > lft_length:
				print('New track out of nowhere!')
			else:
				print("We've lost a track :(")
			
			# Carry over any lost tracks
			trimmed_tft = np.append(trimmed_tft, trimmed_lft)
			trimmed_tft = np.unique(trimmed_tft)
			tft_length = len(trimmed_tft)
			lft_length = len(trimmed_lft)

			print('Old tracks:  ', trimmed_lft)
			print('New tracks:  ', trimmed_tft)

			# Iterate through all track combinations.
			# We want to check if the same track number is actually the same track too
			# So we calculate a cost, even for a track to itself.

			# Initialize cost matrix
			cost_matrix = np.zeros((lft_length, tft_length))

			# Initialize dicts for track indices
			lft_indices = {}
			for idx in range(len(trimmed_lft)):
				lft_indices[trimmed_lft[idx]] = idx

			tft_indices = {}
			for idx in range(len(trimmed_tft)):
				tft_indices[trimmed_tft[idx]] = idx

			for lf_track in trimmed_lft:
				# Try to find entry in bee movements dataframe
				try:
					last_frame_df = frame_dfs[frame - 1]
					lf_track_row = last_frame_df.loc[last_frame_df['Track'] == lf_track]
					lf_track_X = float(lf_track_row['tagX'])
					lf_track_Y = float(lf_track_row['tagY'])
					lf_success = True
				except:
					print(f'On frame {frame - 1} could not retrieve data for track {int(lf_track)}')
					lf_success = False

				for tf_track in trimmed_tft:
					try:
						this_frame_df = frame_dfs[frame]
						tf_track_row = this_frame_df.loc[this_frame_df['Track'] == tf_track]
						tf_track_X = float(tf_track_row['tagX'])
						tf_track_Y = float(tf_track_row['tagY'])
						tf_success = True
					except:
						print(f'On frame {frame} could not retrieve data for track {int(lf_track)}')
						tf_success = False

					# If we could find data for both this frame AND last frame for this track transition combination
					if lf_success and tf_success:

						# Distance cost calculation
						distance = np.sqrt(np.power(tf_track_X - lf_track_X, 2) + np.power(tf_track_Y - lf_track_Y, 2))

						# We use the previously generated dicts to help index on the cost matrix
						cost_matrix[lft_indices[lf_track], tft_indices[tf_track]] += (100.) * sigmoid(distance, 50., 0.1)

					# Penalty for changing tracks
					if tf_track != lf_track:
						cost_matrix[lft_indices[lf_track], tft_indices[tf_track]] += 30.


			# Hungarian matching
			if not first_trigger:
				print(tabulate(cost_matrix.astype('int'), tablefmt = 'pretty'))
				hungarian_result = m.compute(np.copy(cost_matrix))
				print('Chosen pairs:               ', hungarian_result)
				hungarian_pairs = {}
				for lf_track, tf_track in hungarian_result:
					hungarian_pairs[int(trimmed_lft[lf_track])] = int(trimmed_tft[tf_track])
				print('Resultant track transitions:', hungarian_pairs)

				# Change necessary pairings
				frame -= 1
				print('Previous assignments:       ', str(pairings[:, frame]))
				for idx in range(len(pairings[:, frame])):
					if pairings[idx, frame] != 0:
						pairings[idx, frame] = hungarian_pairs[int(pairings[idx, frame])]
				print('Modified assignments:       ', str(pairings[:, frame]))
				frame += 1

			else:
				first_trigger = False

		# Manage history-dependent variables
		last_frame_tracks = np.copy(this_frame_tracks)
		frame += 1

	np.savetxt(output_path, pairings, delimiter = ",")












	# sleap_file = sleap_reader(slp_predictions_path)
	# sleap_predictions = np.array(sleap_file['pred_points'])
	# sleap_instances   = np.array(sleap_file['instances'])
	# sleap_frames = np.array(sleap_file['frames'])

	
	

	# unique_tracks = np.sort(np.unique([j[4] for j in sleap_instances]))
	
	# # Dictionary of the indices of tags
	# track_indices = {}
	# for idx in range(len(unique_tracks)):
	# 	track_indices[unique_tracks[idx]] = idx

	# cumulative_track_lengths = np.zeros(len(unique_tracks))

	# print('\n')
	# print('Unique tracks:              ', unique_tracks)
	# print('Unique tags:                ', tags)

	# last_frame_aruco = np.max(aruco_df.index)
	# last_frame_sleap = sleap_instances[-1][2]
	# last_frame = int(np.min([last_frame_aruco, last_frame_sleap]))
	# print('Last frame with ArUco data: ', last_frame_aruco)
	# print('Last frame with SLEAP data: ', last_frame_sleap)
	# print('Overall last frame:         ', last_frame)
	# print('\n')
	# print('Starting tag - track distance computations ...')

	# instances_idx = 0
	# # Initialization code for last_frame_tracks
	# current_frame_instances = []
	# current_sleap_frame = sleap_frames[0] 
	# for current_frame_idx in range(current_sleap_frame[3], current_sleap_frame[4]): # range(instance_id_start, instance_id_end)
	# 	current_frame_instances.append(sleap_instances[current_frame_idx])

	# last_frame_tracks = []
	# chunk_end_frames = []
	# hungarian_pairs = []

	# m = Munkres()
	# for frame in sleap_frames:
	# 	current_frame = int(frame[0])

	# 	# Break if we've reached the end
	# 	if current_frame == last_frame:
	# 		break

	# 	current_frame_df = bee_movements_df.loc[frame]
	# 	frame_dfs.append(current_frame_df)
		
	# 	# Put instances for current frame in a list
	# 	# Technically, I don't think we need to put together this list... we could combine this loop with the next ones
	# 	# Right now though, this just makes it easier to debug
	# 	# TODO: combine this loop and the next ones
	# 	current_frame_instances = []
	# 	current_sleap_frame = sleap_frames[current_frame] 
	# 	for current_frame_idx in range(current_sleap_frame[3], current_sleap_frame[4]): # range(instance_id_start, instance_id_end)
	# 		current_frame_instances.append(sleap_instances[current_frame_idx])

	# 	# Check if the tracks on this frame are the same.
	# 	# If they change, run Hungarian algorithm on what we have so far, and start a new chunk
	# 	this_frame_tracks = []
	# 	for instance in current_frame_instances:
	# 		this_frame_tracks.append(instance[4]) # track

	# 	this_frame_tracks = np.sort(this_frame_tracks)

	# 	for track in this_frame_tracks:
	# 		cumulative_track_lengths[track_indices[track]] += 1

	# 	if not np.array_equal(this_frame_tracks, last_frame_tracks):
	# 		if len(last_frame_tracks) == 0:
	# 			print('\n' + '=' * 80)
	# 			mini_cost_matrix = np.zeros((len(this_frame_tracks), len(this_frame_tracks)))
	# 			print('First frame.')


	# 		else:
	# 			print('\n' + '=' * 80)

	# 			# Calculate cost matrix for transitions
	# 			lft_length = len(last_frame_tracks)
	# 			tft_length = len(this_frame_tracks)
	# 			mini_cost_matrix = np.zeros((lft_length, tft_length))
	# 			if tft_length == tft_length:
	# 				print('No tracks gained or lost!')
	# 			elif tft_length > lft_length:
	# 				print('New track out of nowhere!')
	# 			else:
	# 				print('We lost a track :(')

	# 			# Prep dictionaries for the indices of specific tracks in cost matrix
	# 			# Both for the last consistent run of tracks, and the upcoming one.
	# 			lf_track_indices = {}
	# 			for idx in range(len(last_frame_tracks)):
	# 				lf_track_indices[last_frame_tracks[idx]] = idx

	# 			tf_track_indices = {}
	# 			for idx in range(len(this_frame_tracks)):
	# 				lf_track_indices[this_frame_tracks[idx]] = idx

	# 			# Iterate through all track combinations.
	# 			# We want to check if the same track number is actually the same track too
	# 			# So we calculate a cost, even for a track to itself.
	# 			# This also serves as a good POC that this algorithm works as well!
	# 			# Now
	# 			for lf_track in last_frame_tracks:
	# 				for tf_track in this_frame_tracks:
	# 					cost = 0

	# 					last_frame_row = current_frame_df.loc[current_frame_df['Track'] == lf_track]
	# 					this_frame_row = current_frame_df.loc[current_frame_df['Track'] == lf_track]

	# 					# Distances
	# 					try:
	# 						lf_track_row = frame_dfs[current_frame - 1].loc[frame_dfs[current_frame - 1]['Track'] == lf_track]
	# 						lf_track_X = float(lf_track_row['tagX'])
	# 						lf_track_Y = float(lf_track_row['tagY'])

	# 						tf_track_row = frame_dfs[current_frame].loc[frame_dfs[current_frame]['Track'] == tf_track]
	# 						tf_track_X = float(tf_track_row['tagX'])
	# 						tf_track_Y = float(tf_track_row['tagY'])
	# 					except:
	# 						print(f'Error on frame {current_frame} cost calculation between former track {lf_track} and new track {tf_track}!')


	# 			# Run Hungarian algorithm
	# 			for_hungarian = np.copy(mini_cost_matrix)

	# 			# Calculate costs for transition


	# 			# Normalize
	# 			print('Previous cost matrix (processed, rounded to nearest integer):')
	# 			print(tabulate(for_hungarian.astype(int), tablefmt = 'pretty', headers = last_frame_tracks))
	# 			print('Relevant tracks:            ', last_frame_tracks)
	# 			hungarian_result = m.compute(for_hungarian)
	# 			print('Chosen pairs:               ', hungarian_result)
	# 			current_run_pairs = []
	# 			for tag, track in hungarian_result:
	# 				current_run_pairs.append((trimmed_tags[tag], last_frame_tracks[track]))
	# 			print('Inferred tag - track pairs: ', current_run_pairs)
	# 			hungarian_pairs.append(current_run_pairs)

	# 			# Create new cost matrix
	# 			print('\n\nCreating new cost matrix ...')
	# 			mini_cost_matrix = np.zeros((len(this_frame_tracks), len(this_frame_tracks)))
	# 			print(f'Matrix shape: {mini_cost_matrix.shape}')

	# 			# Dict for the indices of the track numbers
	# 			tracks_indices = {}
	# 			for idx in range(len(this_frame_tracks)):
	# 				tracks_indices[this_frame_tracks[idx]] = idx

	# 			chunk_end_frames.append(current_frame)
	# 			print('\n')
	# 			print('Current frame:   ', row.Index)
	# 			print('Tags:            ', tags)
	# 			print('Relevant tracks: ', this_frame_tracks)
	# 			print('\n' + '=' * 80)

	# 	# Shift variables as necessary
	# 	last_frame_tracks = this_frame_tracks


	# print('\n' + '=' * 80)
	# print('Last batch to process; no new cost matrix after this.')
	# print('Previous cost matrix (raw, rounded to nearest integer):')
	# print(tabulate(mini_cost_matrix.astype(int), headers = tags, tablefmt = 'pretty'))
	# print('\nRunning Hungarian matching algorithm on previous cost matrix.')

	# # Run Hungarian algorithm
	# for_hungarian = np.copy(mini_cost_matrix)
	# i = 0
	# for_hungarian = np.transpose(for_hungarian)
	# trimmed_tags = np.copy(tags)
	# # Eliminate zero-detection columns
	# while i < len(for_hungarian):
	# 	if np.sum(for_hungarian[i]) == 0:
	# 		for_hungarian = np.delete(for_hungarian, i, 0)
	# 		trimmed_tags = np.delete(trimmed_tags, i)
	# 	else:
	# 		i += 1
	# # Normalize
	# for j in range(len(for_hungarian)):
	# 	row_max = np.max(for_hungarian[j])
	# 	for k in range(len(for_hungarian[j])):
	# 		for_hungarian[j, k] = for_hungarian[j, k] * 100. / row_max

	# print('Previous cost matrix (processed, rounded to nearest integer):')
	# print(tabulate(for_hungarian.astype(int), tablefmt = 'pretty', headers = last_frame_tracks))
	# print('Relevant tracks:            ', last_frame_tracks)
	# hungarian_result = m.compute(for_hungarian)
	# print('Chosen pairs:               ', hungarian_result)
	# current_run_pairs = []
	# for tag, track in hungarian_result:
	# 	current_run_pairs.append((trimmed_tags[tag], last_frame_tracks[track]))
	# print('Inferred tag - track pairs: ', current_run_pairs)
	# hungarian_pairs.append(current_run_pairs)
	
	# # np.savetxt("cost_matrix.csv", track_identity_votes, delimiter=",")
	
	# # Convert data into a more generalizable format
	# print('\n' + '=' * 80)
	# print('=' * 80)
	# print('Taking pairs data into frame & track 2d array ...')
	# print('Chunk breaks: ', chunk_end_frames)
	
	# tag_tracks_2d_array = np.zeros((len(tags), last_frame))
	# chunk_end_frames.append(last_frame)
	# hungarian_pairs_list_row = 1
	# for current_frame in range(last_frame):
	# 	for tag, track in hungarian_pairs[hungarian_pairs_list_row]:
	# 		tag_tracks_2d_array[tags_indices[tag], current_frame] = track

	# 	if hungarian_pairs_list_row < len(hungarian_pairs) - 1:
	# 		if current_frame == chunk_end_frames[hungarian_pairs_list_row]:
	# 			hungarian_pairs_list_row += 1

	# # print(tabulate(np.transpose(tag_tracks_2d_array), tablefmt = 'pretty'))
	# print('\nTracks associated with each tag:')
	# associated_tracks_matrix = []
	# for tag_idx in range(len(tag_tracks_2d_array)):
	# 	associated_tracks = [f'Tag {tags[tag_idx]}']
	# 	associated_tracks.extend(np.unique(tag_tracks_2d_array[tag_idx]))
	# 	associated_tracks_matrix.append(np.delete(associated_tracks, 1)) # 0 points to empty entries, not tags... remove that entry.

	# print(tabulate(associated_tracks_matrix, tablefmt = 'pretty'))


if __name__ == '__main__':
	bee_movements_df = import_sleap_instances_df('d:\\20210715_run001_00000000_cut.mp4.predictions.slp', 20.)
	bee_movements_df.to_pickle('bee_movements_df_pickle.pkl')
	bee_movements_df = pd.read_pickle('bee_movements_df_pickle.pkl')
	# print(bee_movements_df)
	# tag_speeds_histogram(bee_movements_df)
	# print(real_tag_speed_confidence(bee_movements_df, 100))
	# print(real_rotation_speed_confidence(bee_movements_df, 0.1))
	pairings = np.genfromtxt('aruco_sleap_matching_output.csv', delimiter = ',')
	merge_sleap_tracklets('d:\\20210715_run001_00000000_cut.mp4.predictions.slp', 'aruco_sleap_matching_output_merged.csv', bee_movements_df, pairings)