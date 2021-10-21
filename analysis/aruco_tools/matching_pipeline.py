# Scroll to the bottom to see detailed comments and the entire pipeline condensed into a single block of code.
"""
************************************/(#/****************************************
**************************************/%(*//*****//*****************************
***************************************/(%///////**////**//*********************
********************************//**//////%#//////////////////*///**************
*********************************//////////(%((/((//////////*****/**************
*****************************/////((((((((((((#%%(((((////////***//(((/*********
**********************///////////(%&&(((((%&&&&&&&&&&&&&%#%%%#((///**/**********
*********************////////////(%&&%(#%&@@@@&&&@@@@&%((////////////***********
****************/****//////////(((((#/**/#%&&&&&&&@@@&&&#((///////////**********
**********/*****//////////////(((((//(&&&#/**(&&%%%%&&&&#((((////////***********
*********/**///////////////////((/*/%&%//(%&&%/***(#%%%%&&&&%(///////*/**/******
********/////////(((((((((((((((**#&&&%/*****(%&&#*,/##%%%#((((//////***********
******//////(%%&&&&&&&&&#(((((/*(&&&&&&&&(///(&&(**(%#((((((////////*/**********
******//(###%%%%%##%&&&&&&&%#(*,**(%&&&&&&&&&&#*,*%&&#((((////////////**********
***//(((/////////((((((((#%&&&&%(/***/(%&&&&%(,*(&&&%#(((((((///////////********
***/**//////////////((((((###%&&%%&&%#****//*,/%&%&%###((((((((////////*/*******
***//*//*//////////((((#&&&&&&&%%&&&&&&&&%(//(%%%%%####((((((((((///////********
*****//////////////((#&&%##%&&&%%&&&&@&&@@&&&%&%%&@&&&&&&&&&&&&&%#(//////*******
*//**/////////////((%&&&##%&&&%%&&%&&@@&&%%%&&%%&&%%%%%#((#%%&&%&&&%#///********
***/**////////////(%&&&%#%@@&&%&&%%&&&&%%%%&%%&&&@@&&#(((((((((((#%%%%(//**/****
*****////////////(#%&&&%#&&&&&&&&&&&&&%&%&&%%&&&%%%&&%((((((((////////(#(/******
*****////////////(#&&&&#%&&&&&&%&&&%%&&&%&%%&&&&%#&@&%#((////////////////(///***
*******//////////(#&&#(#&&&&&%%%&&&&&%%&%%&&&&&&%&&&&#((///////////////*********
***/*/*/////////(%&#((((%%&#&&&&&&&&&%%%%%&&&&&&&&&&%((//////////////////*******
******//////////##(/((((#%%%&&&&&&&&%&&&&&&&&&&%&&&%#(///////////////********/**
*******/*///////%(/////((%#%&&&%%%%&&&&&&&&&&&%#&&%#((//////////******////******
*******//**/////#(///////##%%%%%%&&&&&&&&&&&%#((%&#((///////////*///*****/******
*******/*/***///((///////(#%%&&&&&&&&&&%&&&#((((%#(/////////////***/*****/***//*
**********//////*//////////(((##%%%%%%&%%##((((#((///////////**/////*/////***/**
**********/****//////////////((((((((((((((/((#(///////////////////*////////////
"""

import os
import sys
import argparse
import logging
import json
import cv2
import skvideo.io
import numpy as np
import pandas as pd
import time
import h5py
import threading
import multiprocessing
import concurrent.futures
import scipy
from datetime import datetime
from contextlib import redirect_stdout
from tabulate import tabulate
from tqdm import tqdm

import aruco_utils_pd as awpd

# Logging: display AND save output
logger = logging.getLogger("matching_pipeline_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def sleap_reader(slp_predictions_path: str) -> h5py.File:
    # Setting driver = 'core' forces the entire file to be loaded in to RAM: much faster
    # SLEAP files aren't huge so this should be OK.
    f = h5py.File(slp_predictions_path, "r", driver="core")
    return f


def get_skeleton_dict(slp_json: str):
    """
    Args:
            slp_json: .slp file '/metadata' json string.

    Find the order that skeleton nodes are saved within the .slp predictions dataset.  Return as a dict that returns the relevant index when given the name of a skeleton node.
    """
    nodes_list = json.loads(slp_json)["nodes"]

    skeleton_dict = {}
    idx = 0
    for node in nodes_list:
        skeleton_dict[node["name"]] = idx
        idx += 1

    return skeleton_dict


def get_edges_list(slp_json: str):
    """
    Args:
            slp_json: .slp file '/metadata' json string.

    Find the links between skeleton nodes that are saved within the .slp predictions dataset.  Return as a dict that returns the relevant index when given the name of a skeleton node.
    """
    nodes_list = json.loads(slp_json)["skeletons"][0]["links"]

    edges_list = []
    for node in nodes_list:
        edges_list.append((int(node["source"]), int(node["target"])))

    return edges_list


def find_min_idx(x):
    """
    Args:
            x: 2D array

    Find the coordinates of the minimum value in a 2D array.
    Code stolen (and slightly slightly modified) from https://stackoverflow.com/questions/30180241/numpy-get-the-column-and-row-index-of-the-minimum-value-of-a-2d-array
    """
    k = x.argmin()
    ncol = x.shape[1]
    return k // ncol, k % ncol


def ArUco_SLEAP_matching_wrapper(p):
    # Allow ArUco_SLEAP_matching to be called by passing the arguments as a tuple.
    return ArUco_SLEAP_matching(*p)


def ArUco_SLEAP_matching(
    video_path: str,
    slp_predictions_path: str,
    start_end_frame: tuple,
    minimum_sleap_score: float = 0.1,
    crop_size: int = 50,
    half_rolling_window_size: int = 50,
    enhanced_output: bool = False,
    display_images_cv2: bool = False,
    sleap_predictions=None,
    sleap_instances=None,
    sleap_frames=None,
    tag_node=0,
    hungarian_matching=True,
) -> np.ndarray:
    """
    Args:
            video_path: path to bee video file.
            slp_predictions_path: path to .slp file containing inference results on relevant video file.
            results_df_path: Where to output the results of SLEAP-based cropping ArUco data.  This data is necessary for video annotation, and probably just great to have in general.
            start_end_frame: Tuple in the form (first frame to process, last frame to process)
            minimum_sleap_score: Minimum SLEAP prediction score for the data point to be used to look for an ArUco tag.  A crude way to try to avoid wasting time on bad SLEAP predicitons.
            crop_size: The number of pixels horizontally and vertically around the SLEAP tag prediction point to crop away to run ArUco on.  Smaller values are faster, but risk cutting off tags and ruining perfectly good data.
            half_rolling_window)size: See next section of the docstring.  Used to specify the size of the rolling window for Hungarian matching.
            enhanced_output: If set to true, this function will spit out a ton of console output useful for debugging.  Best to pipe to a .txt file and carefully read through as necessary.
            display_images_cv2: If set to true, displays cropped images.  Useful for judging crop_size.  Only for local running, and never for actual large batches of data processing.

    Frames assignment and rolling window:
            The rolling window is centered at the current frame being processed, with half_rolling_window_size frames on each side.
            This means that for a frame to be processed, it needs to have half_rolling_window_size frames on either side within the assigned range.
            For instance, if I assigned start_end_frame = (1000, 2000) with half_rolling_window_size = 50, the output would have pairings only for frames in the range [1051, 1949]

    Overall process:
            1. Load SLEAP data.
            2. Pull out the SLEAP data we care about from the h5 format so that we can iterate over each SLEAP prediction of a tag. > results_df_path
            3. Around each SLEAP tag prediction, crop a small segment of the frame and run ArUco on this.
            4. Take the data from step 3 to create a rolling-window cost matrix and apply Hungarian matching between tags and tracks. > return pairings in 2d array
    """
    # This function ....
    # Throw an error if the rolling window is too large for the assigned range of frames; self-explanatory

    if start_end_frame[1] - start_end_frame[0] < (2 * half_rolling_window_size) + 1:
        raise ValueError(
            f"[ArUco_SLEAP_matching {start_end_frame}] The rolling window is size (2 * {half_rolling_window_size} + 1) = {(2 * half_rolling_window_size) + 1}, larger than the range of assigned frames which is {start_end_frame[1]} - {start_end_frame[0]} = {start_end_frame[1] - start_end_frame[0]}"
        )

    # We have the option of passing in a SLEAP file to avoid loading it within the function.
    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] Starting assignment with frame range {start_end_frame}."
    )

    # Now, let's put the relevant SLEAP tracks into the same data structure: a dict containing interpolated coords for each track
    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [SLEAP Data Restructuring] Started"
    )
    sleap_interpolation_start = time.perf_counter()
    if (
        (sleap_predictions is None)
        or (sleap_instances is None)
        or (sleap_frames is None)
    ):
        logger.info(f"[ArUco_SLEAP_matching {start_end_frame}] Loading SLEAP file...")
        sleap_file = sleap_reader(slp_predictions_path)
        sleap_predictions = sleap_file["pred_points"][:]
        sleap_instances = sleap_file["instances"][:]
        sleap_frames = sleap_file["frames"][:]
    else:
        logger.info(
            f"[ArUco_SLEAP_matching {start_end_frame}] Received SLEAP file; moving immediately to processing"
        )

    # Not sure why, but we get a 'track -1'... get rid of it!
    unique_tracks = np.sort(np.unique([int(j[4]) for j in sleap_instances]))
    if unique_tracks[0] == -1:
        unique_tracks = unique_tracks[1:-1]

    # Iterate from the start to end frame of the range specified to collect the SLEAP predictions into a simple dataframe
    # We start with everything in a list which is cheaper to append to.  Then we instantiate a dataframe using the list.
    sleap_predictions_df = []
    for frame_number in range(start_end_frame[0], start_end_frame[1] + 1):
        # Throw an exception if we can't find proper SLEAP data in the provided .slp file path.
        try:
            iter_sleap_frame = sleap_frames[frame_number]
        except:
            raise ValueError(
                f"[ArUco_SLEAP_matching {start_end_frame}] Provided .slp file does not have predictions for frame {frame_number}, which is within the assigned range of {start_end_frame}!"
            )
        for iter_frame_idx in range(
            iter_sleap_frame["instance_id_start"], iter_sleap_frame["instance_id_end"]
        ):  # range(instance_id_start, instance_id_end)
            current_instance = sleap_instances[iter_frame_idx]
            prediction_index = current_instance[
                "point_id_start"
            ]  # Member 'point_id_start':  H5T_STD_U64LE (uint64)
            track_number = current_instance[
                "track"
            ]  # Member 'track':  H5T_STD_I32LE (int32)
            prediction = sleap_predictions[int(prediction_index + tag_node)]
            if (
                prediction["score"] >= minimum_sleap_score
            ):  # Member 'score':  H5T_IEEE_F64LE (double)
                # if prediction[2] == 1 and prediction[3] == 1: # Member 'visible':  H5T_ENUM, Member 'complete':  H5T_ENUM
                sleap_predictions_df.append(
                    (
                        frame_number,
                        track_number,
                        float(prediction["x"]),
                        float(prediction["y"]),
                    )
                )

    # Instantiate the dataframe
    sleap_predictions_df = pd.DataFrame(
        sleap_predictions_df, columns=["Frame", "Track", "cX", "cY"]
    )

    sleap_interpolation_end = time.perf_counter()
    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [SLEAP Data Restructuring] Ended, FPS: {round(float(start_end_frame[1] - start_end_frame[0] + 1) / float(sleap_interpolation_end - sleap_interpolation_start), 2)}"
    )

    if enhanced_output:
        logger.info(sleap_predictions_df)

    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [SLEAP-cropped ArUco] Started"
    )
    ScA_start = time.perf_counter()

    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [SLEAP-cropped ArUco] Initializing variables..."
    )
    # We currently use the collection of 100 tags with 4 x 4 = 16 pixels.
    # define names of a few possible ArUco tag OpenCV supports
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    }
    tagset_name = "DICT_4X4_100"
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[tagset_name])
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # ArUco parameters.  These have been adjusted by dyknapp but are worth playing with if ArUco is too slow or not detecting enough tags
    # These thresholding parameters DRAMATICALLY improve detection rate, while DRAMATICALLY hurting performance.  Since super fast processing isn't really necessary here they should be fine as is.
    arucoParams.adaptiveThreshWinSizeStep = 3
    arucoParams.adaptiveThreshWinSizeMin = 3
    arucoParams.adaptiveThreshWinSizeMax = 30
    # If too slow, start by adjusting this one up.  If we want more tags, lower it (diminishing returns)
    arucoParams.adaptiveThreshConstant = 12

    arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

    # If false positives are a problem, lower this parameter.
    arucoParams.errorCorrectionRate = 0.0

    # Appending results to a list is cheaper than appending them to a dataframe.
    results_array = []

    # Initialize OpenCV videostream
    # This isn't the fastest way to read video, but it's good enough.
    vs = cv2.VideoCapture(video_path)
    vs.set(cv2.CAP_PROP_POS_FRAMES, start_end_frame[0])

    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [SLEAP-cropped ArUco] Iterating through frames and running ArUco on crops"
    )
    # This variable is used to keep track of whether we're still processing the same frame or not.
    # We want to know when we move to a new frame so that we can load it.
    previous_frame = start_end_frame[0] - 1
    # This variable counts tags detected; only relevant for enhanced_output = True
    detections = 0
    for row in tqdm(sleap_predictions_df.itertuples()):
        # If we've moved onto processing a new frame
        if previous_frame != row.Frame:
            # vs.set involves going to the nearest keyframe and then traversing the video from there.
            # As such, it's very computationally intensive.
            # We only want to call it when a frame is skipped; otherwise we're only traversing sequentially anyways.
            # Frame skips should only occur in the UNLIKELY scenario that a frame has NO SLEAP data... Maybe a camera glitch or a temporary obstruction.
            if previous_frame != row.Frame - 1:
                vs.set(cv2.CAP_PROP_POS_FRAMES, int(row.Frame))

            # Load the new frame
            success, frame = vs.read()
            if enhanced_output and row.Frame != start_end_frame[0]:
                logger.info(
                    f"Frame {previous_frame}: {detections} tag(s), FPS: {round((previous_frame + 1.) / (time.perf_counter() - ScA_start), 2)}"
                )
                detections = 0

        # Only bother with this if the frame could be succesfully loaded.
        # Obviously, no point in trying to run ArUco on junk data
        if success:
            cropped_area = frame[
                np.maximum(int(row.cY) - crop_size, 0) : np.minimum(
                    int(row.cY) + crop_size, frame.shape[0] - 1
                ),
                np.maximum(int(row.cX) - crop_size, 0) : np.minimum(
                    int(row.cX) + crop_size, frame.shape[1] - 1
                ),
                0,
            ]

            # Display cropped tags.
            # This is useful for diagnosing whether the parameter crop_size is set to a reasonable value.
            if display_images_cv2:
                cv2.imshow("cropped area", cv2.resize(cropped_area, (500, 500)))
                cv2.waitKey(0)

            # Run ArUco
            (corners, ids, rejected) = cv2.aruco.detectMarkers(
                cropped_area, arucoDict, parameters=arucoParams
            )

            # If we detected any tags
            if len(corners) > 0:
                # Add the number of detected tags to the detections count
                detections += len(corners)

                # Iterate through detected tags and append results to a results list
                # As before in the SLEAP preprocessing step, appending to a list is cheaper so we do so and then instantiate a dataframe.
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

                    # Calculate tag rotation.
                    # This doesn't really work all that well... doesn't hurt to have though.
                    Theta = np.arctan2(
                        topRight[1] - bottomLeft[1], topRight[0] - bottomLeft[0]
                    )

                    # 'Frame', 'Track', 'Tag', 'cX','cY', 'Theta'
                    results_array.append(
                        (row.Frame, row.Track, int(markerID[0]), row.cX, row.cY, Theta)
                    )

        previous_frame = row.Frame

    if enhanced_output:
        logger.info(
            f"[ArUco_SLEAP_matching {start_end_frame}] Frame {previous_frame}: {detections} tag(s)"
        )

    # Instantiate a results dataframe and save it!
    results_df = pd.DataFrame(
        results_array, columns=["Frame", "Track", "Tag", "cX", "cY", "Theta"]
    )
    results_df = results_df.astype(
        {
            "Frame": int,
            "Track": int,
            "Tag": int,
            "cX": np.float64,
            "cY": np.float64,
            "Theta": np.float64,
        }
    )

    ScA_end = time.perf_counter()
    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [SLEAP-cropped ArUco] Ended, FPS: {round(float(start_end_frame[1] - start_end_frame[0] + 1) / float(ScA_end - ScA_start), 2)}"
    )

    if enhanced_output:
        logger.info(f"\nResults dataframe:\n{results_df}\n\n")

    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [Rolling Window Tag-Track Association] Started"
    )
    RWTTA_start = time.perf_counter()

    # This dict stores the cost matrices for individual frames.  The key is the frame number.
    # When we want the cost matrix for a window, we just sum the matrices from the constituent frames.
    frame_cost_matrices_dict = {}

    # Find unique tags and tracks within the data we've collected so far.
    tags = np.sort(pd.unique(results_df["Tag"]))
    tracks = np.sort(pd.unique(results_df["Track"]))

    # These dicts are useful for indexing in cost matrices (and some others).  The keys are the tag/track numbers, and they return the index for the tag or track number.
    tag_indices = {}
    for idx in range(len(tags)):
        tag_indices[int(tags[idx])] = idx

    track_indices = {}
    for idx in range(len(tracks)):
        track_indices[int(tracks[idx])] = idx

    # Array to store results:
    # Initialize an array full of zeros
    tag_tracks_2d_array = np.zeros(
        (
            1 + len(tags),
            start_end_frame[1]
            - half_rolling_window_size
            - (start_end_frame[0] + half_rolling_window_size)
            + 1,
        )
    )
    tag_tracks_2d_array[:] = -1
    # Add a column of labels to show which tags are associated to which rows.  This isn't necessary at all on the programming side, but it GREATLY enhances readability for humans trying to debug.
    tag_tracks_2d_array[:, 0] = np.concatenate(([0], tags))
    # Column headers denoting frame numbers.  Same deal as above, although the program does use this, mostly out of convenience.  If it's there for human convenience, we might as well use it when convenient for the program too!
    tag_tracks_2d_array[0, :] = np.concatenate(
        (
            [start_end_frame[0] - 1],
            np.arange(
                start_end_frame[0] + half_rolling_window_size + 1,
                start_end_frame[1] - half_rolling_window_size + 1,
            ),
        )
    )

    if enhanced_output:
        logger.info("\n")
        logger.info(f"Detected tags:   {tags}")
        logger.info(f"Detected tracks: {tracks}")
        logger.info("\n")

    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [Rolling Window Tag-Track Association] Filling initial window"
    )
    # Go ahead and fill data for the first window
    # This lets us move forward in the rolling window just by computing the next frame entering the window each time: fast!
    for frame in tqdm(
        range(
            start_end_frame[0], start_end_frame[0] + (2 * half_rolling_window_size) + 1
        )
    ):
        new_frame_df = results_df.loc[results_df["Frame"] == int(frame)]

        frame_cost_matrices_dict[frame] = np.zeros((len(tags), len(tracks)))

        for row in new_frame_df.itertuples():
            frame_cost_matrices_dict[frame][
                tag_indices[int(row.Tag)], track_indices[int(row.Track)]
            ] -= 1

    if hungarian_matching:
        logger.info(
            f"[ArUco_SLEAP_matching {start_end_frame}] [Rolling Window Tag-Track Association] Starting Hungarian assignments with rolling window"
        )
    else:
        logger.info(
            f"[ArUco_SLEAP_matching {start_end_frame}] [Rolling Window Tag-Track Association] Starting Greedy assignments with rolling window"
        )
    # Start rolling the window forward.
    for center_of_window_frame in range(
        start_end_frame[0] + half_rolling_window_size + 1,
        start_end_frame[1] - half_rolling_window_size + 1,
    ):
        if enhanced_output:
            logger.info("\n\n" + "=" * 80)
            logger.info(f"Frame (center of window): {center_of_window_frame}\n")

        # Delete data for the frame that's left the window:
        del frame_cost_matrices_dict[
            center_of_window_frame - half_rolling_window_size - 1
        ]

        # Generate data for frame entering the window:
        # As stated before, we only need to add the results of the very forward-most frame as we roll the window forward.
        frame_cost_matrices_dict[
            center_of_window_frame + half_rolling_window_size
        ] = np.zeros((len(tags), len(tracks)))
        new_frame_df = results_df.loc[
            results_df["Frame"]
            == int(center_of_window_frame) + half_rolling_window_size
        ]
        for row in new_frame_df.itertuples():
            frame_cost_matrices_dict[center_of_window_frame + half_rolling_window_size][
                tag_indices[int(row.Tag)], track_indices[int(row.Track)]
            ] -= 1

        # Calculate the cost matrix for this window; just by summing over the already-saved individual frame cost matrices.
        # Technically, it's faster to subtract from the cost matrix the one frame leaving the window, then add the cost matrix of the frame entering the window.
        # That compromises readability and copy-paste-ability, and since this really isn't the speed bottleneck anyways, we can let it pass.
        cost_matrix = np.zeros((len(tags), len(tracks)))
        for window_frame in range(
            center_of_window_frame - half_rolling_window_size,
            center_of_window_frame + half_rolling_window_size + 1,
        ):
            cost_matrix += frame_cost_matrices_dict[window_frame]

        if hungarian_matching:
            # Remove zero rows and columns from cost matrix
            # Rows
            idx = np.argwhere(np.all(cost_matrix[:, ...] == 0, axis=1))
            cost_matrix = np.delete(cost_matrix, idx, axis=0)
            trimmed_tags = np.delete(tags, idx)

            # Columns
            idx = np.argwhere(np.all(cost_matrix[..., :] == 0, axis=0))
            cost_matrix = np.delete(cost_matrix, idx, axis=1)
            trimmed_tracks = np.delete(tracks, idx)

            # The Hungarian algorithm is designed for square matrices, and bar coincidence (or perfection on both ArUco and SLEAP sides), there will always be a different number of candidate tracks and tags.
            # TODO: Update comments to reflect scipy.optimize.linear_sum_assignment
            # The Munkres package has automatic padding, but it still wants the matrix to have more columns then rows when doing so.
            # We transpose the matrix when running the Hungarian algorithm if necessary, to make sure that Munkres is happy.
            # If there are fewer tracks than tags, every track gets a tag, and vice versa.
            if len(tracks) < len(tags):
                cost_matrix = np.transpose(cost_matrix)
                hungarian_result_raw = scipy.optimize.linear_sum_assignment(cost_matrix)
                hungarian_result = list(
                    zip(hungarian_result_raw[0], hungarian_result_raw[1])
                )
                hungarian_pairs = []
                for track, tag in hungarian_result:
                    hungarian_pairs.append((trimmed_tags[tag], trimmed_tracks[track]))
                cost_matrix = np.transpose(cost_matrix)
            else:
                hungarian_result_raw = scipy.optimize.linear_sum_assignment(cost_matrix)
                hungarian_result = list(
                    zip(hungarian_result_raw[0], hungarian_result_raw[1])
                )
                hungarian_pairs = []
                for tag, track in hungarian_result:
                    if cost_matrix[tag, track] != 0:
                        hungarian_pairs.append(
                            (trimmed_tags[tag], trimmed_tracks[track])
                        )
        else:
            # Quick & dirty greedy matching algorithm.
            # Take the lowest cost, assign, and delete.
            # We don't even worry about equal minimums.
            hungarian_pairs = []
            cost_matrix_copy = cost_matrix.copy()
            while True:
                tag, track = find_min_idx(cost_matrix_copy)
                if cost_matrix_copy[tag, track] >= 0:
                    break
                cost_matrix_copy[tag, track] = 0
                hungarian_pairs.append((tags[tag], tracks[track]))

        # hungarian_pairs is a collection of tuples holding the raw (tag, track) pairing for this particular frame.
        # We want the pairings to be able to change between frames, so we stick them into the tag_tracks_2d_array
        for tag, track in hungarian_pairs:
            # indexing is a bit messy for this array so we make things easier with np.searchsorted.
            # Not optimal, but much more robust and much more readable than stuffing a ton of arithmetic into the index
            tag_tracks_2d_array[
                1 + tag_indices[tag],
                np.searchsorted(tag_tracks_2d_array[0, :], center_of_window_frame),
            ] = track

        if enhanced_output:
            # Remove zero rows and columns from cost matrix
            # Rows
            idx = np.argwhere(np.all(cost_matrix[:, ...] == 0, axis=1))
            cost_matrix = np.delete(cost_matrix, idx, axis=0)
            trimmed_tags = np.delete(tags, idx)

            # Columns
            idx = np.argwhere(np.all(cost_matrix[..., :] == 0, axis=0))
            cost_matrix = np.delete(cost_matrix, idx, axis=1)
            trimmed_tracks = np.delete(tracks, idx)

            display_matrix = np.copy(np.transpose(cost_matrix.astype(int).astype(str)))
            for x in range(display_matrix.shape[0]):
                for y in range(display_matrix.shape[1]):
                    if display_matrix[x, y] == "0":
                        display_matrix[x, y] = " "
            logger.info(
                tabulate(
                    np.transpose(np.vstack([trimmed_tags.astype(int), display_matrix])),
                    tablefmt="pretty",
                    headers=trimmed_tracks,
                )
            )
            logger.info("\n")
            logger.info(f"Assigned tag-track pairs: {hungarian_pairs}")
            logger.info(f"Cost matrix shape: {cost_matrix.shape}")
            logger.info(
                f"Cumulative FPS: {round((previous_frame + 1.) / (time.perf_counter() - RWTTA_start), 2)}"
            )

    RWTTA_end = time.perf_counter()
    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [Rolling Window Tag-Track Association] Ended, FPS: {round(float(start_end_frame[1] - start_end_frame[0] + 1) / float(RWTTA_end - RWTTA_start), 2)}"
    )

    # Inherit tracks -- forward fill
    logger.info(f"[ArUco_SLEAP_matching {start_end_frame}] [Track Inheritance] Running")

    # https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    mask = tag_tracks_2d_array == -1
    logger.info(tag_tracks_2d_array[0])
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    tag_tracks_2d_array = tag_tracks_2d_array[np.arange(idx.shape[0])[:, None], idx]
    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] [Track Inheritance] Finished!"
    )
    logger.info(
        f"[ArUco_SLEAP_matching {start_end_frame}] Done with SLEAP-based cropping ArUco tag-track association"
    )

    return tag_tracks_2d_array.astype(int)


def annotate_video_sleap_aruco_pairings(
    video_path: str,
    video_output_path: str,
    aruco_csv_path: str,
    slp_predictions_path: str,
    pairings: list,
    frames_to_annotate: list,
) -> None:
    """
    annotate video with pairings of aruco tags and sleap tracks
    output is .avi

    Args:
            video_path: path of video to be annotated
            video_output_path: output path for annotated video
            aruco_csv_path: Path with ArUco data
            slp_predictions_path: path of .slp file with relevant inference data for the video
            pairings: the return value of the matching function
            frames_to_annotate: iterable of frames to annotate

    """
    logger = logging.getLogger("matching_pipeline_logger")
    logger.info("\nStarted video annotation of matching results.")
    aruco_df = awpd.load_into_pd_dataframe(aruco_csv_path)
    tags = np.sort(awpd.find_tags_fast(aruco_df))
    sleap_file = sleap_reader(slp_predictions_path)
    sleap_predictions = np.array(sleap_file["pred_points"])
    sleap_instances = np.array(sleap_file["instances"])
    metadata_json_string = sleap_file["/metadata"].attrs["json"]
    skeleton_dict = get_skeleton_dict(metadata_json_string)
    edges_list = get_edges_list(metadata_json_string)

    video_data = cv2.VideoCapture(video_path)
    fps = video_data.get(cv2.CAP_PROP_FPS)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Below code heavily based on SLEAP (sleap.io.videowriter.py)
    fps = str(fps)
    crf = 28
    scale_factor = 2
    preset = "veryfast"
    writer = skvideo.io.FFmpegWriter(
        video_output_path,
        inputdict={
            "-r": fps,
        },
        outputdict={
            "-c:v": "libx264",
            "-preset": preset,
            "-framerate": fps,
            "-crf": str(crf),
            "-pix_fmt": "yuv420p",
            "-vf": f"scale=w=iw/{scale_factor}:h=ih/{scale_factor}",
        },  # verbosity = 1
    )

    success, image = video_data.read()
    font = cv2.FONT_HERSHEY_SIMPLEX

    green = (17, 136, 0)
    red = (238, 0, 17)
    pink = (221, 17, 136)

    # ArUco persistent plot
    last_seen = np.zeros((3, len(tags)), dtype=int)

    current_frame_idx = 0
    next_frame_idx = 0
    errors = 0
    previous_frame = frames_to_annotate[0]
    for frame in tqdm(frames_to_annotate, fill="X"):
        if frame != previous_frame + 1:
            video_data.set(1, frame)
        success, image = video_data.read()
        # Find starting point in .slp instances data
        nth_inst_tuple = sleap_instances[current_frame_idx]

        # Skip until our current frame
        while nth_inst_tuple[2] != frame:  # frame_id
            current_frame_idx += 1
            nth_inst_tuple = sleap_instances[current_frame_idx]

        while nth_inst_tuple[2] != frame + 1:  # frame_id
            next_frame_idx += 1
            nth_inst_tuple = sleap_instances[next_frame_idx]

        for idx in range(current_frame_idx, next_frame_idx):
            nth_inst_tuple = sleap_instances[idx]
            prediction_start_idx = nth_inst_tuple[7]
            prediction_end_idx = nth_inst_tuple[8]
            prediction_coords = [[], []]  # x, y
            for pred_idx in range(prediction_start_idx, prediction_end_idx):
                prediction_tuple = sleap_predictions[pred_idx]
                prediction_coords[0].append(float(prediction_tuple[0]))
                prediction_coords[1].append(float(prediction_tuple[1]))
                try:
                    image = cv2.circle(
                        image,
                        (
                            int(round(float(prediction_tuple[0]))),
                            int(round(float(prediction_tuple[1]))),
                        ),
                        10,
                        pink,
                        cv2.FILLED,
                    )
                except:
                    errors += 1

            # if len(prediction_coords[0]) == 4:
            # 	for edge in edges_list:
            # 		try:
            # 			image = cv2.line(image, (int(prediction_coords[0][edge[0]]), int(prediction_coords[1][edge[0]])), (int(prediction_coords[0][edge[1]]), int(prediction_coords[1][edge[1]])), pink, 3)
            # 		except:
            # 			pass

            try:
                prediction_tuple = sleap_predictions[
                    prediction_start_idx
                ]  # start_idx corresponds to the tag
                pX = int(round(prediction_tuple[0]))
                pY = int(round(prediction_tuple[1]))
                start_point = (pX - int((crop_size / 2)), pY - int((crop_size / 2)))
                end_point = (pX + int((crop_size / 2)), pY + int((crop_size / 2)))
                color = (0, 0, 255)
                thickness = 2
                image = cv2.rectangle(image, start_point, end_point, color, thickness)

                current_track = int(nth_inst_tuple[4])
                pairings_frame_idx = np.searchsorted(pairings[0, 1:-1], frame)
                current_tag = "?"
                idx = 0
                for entry in pairings[:, pairings_frame_idx]:
                    if entry == current_track and entry >= 0:
                        current_tag = pairings[idx, 0]
                    idx += 1

                cv2.putText(image, str(current_tag), (pX, pY - 75), font, 4, green, 2)
                cv2.putText(
                    image, str(int(current_track)), (pX, pY + 100), font, 2, green, 2
                )
            except Exception as e:
                logger.error(f"Exception raised when writing on frame {frame}")
                logger.error(f"Exception: {e}")
                errors += 1

        # Persistent ArUco location plot
        if frame in aruco_df.index:
            for row in aruco_df[aruco_df.index == frame].itertuples():
                try:
                    last_seen[0, np.searchsorted(tags, row.Tag)] = int(round(row.cX))
                    last_seen[1, np.searchsorted(tags, row.Tag)] = int(round(row.cY))
                    last_seen[2, np.searchsorted(tags, row.Tag)] = float(row.Theta)
                except:
                    pass

        for k in range(0, len(tags)):
            if last_seen[0, k] > 0 and last_seen[1, k] > 0:
                image = cv2.circle(image, (last_seen[0, k], last_seen[1, k]), 5, red, 2)
                Theta = last_seen[2, k]
                cX = last_seen[0, k]
                cY = last_seen[1, k]
                cv2.putText(
                    image,
                    str(tags[k]),
                    (last_seen[0, k], last_seen[1, k]),
                    font,
                    2,
                    red,
                    2,
                )
        writer.writeFrame(image)

        current_frame_idx = next_frame_idx
        previous_frame = frame

    # Do a bit of cleanup
    video_data.release()
    writer.close()


def generate_final_output_dataframe(
    pairings: np.ndarray,
    output_path: str,
    start_end_frame=tuple,
    sleap_predictions=None,
    sleap_instances=None,
    sleap_frames=None,
    skeleton_dict: dict = {},
) -> None:
    """
    Args:
            pairings: 2d ndarray generated by ArUco_SLEAP_matching().  Rows represent tags, columns represent frames, and entries represent tracks.  Includes a row and a column for column and row headers, respectively.
            output_path: Path to save the output csv to.
            sleap_predictions: 'pred_points' dataset from .slp inferences.  Pass this in if you already have it to save the trouble of reloading it within the function.
            sleap_instances: 'instances' dataset from .slp inferences.  Pass this in if you already have it to save the trouble of reloading it within the function.
            sleap_frames: 'frames' dataset from .slp inferences.  Pass this in if you already have it to save the trouble of reloading it within the function.
    """
    # Generate dataframe coordinates output
    # Frame, Tag, TagX, TagY
    # TODO: Annotate this
    # TODO: Add all body parts to the dataframe
    # TODO: Weird bug where last column of pairings is ignored.

    logger = logging.getLogger("matching_pipeline_logger")

    if (
        (sleap_predictions is None)
        or (sleap_instances is None)
        or (sleap_frames is None)
    ):
        logger.info(f"[ArUco_SLEAP_matching {start_end_frame}] Loading SLEAP file...")
        sleap_file = sleap_reader(slp_predictions_path)
        sleap_predictions = sleap_file["pred_points"][:]
        sleap_instances = sleap_file["instances"][:]
        sleap_frames = sleap_file["frames"][:]
    else:
        logger.info(
            f"[Final Output Generation {start_end_frame}] Received SLEAP file; moving immediately to processing"
        )

    output_data = []
    failures = 0
    frame_idx = 0
    if enhanced_output:
        logger.info(f"All frames for final data generation: {pairings[0, 1 : -1]}")
    for frame in pairings[0, 1:-1]:
        frame_idx += 1
        current_sleap_frame = sleap_frames[frame]
        current_frame_idx = current_sleap_frame["instance_id_start"]
        next_frame_idx = current_sleap_frame["instance_id_end"]
        for idx in range(current_frame_idx, next_frame_idx):
            nth_inst_tuple = sleap_instances[idx]
            prediction_start_idx = nth_inst_tuple["point_id_start"]
            prediction = sleap_predictions[
                int(prediction_start_idx + skeleton_dict["Tag"])
            ]
            cX = float(prediction["x"])
            cY = float(prediction["y"])
            prediction = sleap_predictions[
                int(prediction_start_idx + skeleton_dict["Head"])
            ]
            headX = float(prediction["x"])
            headY = float(prediction["y"])
            prediction = sleap_predictions[
                int(prediction_start_idx + skeleton_dict["Abdomen"])
            ]
            abdomenX = float(prediction["x"])
            abdomenY = float(prediction["y"])
            prediction = sleap_predictions[
                int(prediction_start_idx + skeleton_dict["Thorax"])
            ]
            thoraxX = float(prediction["x"])
            thoraxY = float(prediction["y"])
            Theta = np.arctan2(abdomenY - headY, abdomenX - headX)
            current_track = nth_inst_tuple["track"]
            if current_track >= 0 and current_track in pairings[1:-1, frame_idx]:
                # If there exists a pairing for this particular SLEAP instance
                # Otherwise, we still append the data, but with tag number -1
                idx = 0
                for entry in pairings[:, frame_idx]:
                    if entry == current_track:
                        current_tag_idx = idx
                        break
                    idx += 1
                current_tag = pairings[current_tag_idx, 0]
                try:
                    if int(current_tag) >= 0:
                        output_data.append(
                            (
                                frame,
                                int(current_tag),
                                cX,
                                cY,
                                headX,
                                headY,
                                thoraxX,
                                thoraxY,
                                abdomenX,
                                abdomenY,
                                Theta,
                            )
                        )
                except:
                    pass

            else:
                try:
                    output_data.append(
                        (
                            frame,
                            -1,
                            cX,
                            cY,
                            headX,
                            headY,
                            thoraxX,
                            thoraxY,
                            abdomenX,
                            abdomenY,
                            Theta,
                        )
                    )
                except:
                    pass

    output_df = pd.DataFrame(
        output_data,
        columns=[
            "Frame",
            "Tag",
            "cX",
            "cY",
            "headX",
            "headY",
            "thoraxX",
            "thoraxY",
            "abdomenX",
            "abdomenY",
            "Theta",
        ],
    )
    logger.info(output_df)
    output_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # SLEAP data is necessary before running this pipeline; ArUco data will be generated here.
    # We also need the python file aruco_utils_pd.py in the same folder.
    # Video is also assumed to be cropped, since all recent data has been pre-cropped.
    # Cropping functionality can be replaced by changing the 'dimension' value to a positive integer.

    # TODO: allow specification of frames to run pipline on for easier parallelization
    # TODO: write functionality for the pipline to start partway through if data already exists; this speeds up recovery from failures, and can also lay the basis for much more efficient experimentation with parameters!
    # TODO: Finish fixing bugs with multiprocessing

    # Argument parsing.  The help = ... are just as good as comments!
    # Here's an example of how one might run this (if you run this on tiger.princeton.edu, it should work verbatim!):
    # python /tigress/dknapp/scripts/matching_pipeline.py /tigress/dknapp/sleap_videos/20210715_run001_00000000.mp4 /tigress/dknapp/sleap_videos/20210715_run001_00000000.mp4.predictions.slp /tigress/dknapp/scripts/matching_work 20210715_run001_1min_full_pipline_test 0 12000 -a -p -v 1

    # Just a nice litte counter to keep track of the total runtime of the pipeline
    total_runtime_start = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Import SLEAP data, locate ArUco tags, and output SLEAP tracks with corresponding ArUco tag."
    )

    parser.add_argument(
        "video_path",
        help="The filepath of the video (preferably re-encoded as .mp4) to generate coordinates from as input, requires corresponding SLEAP (.slp) file.",
        type=str,
    )

    parser.add_argument(
        "slp_file_path",
        help="The filepath of the SLEAP (.slp) file to generate coordinates from, corresponding with the input video file.",
        type=str,
    )

    parser.add_argument(
        "files_folder_path",
        help="The filepath of a directory to save output files in. If the directory does not exist, it will be created, or failing that, an error will be thrown.",
        type=str,
    )

    parser.add_argument(
        "name_stem",
        help="A string to include as stems of filenames saved to files_folder_path.",
        type=str,
    )

    parser.add_argument(
        "start_here_frame",
        help="First frame in assignment.  NOTE: The first frame with data output is not this frame.  See comments in source code for more info.",
        type=int,
    )
    parser.add_argument(
        "end_here_frame",
        help="Last frame in assignment.  NOTE: The last frame with data output is not this frame.  See comments in source code for more info.",
        type=int,
    )

    parser.add_argument(
        "-a",
        "--annotate",
        help="Output a video with annotations (circles around bees with track and tag numbers; markings for ArUco tag detections).",
        action="store_true",
    )

    parser.add_argument(
        "-p",
        "--parallelize",
        help="Parallelize computation for more speed.  Only use for long runs (> 10^4 frames) because there is overhead associated with parallelization.  In shorter runs, single-threaded computation is faster and easier.",
        action="store_true",
    )

    parser.add_argument(
        "-d",
        "--display",
        help="Display cropped images of ArUco tags.  Useful for judging crop_size.  Only for local running, and never for actual large batches of data processing.",
        action="store_true",
    )

    parser.add_argument(
        "-ao",
        "--annotate_only",
        help="Skip matching step and annotate video based on previously saved pairings (*_matching_result.csv), the filepath to which must be passed along.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        help="0: Minimal, 1: Everything.  More options to be added (?)",
        type=int,
        choices=[0, 1],
        default=0,
    )

    parser.add_argument(
        "-m",
        "--minimum_sleap_score",
        help="Minimum SLEAP prediction score for the data point to be used to look for an ArUco tag.  A crude way to try to avoid wasting time on bad SLEAP predicitons.",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "-c",
        "--crop_size",
        help="The number of pixels horizontally and vertically around the SLEAP tag prediction point to crop away to run ArUco on.  Smaller values are faster, but risk cutting off tags and ruining perfectly good data.",
        type=int,
        default=50,
    )

    parser.add_argument(
        "-w",
        "--half_rolling_window_size",
        help="Used to specify the size of the rolling window for Hungarian matching. When SLEAP makes mistakes and passes a track from one bee to another, there's a transition region where the matching will not be guaranteed to follow perfectly; While the track transition is within the rolling window, we're not guaranteed to assign correctly. For instance, with half_rolling_window_size = 20 on a 20 fps video, we're not completely confident about the matching until 1 second before, and 1 second after the transition.",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-t",
        "--threads",
        help="Used to specific max number of threads used by concurrent.futures.ThreadPoolExecutor",
        type=int,
        default=multiprocessing.cpu_count(),
    )

    parser.add_argument(
        "-hm",
        "--hungarian",
        help="Use hungarian matching for cost matrix opimization in tag-track matching.  Much slower, but guaranteed to minimuze overall cost.",
        action="store_true",
    )

    args = parser.parse_args()
    video_path = args.video_path
    slp_file_path = args.slp_file_path
    files_folder_path = args.files_folder_path
    start_here_frame = args.start_here_frame
    end_here_frame = args.end_here_frame
    threads = args.threads

    # If files folder doesn't exist, create it!
    if not os.path.exists(files_folder_path):
        try:
            os.mkdir(files_folder_path)
        except:
            raise ValueError("files_folder_path is not a valid directory path.")

    name_stem = args.name_stem
    annotate_video = args.annotate
    multithreaded = args.parallelize
    display_images_cv2 = args.display

    if args.verbosity == 0:
        enhanced_output = False
    else:
        enhanced_output = True

    minimum_sleap_score = args.minimum_sleap_score
    crop_size = args.crop_size
    half_rolling_window_size = args.half_rolling_window_size

    # Logger add handler for saving output to log file
    logger.addHandler(
        logging.FileHandler(files_folder_path + "/" + name_stem + "_output_logs.log")
    )

    # Better numpy printing
    np.set_printoptions(
        edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
    )

    # ArUco and SLEAP matching with a rolling window Hungarian matching system

    # Load SLEAP file and sort data into relevant variables for later reference
    logger.info("[MAIN] Loading SLEAP file...")
    sleap_file = sleap_reader(slp_file_path)
    sleap_predictions = sleap_file["pred_points"][:]
    sleap_instances = sleap_file["instances"][:]
    sleap_frames = sleap_file["frames"][:]
    metadata_json_string = sleap_file["/metadata"].attrs["json"]
    skeleton_dict = get_skeleton_dict(metadata_json_string)
    edges_list = get_edges_list(metadata_json_string)

    logger.info(f"[MAIN] Detected .slp skeleton nodes: {skeleton_dict}")
    logger.info(f"[MAIN] Detected .slp skeleton edges: {edges_list}")

    ArUco_csv_path = (
        files_folder_path + "/" + name_stem + "_aruco_data_with_track_numbers.csv"
    )
    if multithreaded and args.annotate_only == "":
        # Find number of CPU to maximize parallelization!
        number_of_cpus = multiprocessing.cpu_count()
        logger.info(f"[MAIN] {number_of_cpus} CPUs available!")

        # Split the assigned frames into parallel chunks
        # The code is slightly messy because the chunks must overlap by half_rolling_window_size... for details see the docstring for ArUco_sleap_matching
        assignment_tuples = []
        frames_per_cpu = int((end_here_frame - start_here_frame) / threads)

        assignment_tuples.append(
            (
                start_here_frame,
                start_here_frame + frames_per_cpu + half_rolling_window_size,
            )
        )

        while (
            assignment_tuples[-1][1]
            + frames_per_cpu
            + (2 * half_rolling_window_size + 1)
            < end_here_frame
        ):
            assignment_tuples.append(
                (
                    assignment_tuples[-1][1] - (2 * half_rolling_window_size),
                    assignment_tuples[-1][1]
                    + frames_per_cpu
                    + half_rolling_window_size,
                )
            )

        assignment_tuples.append(
            (assignment_tuples[-1][1] - (2 * half_rolling_window_size), end_here_frame)
        )

        logger.info(f"[MAIN] Assignment ranges: {assignment_tuples}")

        # Put together a list of parameter tuples to pass into the parallel instances

        chunks_to_assign = []
        for chunk in assignment_tuples:
            chunks_to_assign.append(
                (
                    video_path,
                    slp_file_path,
                    chunk,
                    minimum_sleap_score,
                    crop_size,
                    half_rolling_window_size,
                    False,
                    False,
                    sleap_predictions,
                    sleap_instances,
                    sleap_frames,
                    skeleton_dict["Tag"],
                    args.hungarian,
                )
            )
            logger.info(f"[MAIN] Created assignment for {chunk}")

        logger.info("[MAIN] Done preparing assignment chunks.")

        # Start the parallel tasks!
        start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            logger.info("[MAIN] Tasks now in queue...")
            results_generator = executor.map(
                ArUco_SLEAP_matching_wrapper, chunks_to_assign
            )
        end = time.perf_counter()
        logger.info(
            f"[MAIN] Multiprocessed matching ended, effective overall FPS: {round(float(end_here_frame - start_here_frame + 1) / float(end - start), 2)}"
        )

        # Once results are available, stack them up!
        # Well, this is actually requires a bit more subtelty; the different chunks may have different tags

        # Put results into list
        results = []
        for result in results_generator:
            results.append(result)

        # Set meaningless corner entry to -1 for cleanliness
        for idx in range(len(results)):
            results[idx][0, 0] = -1

        # Collect tags in each of the results
        result_tags = []
        for idx in range(len(assignment_tuples)):
            result_tags.append(results[idx][1:-1, 0])

        # Find all of the unique tags in the entire range
        all_unique_tags = np.sort(np.unique(np.concatenate(result_tags)))

        # If one of the result chunks is missing rows for tags, put them in
        # This lets us then simply stack the arrays
        for idx in range(len(assignment_tuples)):
            for tag in all_unique_tags:
                if not (tag in results[idx][1:-1, 0]):
                    insert_idx = np.searchsorted(results[idx][1:-1, 0], tag) + 1
                    results[idx] = np.insert(
                        results[idx],
                        insert_idx,
                        np.zeros(results[idx].shape[1]),
                        axis=0,
                    )
                    results[idx][insert_idx, 0] = tag

                    if enhanced_output:
                        logger.info(f"Added empty row for tag {tag}")

            if enhanced_output:
                logger.info(results[idx])
                logger.info("\n")

        # Horizontally stack up the results
        pre_stack_results = []
        for idx in range(len(results)):
            # We do this.  With just the below line:
            # pre_stack_results.append(results[idx][:, 1:-1])
            # the code somehow drops  the last column of data, which doesn't make any sense to me (dknapp).
            results[idx] = np.delete(results[idx], 0, axis=1)
            pre_stack_results.append(results[idx])
            if enhanced_output:
                logger.info(np.transpose(pre_stack_results[idx]))
        overall_result = np.hstack(pre_stack_results)
        if enhanced_output:
            logger.info("\n\nStacked multiprocessing results:\n")
            logger.info(np.transpose(overall_result))

        pairings = overall_result

    # If single-threaded
    elif args.annotate_only == "":
        pairings = ArUco_SLEAP_matching(
            video_path,
            slp_file_path,
            (start_here_frame, end_here_frame),
            minimum_sleap_score,
            crop_size,
            half_rolling_window_size,
            enhanced_output,
            display_images_cv2,
            sleap_predictions,
            sleap_instances,
            sleap_frames,
            skeleton_dict["Tag"],
            args.hungarian,
        )
        if enhanced_output:
            logger.info(np.transpose(pairings))

    # Save matching results as a CSV
    if args.annotate_only == "":
        matching_results_path = (
            files_folder_path + "/" + name_stem + "_matching_result.csv"
        )
        np.savetxt(matching_results_path, np.copy(pairings), delimiter=",")
    else:
        try:
            pairings = np.loadtxt("annotate_only", delimiter=",")
        except:
            raise ValueError(
                "The path to matching results for video annotation is incorrect"
            )

    generate_final_output_dataframe(
        pairings,
        ArUco_csv_path,
        (start_here_frame, end_here_frame),
        sleap_predictions,
        sleap_instances,
        sleap_frames,
        skeleton_dict,
    )

    # We're done with actual data processing!  Yay!
    # Now, we're left with the optional process of annotating the video with our pairings.
    if annotate_video:
        annotate_video_sleap_aruco_pairings(
            video_path,
            files_folder_path + "/" + name_stem + "_annotated.mp4",
            ArUco_csv_path,
            slp_file_path,
            pairings,
            range(start_here_frame, end_here_frame),
        )

    total_runtime_end = time.perf_counter()
    logger.info(
        f"Total runtime of matching pipline: {round(total_runtime_end - total_runtime_start, 2)}"
    )

# python matching_pipeline.py d:\\20210715_run001_00000000_cut.mp4 d:\\20210725_preds_1200frames.slp d:/matching_testing crop_ArUco_testing False 0 300 True
# /Genomics/grid/users/swwolf/.conda/envs/sleap/bin/python python matching_pipeline.py -a -v 1 -w 10 -c 75 20210715_run001_00000000_1h.mp4 20210725_preds_71998.slp crop_matching_71998 crop_matching 0 71998
