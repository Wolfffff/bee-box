import argparse
import multiprocessing
import concurrent.futures
import time
import logging
import os
import numpy as np

from aruco_utils import *

logger = logging.getLogger("matching_pipeline_logger")
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    # SLEAP data is necessary before running this pipeline; ArUco data will be generated here.
    # We also need the python file aruco_utils_pd.py in the same folder.
    # Video is also assumed to be cropped, since all recent data has been pre-cropped.
    # Cropping functionality can be replaced by changing the 'dimension' value to a positive integer.

    # TODO: allow specification of frames to run pipeline on for easier parallelization
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

    parser.add_argument(
        "-dm",
        "--democratic",
        help="Instead of weighting track detection by SLEAP confidence score, treat each detection as equal.",
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

    # Logger add handler for tqdm progress bars
    logger.addHandler(TqdmLoggingHandler())

    # Better numpy printing

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
                    args.democratic,
                    "do not save",
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
            args.democratic,
            files_folder_path + "/" + name_stem + "_cost_matrices.csv"
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
            crop_size
        )

    total_runtime_end = time.perf_counter()
    logger.info(
        f"Total runtime of matching pipeline: {round(total_runtime_end - total_runtime_start, 2)}"
    )