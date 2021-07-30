# ArUco Tools

## matching_pipeline.py
Takes .slp file and video file, runs ArUco using SLEAP predictions to crop around tags, and matches ArUco tags to SLEAP tracks.  Included is functionality for annotating video.

Here's an example command that should run as-is on Tiger:
`python /tigress/dknapp/scripts/matching_pipeline.py /tigress/dknapp/sleap_videos/20210715_run001_00000000.mp4 /tigress/dknapp/sleap_videos/20210715_run001_00000000.mp4.predictions.slp /tigress/dknapp/scripts/matching_work 20210715_run001_1min_full_pipline_test 0 1200 -a -p -v 1`

### Method
matching_pipeline.py locates SLEAP predictions of ArUco fiducial tag locations, crops images around these locations and runs ArUco on these cropped images.  This allows optimization of ArUco by avoiding processing of empty parts of the images, and allows easy determination of tag and track associations.  To generate final SLEAP track and ArUco tag pairings that remain correct across track breaks and errors, the tag-track associations are accumulated over a rolling window, and Hungarian matching is used to produce the combination of tag-track pairings that optimizes the combination of accumulated associations.

![image](https://user-images.githubusercontent.com/81590411/127677749-fb60fe23-c2f9-46c2-9c41-975315a58ad8.png)

(Image generated using the `-d` flag)

### Arguments:
    positional arguments:
      video_path            The filepath of the video (preferably re-encoded as .mp4) to generate coordinates from as input, requires corresponding SLEAP (.slp) file.
      slp_file_path         The filepath of the SLEAP (.slp) file to generate coordinates from, corresponding with the input video file.
      files_folder_path     The filepath of a directory to save output files in. If the directory does not exist, it will be created, or failing that, an error will be thrown.
      name_stem             A string to include as stems of filenames saved to files_folder_path.
      start_here_frame      First frame in assignment. NOTE: The first frame with data output is not this frame. See comments in source code for more info.
      end_here_frame        Last frame in assignment. NOTE: The last frame with data output is not this frame. See comments in source code for more info.
    
    optional arguments:
      -h, --help            show this help message and exit
      -a, --annotate        Output a video with annotations (circles around bees with track and tag numbers; markings for ArUco tag detections).
      -p, --parallelize     Parallelize computation for more speed. Only use for long runs (> 10^4 frames) because there is overhead associated with parallelization. In shorter runs, single-threaded computation is faster and easier.
      -d, --display         Display cropped images of ArUco tags. Useful for judging crop_size. Only for local running, and never for actual large batches of data processing.
      -ao ANNOTATE_ONLY, --annotate_only ANNOTATE_ONLY
                            Skip matching step and annotate video based on previously saved pairings (*_matching_result.csv), the filepath to which must be passed along.
      -v {0,1}, --verbosity {0,1}
                            0: Minimal, 1: Everything. More options to be added (?)
      -m MINIMUM_SLEAP_SCORE, --minimum_sleap_score MINIMUM_SLEAP_SCORE
                            Minimum SLEAP prediction score for the data point to be used to look for an ArUco tag. A crude way to try to avoid wasting time on bad SLEAP predicitons.
      -c CROP_SIZE, --crop_size CROP_SIZE
                            The number of pixels horizontally and vertically around the SLEAP tag prediction point to crop away to run ArUco on. Smaller values are faster, but risk cutting off tags and ruining perfectly good data.
      -w HALF_ROLLING_WINDOW_SIZE, --half_rolling_window_size HALF_ROLLING_WINDOW_SIZE
                            Used to specify the size of the rolling window for Hungarian matching. When SLEAP makes mistakes and passes a track from one bee to another, there's a transition region where the matching will not be guaranteed
                            to follow perfectly; While the track transition is within the rolling window, we're not guaranteed to assign correctly. For instance, with half_rolling_window_size = 20 on a 20 fps video, we're not completely
                            confident about the matching until 1 second before, and 1 second after the transition.

### Running configurations
Using flags, `matching_pipeline.py` can be run in several different configurations
- ArUco tag and SLEAP track matching only: no additional flags
	- Single-threaded: no additional flags
	- Multiprocessing: `-p` flag
- ArUco tag and SLEAP track matching (both above configurations) + output video annotation: `-a` flag
- Annotation of output video using matching data from a previous run:  `-ao {matching_output_csv_filepath}`

### Multiprocessing
Multiprocessing introduces a siginficant speed boost for longer runs, about 60fps on Tigercpu compared to 20fps on single-threaded running.  However, multiprocessing should only be used for longer runs (>10,000 frames or so) because shorter runs are inefficient because of overhead associated with multiprocessing.  Furthermore, multiprocessing log output is less detailed, making it unsuited for detailed debugging.

## aruco_utils_pd.py

A fundamental .py file with a ton of useful ArUco-related utility functions.

## aruco_confusion_quantifier

Use to determine which ArUco tags can tend to be misidentified.

## create_edge_list.py

Legacy interpolation code.  Delete?
