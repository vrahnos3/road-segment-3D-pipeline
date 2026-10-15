from pathlib import Path
from rich.console import Console
if __package__:
    from .utils import run_command
else:
    from utils import run_command
from datetime import timedelta
from time import perf_counter
import re
import sys
import math
import argparse

CONSOLE = Console(width=120)


def get_num_video_frames(video_path: Path):
    cmd = f'ffprobe -v error -select_streams v:0 -count_packets \
        -show_entries stream=nb_read_packets -of csv=p=0 "{video_path}"'
    output = run_command(cmd)
    assert output is not None
    number_match = re.search(r"\d+", output)
    assert number_match is not None
    return int(number_match[0])

def convert_video_to_images(video_path: Path,
                            image_dir: Path,
                            num_frames_target: int,
                            num_downscales: int,
                            verbose: bool = False,
                            image_prefix: str = "frame_"):
    image_dir.mkdir(exist_ok=True, parents=True)

    if video_path.exists() is False:
        CONSOLE.print(f"[bold red] Error: Video does not exist {video_path}")
        sys.exit(1)

    num_frames = get_num_video_frames(video_path)
    if num_frames == 0:
        CONSOLE.print(f"[bold red] Error: Video has no frames {video_path}")
        sys.exit(1)
    CONSOLE.print(f"[bold blue] Number of frames in video: {num_frames}")

    downscale_chains = [f"[t{i}]scale=iw/{2**i}:ih/{2**i}[out{i}]" for i in range(num_downscales + 1)]
    downscale_dirs = [Path(str(image_dir) + (f"_{2**i}" if i > 0 else "")) for i in range(num_downscales + 1)]
    downscale_paths = [downscale_dirs[i] / f"{image_prefix}%05d.png" for i in range(num_downscales + 1)]

    for downscale_dir in downscale_dirs:
        downscale_dir.mkdir(parents=True, exist_ok=True)

    """
    This chain produces this FFmpeg filter string (for num_downscales=3):
    split=4[t0][t1][t2][t3];
    [t0]scale=iw/1:ih/1[out0];
    [t1]scale=iw/2:ih/2[out1];
    [t2]scale=iw/4:ih/4[out2];
    [t3]scale=iw/8:ih/8[out3]

    split=4[t0][t1][t2][t3]: takes the incoming video frame stream and duplicate it into 4 streams.
    - Then keeps stream t0 at original size and name the result out0.
    - Scales the rest of the streams t1,t2,t3 and name it outX.

    """
    downscale_chain = (
            f"split={num_downscales + 1}"
            + "".join([f"[t{i}]" for i in range(num_downscales + 1)])
            + ";"
            + ";".join(downscale_chains)
        )   
    spacing = num_frames // num_frames_target
    spacing_cmd_arg = ""
    pix_fmt_arg = ""
    if spacing > 1:
        CONSOLE.print(f"Extracting {math.ceil(num_frames/ spacing)} frames in evenly spaced intervals...")
        spacing_cmd_arg = f"thumbnail={spacing},setpts=N/TB,"
    else:
        CONSOLE.print(f"[bold red]Can't satisfy requested number of frames. Extracting all frames ")
        pix_fmt_arg = "-pix_fmt bgr8 "

    map_args = " ".join(
        f'-map "[out{i}]" "{downscale_paths[i]}"'
        for i in range(num_downscales + 1)
    )
    ffmpeg_cmd = (
        f'ffmpeg -i "{video_path}" '
        f'-vsync vfr '
        f'{pix_fmt_arg}'
        f'-filter_complex "{spacing_cmd_arg}{downscale_chain}" '
        f'{map_args}'
    )

    run_command(ffmpeg_cmd, verbose=verbose)

    final_num_frames = len(list(image_dir.glob("*.png")))
    CONSOLE.log(f"[bold green]Successfully converted video into {final_num_frames} frames")

    
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UAV video to frames.")
    parser.add_argument("-v", "--video_path", required=True, type=Path, help="Path to video.mp4")
    parser.add_argument("-i", "--image_dir", required=True, type=Path, help="Directory to extract images")
    parser.add_argument("--num_frames_target", required=True, type=int, help="Number of frames to extract")
    parser.add_argument("--num_downscales", type=int, default=3, help="Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the images by 2x, 4x, and 8x")
    args = parser.parse_args()

    convert_video_to_images(video_path=args.video_path,
                        image_dir=args.image_dir,
                        num_frames_target=args.num_frames_target,
                        num_downscales=args.num_downscales,
                        )
    


# video_path = Path("/home/panosvrachnos/Data/iDriving/custom_road_giannis/videos/DJI_20260520143903_0001_D.MP4")
# images_dir = Path("/home/panosvrachnos/Data/iDriving/custom_road_giannis/part2/images")

if __name__ == "__main__":
    start_time = perf_counter()

    main()

    end_time = perf_counter()
    total_time = end_time - start_time
    formatted_time = str(timedelta(seconds=total_time))
    print(f"Execution time: {formatted_time}")
