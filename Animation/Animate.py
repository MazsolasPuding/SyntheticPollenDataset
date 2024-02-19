"""
This script creates an animation of a pollen grain moving across the screen.

Usage:
    python Animate.py [--pollen_path POLLEN_PATH] [--output_path OUTPUT_PATH] [--num_pollens NUM_POLLENS]
                      [--length LENGTH] [--speed SPEED] [--fps FPS] [--frame_size FRAME_SIZE]

Arguments:
    --pollen_path (str): Path to the segmented Pollen data source (default: '/Users/horvada/Git/Personal/PollenDB/POLLEN73S')
    --output_path (str): Path to save the animation (default: 'Analisis_Output/animation.avi')
    --num_pollens (int): The number of pollens in one frame (default: 1)
    --length (int): Length of the animation in seconds (default: 10)
    --speed (int): Speed of the pollen grain movement (default: 1)
    --fps (int): Frames per second of the animation (default: 24)
    --frame_size (tuple): Size of the animation frame (default: (640, 480))
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pollen import Pollen

PICTURE_FILE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.tif']
ID = 0

def select_pollen(pollen_path: Path, num_pollens: int, frame_size: tuple, init: bool = False):
    global ID
    all_pollen = [pic for subdir in pollen_path.iterdir() if subdir.is_dir()   # Get all Images from DB
                    for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]
    # print(set([all_pollen[i].parent.name for i in range(len(all_pollen))])) # Print all classes
    selection = random.choices(all_pollen, k=num_pollens)
    pollen_selection = []
    for i, path in enumerate(selection):
        pollen_selection.append(Pollen(id=ID,
                                       path=Path(path),
                                       position=[random.randint(0, frame_size[0]) if init else None, random.randint(0, frame_size[1])],
                                       frame_size=frame_size))
        ID += 1
    return pollen_selection if init else pollen_selection[0]

def animate(
        pollens: list,
        frame: np.array,
        speed: int,
        pollen_path: str,
        frame_size: tuple
    ):
    for pollen in pollens.copy():
        if pollen.x_end_pollen < 0:
            pollens.remove(pollen)
            print(f"REMOVED ID: {pollen.id}")
            pollens.append(select_pollen(pollen_path=Path(pollen_path),
                                            num_pollens=1,
                                            frame_size=frame_size))
    for pollen in pollens:
        # Loop over each color channel
        for c in range(3):
            # Define the regions of the Frame and the Pollen (Alpha region is the same columns and rows as the BGR region)
            frame_region = frame[pollen.y_start_frame:pollen.y_end_frame, pollen.x_start_frame:pollen.x_end_frame, c]
            pollen_region = pollen.bgr[ pollen.y_start_pollen : pollen.y_end_pollen, pollen.x_start_pollen : pollen.x_end_pollen, c]
            alpha_region = pollen.alpha[ pollen.y_start_pollen : pollen.y_end_pollen, pollen.x_start_pollen : pollen.x_end_pollen]

            # Calculate the weights for the image and frame regions
            image_weight = alpha_region / 255.0
            frame_weight = 1 - image_weight

            # Blend the image and frame regions
            blended_region = pollen_region * image_weight + frame_region * frame_weight
            frame[pollen.y_start_frame : pollen.y_end_frame, pollen.x_start_frame : pollen.x_end_frame, c] = blended_region.astype(np.uint8)

        pollen.position[0] += speed
    return frame

def main(
        pollen_path: str,
        output_path: str,
        num_pollens: int,
        length: int,
        speed: int,
        fps=24,
        frame_size=(640, 480)
    ):
    pollens = select_pollen(pollen_path=Path(pollen_path),
                            num_pollens=num_pollens,
                            frame_size=frame_size,
                            init=True)
    background = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255

    # Set up the video writer
    num_frames = length * fps
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame_idx in range(num_frames):
        frame = background.copy()
        frame = animate(pollens, frame, speed, pollen_path, frame_size)
        out.write(frame)

    out.release()
    print(f"Animation saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an animation of a pollen grain moving across the screen.")
    parser.add_argument("--pollen_path", type=str, help="Path to the segmented Pollen data source", default='/Users/horvada/Git/Personal/PollenDB/POLLEN73S')
    # parser.add_argument("--bg_path", type=str, help="Path to the segmented Background data source", required=False, default=None) TODO: Implement background
    parser.add_argument("--output_path", type=str, help="Path to save the animation", required=False, default='Analisis_Output/animation.avi')
    parser.add_argument("--num_pollens", type=int, help="The number of pollens in one frame", required=False, default=30)
    parser.add_argument("--length", type=int, help="Length of the animation [s]", required=False, default=30)
    parser.add_argument("--speed", type=int, help="Speed of the pollen grain movement", required=False, default=10)
    parser.add_argument("--fps", type=int, help="Frames per second of the animation", required=False, default=30)
    parser.add_argument("--frame_size", type=tuple, help="Size of the animation frame", required=False, default=(1920, 1080))
    args = parser.parse_args()


    main(
        pollen_path=args.pollen_path,
        output_path=args.output_path,
        num_pollens=args.num_pollens,
        length=args.length,
        speed=args.speed,
        fps=args.fps,
        frame_size=args.frame_size
    )


