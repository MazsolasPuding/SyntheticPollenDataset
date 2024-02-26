"""
This script creates an animation of pollen grains moving across a screen.
And All the YOLO formati annotations needed for cetection tasks.

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
from datetime import datetime
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pollen import Pollen

PICTURE_FILE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.tif']
POLLEN_ID = 0
FRAME_ID = 0

def get_frame_name(timestamp_str: str):
    global FRAME_ID
    file_name = f"{timestamp_str}_frame_{FRAME_ID:06d}"
    FRAME_ID += 1
    return file_name

def select_pollen(pollen_path: Path, num_pollens: int, frame_size: tuple, init: bool = False):
    global POLLEN_ID
    all_pollen = [pic for subdir in pollen_path.iterdir() if subdir.is_dir()   # Get all Images from DB
                    for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]
    # print(set([all_pollen[i].parent.name for i in range(len(all_pollen))])) # Print all classes
    selection = random.choices(all_pollen, k=num_pollens)
    pollen_selection = []
    for i, path in enumerate(selection):
        pollen_selection.append(Pollen(id=POLLEN_ID,
                                       path=Path(path),
                                       position=[random.randint(0, frame_size[0]) if init else None, random.randint(0, frame_size[1])],
                                       frame_size=frame_size))
        POLLEN_ID += 1
    return pollen_selection if init else pollen_selection[0]

def animate(
        pollens: List[Pollen],
        frame: np.array,
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
        pollen.check_annotation()
        # Loop over each color channel
        for c in range(3):
            # Define the regions of the Frame and the Pollen (Alpha region is the same columns and rows as the BGR region)
            frame_region = frame[ pollen.y_start_frame:pollen.y_end_frame, pollen.x_start_frame:pollen.x_end_frame, c ]
            pollen_region = pollen.bgr[ pollen.y_start_pollen : pollen.y_end_pollen, pollen.x_start_pollen : pollen.x_end_pollen, c ]
            alpha_region = pollen.alpha[ pollen.y_start_pollen : pollen.y_end_pollen, pollen.x_start_pollen : pollen.x_end_pollen ]

            # Calculate the weights for the image and frame regions
            image_weight = alpha_region / 255.0
            frame_weight = 1 - image_weight

            # Blend the image and frame regions
            blended_region = pollen_region * image_weight + frame_region * frame_weight
            frame[pollen.y_start_frame : pollen.y_end_frame, pollen.x_start_frame : pollen.x_end_frame, c] = blended_region.astype(np.uint8)

    return frame

def draw_bounding_boxes(frame: np.array, pollens: List[Pollen]):
    for pollen in pollens:
        if not pollen.annotate:
            continue
        height, width, _ = frame.shape
        pos_dot_color = (0, 0, 255)  # Green color
        frame_dot_color = (255, 0, 0)  # Green color
        box_color = (0, 255, 0)  # Green color for bounding box

        # Draw the dot onto the midle
        # dot_radius = 5
        # cv2.circle(frame, pollen.position, dot_radius, pos_dot_color, -1)
        # cv2.circle(frame, [pollen.x_start_frame, pollen.y_start_frame], 3, frame_dot_color, -1)
        # cv2.line(frame, (pollen.x_start_frame, 0), (pollen.x_start_frame, height), [30, 180, 255], thickness=2)
        # cv2.line(frame, (pollen.x_end_frame, 0), (pollen.x_end_frame, height), [180, 30, 255], thickness=2)

        # Draw bounding boxes
        class_index, x_center, y_center, bbox_width, bbox_height = pollen.bounding_box
        x_min = int((x_center - bbox_width / 2) * width)
        y_min = int((y_center - bbox_height / 2) * height)
        x_max = int((x_center + bbox_width / 2) * width)
        y_max = int((y_center + bbox_height / 2) * height)

        # Draw bounding box
        thickness = 2
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, thickness)
    return frame

def save_annotation(path: str, pollens: List[Pollen]):
    annotations = [pollen.bounding_box for pollen in pollens if pollen.annotate]

    with open(path, 'w') as f:
        for annotation in annotations:
            # Convert each annotation to a string and write to the file
            annotation_str = ' '.join(str(x) for x in annotation)
            f.write(annotation_str + '\n')

def shift_pollen(pollens: List[Pollen], speed: int):
    for pollen in pollens:
            pollen.position[0] += speed


def main(
        pollen_path: str,
        output_path: str,
        mode: str = "train",
        num_pollens: int = 30,
        length: int = 30,
        speed: int = 10,
        fps: int = 30,
        frame_size: Tuple[int, int] = (1920, 1080),
        save_video: bool = True,
        save_frames: bool = True,
        save_labels: bool = True,
        draw_bb: bool = True
    ):
    output_path = Path(output_path)
    pollen_path=Path(pollen_path)
    current_time = datetime.now()
    timestamp_format = "%Y-%m-%d_%H-%M"
    timestamp_str = current_time.strftime(timestamp_format)

    pollens = select_pollen(pollen_path / mode,
                            num_pollens,
                            frame_size,
                            init=True)
    background = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255

    # Set up the video writer
    num_frames = length * fps
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    if save_video: out = cv2.VideoWriter(str(output_path / "animation.avi"), fourcc, fps, frame_size)

    for frame_idx in range(num_frames):
        frame = background.copy()
        frame = animate(pollens, frame, str(pollen_path / mode), frame_size)
        frame_name = get_frame_name(timestamp_str)
        if save_labels: save_annotation(str(output_path / "labels" / mode / frame_name) + ".txt", pollens)
        if save_frames: cv2.imwrite(str(output_path / "images" / mode / frame_name) + ".jpg", frame)
        if draw_bb: frame = draw_bounding_boxes(frame, pollens)
        if save_video: out.write(frame)
        shift_pollen(pollens, speed)

    if save_video: out.release()
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an animation of a pollen grain moving across the screen.")
    parser.add_argument("--pollen_path", type=str, help="Path to the segmented Pollen data source", default='/Users/horvada/Git/Personal/datasets/POLLEN73S_PROCESSED')
    # parser.add_argument("--bg_path", type=str, help="Path to the segmented Background data source", required=False, default=None) TODO: Implement background
    parser.add_argument("--output_path", type=str, required=False, default='/Users/horvada/Git/Personal/datasets')
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="train")

    parser.add_argument("--num_pollens", type=int, help="The number of pollens in one frame", required=False, default=40)
    parser.add_argument("--length", type=int, help="Length of the animation [s]", required=False, default=30)
    parser.add_argument("--speed", type=int, help="Speed of the pollen grain movement", required=False, default=10)
    parser.add_argument("--fps", type=int, help="Frames per second of the animation", required=False, default=30)
    parser.add_argument("--frame_size", type=tuple, help="Size of the animation frame", required=False, default=(1920, 1080))

    parser.add_argument("--save_video", type=bool, required=False, default=True)
    parser.add_argument("--save_frames", type=bool, required=False, default=False)
    parser.add_argument("--save_labels", type=bool, required=False, default=False)
    parser.add_argument("--draw_bb", type=bool, required=False, default=True)
    args = parser.parse_args()
    # '/Users/horvada/Git/Personal/PollenDB/POLLEN73S'

    # 'D:/UNI/PTE/Pollen/datasets/POLLEN73S_Segmented_Pollen_Split'
    # "D:/UNI/PTE/Pollen/datasets/SYNTH_dataset_POLLEN73S"


    main(
        pollen_path=args.pollen_path,
        output_path=args.output_path,
        mode=args.mode,
        num_pollens=args.num_pollens,
        length=args.length,
        speed=args.speed,
        fps=args.fps,
        frame_size=args.frame_size,
        save_video=args.save_video,
        save_frames=args.save_frames,
        save_labels=args.save_labels,
        draw_bb=args.draw_bb
    )


