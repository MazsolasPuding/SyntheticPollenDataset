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

def select_pollen(pollen_path: Path, num_pollens: int):
    all_pollen = [pic for subdir in pollen_path.iterdir() if subdir.is_dir()   # Get all Images from DB
                    for pic in subdir.iterdir() if pic.suffix.lower() in PICTURE_FILE_FORMATS]
    selection = random.choices(all_pollen, k=num_pollens)
    return [Pollen(id=i,
                   path=Path(path),
                   y_position=random.randint(0, 480)
                   ) for i, path in enumerate(selection)]
    

def main(
        pollen_path: str,
        output_path: str,
        num_pollens: int,
        length: int,
        speed: int,
        fps=24,
        frame_size=(640, 480)
    ):

    # pollen = Pollen(1, Path('/Users/horvada/Git/Personal/PollenDB/POLLEN73S/hyptis_sp/hyptis_sp (35).jpg'), 300)
    # pollen_1 = Pollen(2, Path('/Users/horvada/Git/Personal/PollenDB/POLLEN73S/hyptis_sp/hyptis_sp (35).jpg'), -20)
    # pollen_2 = Pollen(3, Path('/Users/horvada/Git/Personal/PollenDB/POLLEN73S/hyptis_sp/hyptis_sp (35).jpg'), 100)
    # pollens = [pollen, pollen_1, pollen_2]

    pollens = select_pollen(Path(pollen_path), num_pollens)
    print(pollens)
    background = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255

    # Set up the video writer
    # num_frames = length * fps
    num_frames = int(np.ceil((frame_size[0] + pollens[0].width) / speed))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Store Sliding window details for plotting
    x_start_frame_values, x_end_frame_values, x_start_pollen_values, x_end_pollen_values = [], [], [], []

    for frame_idx in range(num_frames):
        frame = background.copy()
        for pollen in pollens:
            # Loop over each color channel
            for c in range(3):
                # print(f"PollenID: {pollen.id}: x_start_frame: {pollen.x_start_frame} - x_end_frame: {pollen.x_end_frame} - x_start_pollen: {pollen.x_start_pollen} - x_end_pollen: {pollen.x_end_pollen}")
                if pollen.x_end_pollen < 0:
                    # print(f"PollenID: {pollen.id} is out of frame")
                    continue
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

            if pollen.id == 1:
                x_start_frame_values.append(pollen.x_start_frame)
                x_end_frame_values.append(pollen.x_end_frame)
                x_start_pollen_values.append(pollen.x_start_pollen)
                x_end_pollen_values.append(pollen.x_end_pollen)
            pollen.position[0] += speed
        out.write(frame)


    out.release()
    print(f"Animation saved to {output_path}")

    plt.figure(figsize=(10, 8))
    print(pollens[1].width)
    plt.plot(range(-pollens[1].width, len(x_start_frame_values)-pollens[1].width), x_start_frame_values, label='x_start_frame')
    plt.plot(range(-pollens[1].width, len(x_end_frame_values)-pollens[1].width), x_end_frame_values, label='x_end_frame')
    plt.plot(range(-pollens[1].width, len(x_start_pollen_values)-pollens[1].width), x_start_pollen_values, label='x_start_pollen')
    plt.plot(range(-pollens[1].width, len(x_end_pollen_values)-pollens[1].width), x_end_pollen_values, label='x_end_pollen')
    plt.axvline(x=0, color='black', linestyle='dotted')
    plt.axvline(x=frame_size[0] - pollens[1].width, color='black', linestyle='dotted')
    plt.xlabel('Frame')
    plt.ylabel('X Position')
    plt.title('Sliding Windows of Frame and Pollen regions', fontsize=16)
    plt.legend()
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an animation of a pollen grain moving across the screen.")
    parser.add_argument("--pollen_path", type=str, help="Path to the segmented Pollen data source", default='/Users/horvada/Git/Personal/PollenDB/POLLEN73S')
    # parser.add_argument("--bg_path", type=str, help="Path to the segmented Background data source", required=False, default=None) TODO: Implement background
    parser.add_argument("--output_path", type=str, help="Path to save the animation", required=False, default='Analisis_Output/animation.avi')
    parser.add_argument("--num_pollens", type=int, help="The number of pollens in one frame", required=False, default=10)
    parser.add_argument("--length", type=int, help="Length of the animation", required=False, default=10)
    parser.add_argument("--speed", type=int, help="Speed of the pollen grain movement", required=False, default=1)
    parser.add_argument("--fps", type=int, help="Frames per second of the animation", required=False, default=24)
    parser.add_argument("--frame_size", type=tuple, help="Size of the animation frame", required=False, default=(640, 480))
    args = parser.parse_args()

    # pollen_path = '/Users/horvada/Git/PERSONAL/PollenSegmentation/Analisis_Output/hyptis_sp (35).jpg_rembg.png'
    main(
        pollen_path=args.pollen_path,
        output_path=args.output_path,
        num_pollens=args.num_pollens,
        length=args.length,
        speed=args.speed,
        fps=args.fps,
        frame_size=args.frame_size
    )


