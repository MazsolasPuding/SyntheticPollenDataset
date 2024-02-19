import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pollen import Pollen

def main(
        pollen_path: str,
        output_path: str,
        num_pollens: int,
        length: int,
        speed: int,
        fps=24,
        frame_size=(640, 480)
    ):

    pollen = Pollen(id=0,
                    path=Path('/Users/horvada/Git/Personal/PollenDB/POLLEN73S/hyptis_sp/hyptis_sp (35).jpg'),
                    position=[0, frame_size[1] // 2],
                    frame_size=frame_size)
    background = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255

    # Set up the video writer
    # num_frames = length * fps
    num_frames = int(np.ceil((frame_size[0] + pollen.width) / speed))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Store Sliding window details for plotting
    x_start_frame_values, x_end_frame_values, x_start_pollen_values, x_end_pollen_values = [], [], [], []

    for frame_idx in range(num_frames):
        frame = background.copy()

        # Calculate the parts of the image that are inside the frame boundaries
        x_start_frame = max(pollen.position[0], 0)
        x_end_frame = min(pollen.position[0] + pollen.width, frame_size[0])
        x_start_pollen = max(0, -pollen.position[0])
        x_end_pollen = x_start_pollen + (x_end_frame - x_start_frame)

        y_start_frame = max(pollen.position[1], 0)
        y_end_frame = min(pollen.position[1] + pollen.height, frame_size[1])
        y_start_pollen = max(0, -pollen.position[1])
        y_end_pollen = y_start_pollen + (y_end_frame - y_start_frame)

        x_start_frame_values.append(x_start_frame)
        x_end_frame_values.append(x_end_frame)
        x_start_pollen_values.append(x_start_pollen)
        x_end_pollen_values.append(x_end_pollen)

        # Loop over each color channel
        for c in range(3):
            # Define the regions of the Frame and the Pollen (Alpha region is the same columns and rows as the BGR region)
            frame_region = frame[y_start_frame:y_end_frame, x_start_frame:x_end_frame, c]
            pollen_region = pollen.bgr[y_start_pollen:y_end_pollen, x_start_pollen:x_end_pollen, c]
            alpha_region = pollen.alpha[y_start_pollen:y_end_pollen, x_start_pollen:x_end_pollen]

            # Calculate the weights for the image and frame regions
            image_weight = alpha_region / 255.0
            frame_weight = 1 - image_weight

            # Blend the image and frame regions
            blended_region = pollen_region * image_weight + frame_region * frame_weight
            frame[y_start_frame:y_end_frame, x_start_frame:x_end_frame, c] = blended_region.astype(np.uint8)

        out.write(frame)
        pollen.position[0] += speed

    out.release()
    print(f"Animation saved to {output_path}")

    plt.figure(figsize=(10, 8))
    plt.plot(range(-pollen.width, frame_size[0], speed), x_start_frame_values, label='x_start_frame')
    plt.plot(range(-pollen.width, frame_size[0], speed), x_end_frame_values, label='x_end_frame')
    plt.plot(range(-pollen.width, frame_size[0], speed), x_start_pollen_values, label='x_start_pollen')
    plt.plot(range(-pollen.width, frame_size[0], speed), x_end_pollen_values, label='x_end_pollen')
    plt.axvline(x=0, color='black', linestyle='dotted')
    plt.axvline(x=frame_size[0] - pollen.width, color='black', linestyle='dotted')
    plt.xlabel('Frame')
    plt.ylabel('X Position')
    plt.title('Sliding Windows of Frame and Pollen regions', fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an animation of a pollen grain moving across the screen.")
    parser.add_argument("--pollen_path", type=str, help="Path to the segmented Pollen data source", default='/Users/horvada/Git/Personal/PollenDB/POLLEN73S')
    # parser.add_argument("--bg_path", type=str, help="Path to the segmented Background data source", required=False, default=None) TODO: Implement background
    parser.add_argument("--output_path", type=str, help="Path to save the animation", required=False, default='Analisis_Output/animation.avi')
    parser.add_argument("--num_pollens", type=int, help="The number of pollens in one frame", required=False, default=1)
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


