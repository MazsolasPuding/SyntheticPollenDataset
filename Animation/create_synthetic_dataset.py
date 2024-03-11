"""
This script creates a synthetic dataset of pollen grains moving across a screen.
It also generates all the YOLO format annotations needed for detection tasks.

Usage:
    python create_synthetic_dataset.py [--pollen_path POLLEN_PATH] [--output_path OUTPUT_PATH] [--mode MODE]
                                       [--pollen_pos_mode POLLEN_POS_MODE] [--num_pollens NUM_POLLENS]
                                       [--length LENGTH] [--speed SPEED] [--fps FPS] [--frame_size FRAME_SIZE]
                                       [--save_video SAVE_VIDEO] [--save_frames SAVE_FRAMES] [--save_labels SAVE_LABELS]
                                       [--draw_bb DRAW_BB]

Arguments:
    --pollen_path (str): Path to the segmented Pollen data source (default: '/Users/horvada/Git/Personal/PollenDB/POLLEN73S')
    --output_path (str): Path to save the synthetic dataset (default: 'Analisis_Output/animation.avi')
    --mode (str): Mode of the dataset, either 'train' or 'test' (default: 'train')
    --pollen_pos_mode (str): Position mode of the pollens, either 'continuous' or 'discrete' (default: 'continuous')
    --num_pollens (int): The number of pollens in one frame (default: 1)
    --length (int): Length of the animation in seconds (default: 10)
    --speed (int): Speed of the pollen grain movement (default: 1)
    --fps (int): Frames per second of the animation (default: 24)
    --frame_size (tuple): Size of the animation frame (default: (640, 480))
    --save_video (bool): Whether to save the animation as a video (default: True)
    --save_frames (bool): Whether to save the individual frames of the animation (default: True)
    --save_labels (bool): Whether to save the YOLO format annotations (default: True)
    --draw_bb (bool): Whether to draw bounding boxes on the frames (default: True)
"""

import argparse
from datetime import datetime
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ != "__main__":
    from Animation.pollen import Pollen
else:
    from pollen import Pollen


class Singleton():
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

class SyntheticDatasetCreator(Singleton):
    def __init__(
            self,
            pollen_path: str,
            output_path: str,
            mode: str = "train",
            pollen_pos_mode: str = "continuous",
            num_pollens: int = 30,
            pollen_to_frame_ratio: int = 10,
            augment: bool = True,
            length: int = 30,
            speed: int = 10,
            fps: int = 30,
            frame_size: Tuple[int, int] = (1920, 1080),
            save_video: bool = True,
            save_frames: bool = True,
            save_labels: bool = True,
            draw_bb: bool = True
        ):
        self.pollen_path = pollen_path
        self.output_path = output_path
        self.mode = mode
        self.pollen_pos_mode = pollen_pos_mode
        self.num_pollens = num_pollens
        self.pollen_to_frame_ratio = pollen_to_frame_ratio
        self.augment = augment
        self.length = length
        self.speed = speed
        self.fps = fps
        self.frame_size = frame_size
        self.save_video = save_video
        self.save_frames = save_frames
        self.save_labels = save_labels
        self.draw_bb = draw_bb

        self.picture_formats = ['.jpg', '.jpeg', '.png', '.gif', '.tif']
        self.pollen_ID = 0
        self.all_pollen = [pic for subdir in (Path(pollen_path) / mode).iterdir() if subdir.is_dir()   # Get all Images from DB
                    for pic in subdir.iterdir() if pic.suffix.lower() in self.picture_formats]

        self.output_path = Path(output_path)
        self.pollen_path=Path(pollen_path)

        self.save_labels_path = self.output_path / mode / "labels"
        self.save_labels_path.mkdir(exist_ok=True, parents=True)
        self.save_frames_path = self.output_path / mode / "images"
        self.save_frames_path.mkdir(exist_ok=True, parents=True)
        self.save_video_path = self.output_path / "videos"
        self.save_video_path.mkdir(exist_ok=True, parents=True)

        current_time = datetime.now()
        timestamp_format = "%Y-%m-%d_%H-%M"
        self.start_timestamp_str = current_time.strftime(timestamp_format)

        self.pollens = self.select_pollen(
                                num_pollens,
                                init=True
                            )
        
        background_r = np.ones((frame_size[1], frame_size[0]), dtype=np.uint8) * 219
        background_g = np.ones((frame_size[1], frame_size[0]), dtype=np.uint8) * 210
        background_b = np.ones((frame_size[1], frame_size[0]), dtype=np.uint8) * 162
        self.background = np.stack((background_b, background_g, background_r), axis=-1)

        # Set up the video writer
        self.num_frames = length * fps
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.video_out = cv2.VideoWriter(str(self.save_video_path / f"{mode}.avi"), fourcc, fps, frame_size)

    def __call__(self):
        self.create_synthetic_dataset() # Call the create_synthetic_dataset function when the instance is called as a function


    def create_synthetic_dataset(self) -> None:
        for frame_idx in tqdm(range(self.num_frames)):

            self.frame = self.background.copy()
            self.update_pollen()
            self.frame = self.animate(self.pollens, self.frame)
            frame_name = self.get_frame_name(frame_idx)

            if self.save_labels:  self.save_annotation(str(self.save_labels_path / frame_name) + ".txt", self.pollens)
            if self.save_frames:  cv2.imwrite(str(self.save_frames_path / frame_name) + ".jpg", self.frame)
            if self.draw_bb:  self.frame = self.draw_bounding_boxes(self.frame, self.pollens)
            if self.save_video:  self.video_out.write(self.frame)
            if self.pollen_pos_mode == "continuous":  self.shift_pollen(self.pollens, self.speed)

        if self.save_video: self.video_out.release()
        print(f"Output saved to {self.output_path}")


    def get_frame_name(self, frame_idx: int) -> str:
        frame_name = f"{self.start_timestamp_str}_frame_{frame_idx:06d}"
        return frame_name

    def select_pollen(self, num_pollens: int, init: bool = False):
        selection = random.choices(self.all_pollen, k=num_pollens)
        pollen_selection = []
        for i, path in enumerate(selection):
            pollen_selection.append(Pollen(id=self.pollen_ID,
                                        path=Path(path),
                                        position=[random.randint(0, self.frame_size[0]) if init else None, random.randint(0, self.frame_size[1])],
                                        frame_size=self.frame_size,
                                        pollen_to_frame_ratio=self.pollen_to_frame_ratio,
                                        augment=self.augment))
            self.pollen_ID += 1
        return pollen_selection if init else pollen_selection[0]

    def update_pollen(self):    
            if self.pollen_pos_mode == "continuous":
                for pollen in self.pollens.copy():
                    if pollen.x_end_pollen < 0:
                        self.pollens.remove(pollen)
                        self.pollens.append(
                            self.select_pollen(num_pollens=1)
                        )
            elif self.pollen_pos_mode == "random":
                self.pollens = self.select_pollen(
                                        num_pollens=self.num_pollens,
                                        init=True
                                    )

    def animate(
            self,
            pollens: List[Pollen],
            frame: np.array,
        ):
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

    def draw_bounding_boxes(self, frame: np.array, pollens: List[Pollen]):
        for pollen in pollens:
            if not pollen.annotate:
                continue
            height, width, _ = frame.shape
            pos_dot_color = (0, 0, 255)  # Green color
            frame_dot_color = (255, 0, 0)  # Green color
            box_color = (0, 255, 0)  # Green color for bounding box

            # Draw the dot onto the corner
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

    def save_annotation(self, path: str, pollens: List[Pollen]):
        annotations = [pollen.bounding_box for pollen in pollens if pollen.annotate]

        with open(path, 'w') as f:
            for annotation in annotations:
                # Convert each annotation to a string and write to the file
                annotation_str = ' '.join(str(x) for x in annotation)
                f.write(annotation_str + '\n')

    def shift_pollen(self, pollens: List[Pollen], speed: int):
        for pollen in pollens:
                pollen.position[0] += speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an animation of a pollen grain moving across the screen.")
    parser.add_argument("--pollen_path", type=str, help="Path to the segmented Pollen data source", default='/Users/horvada/Git/Personal/datasets/POLLEN73S_PROCESSED')
    # parser.add_argument("--bg_path", type=str, help="Path to the segmented Background data source", required=False, default=None) TODO: Implement background
    parser.add_argument("--output_path", type=str, required=False, default='/Users/horvada/Git/Personal/datasets')
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--pollen_pos_mode", type=str, choices=["continuous", "random"], default="continuous")

    parser.add_argument("--num_pollens", type=int, help="The number of pollens in one frame", required=False, default=40)
    parser.add_argument("--pollen_to_frame_ratio", type=int, help="Frame Height / pollen height = pollen_to_frame_ratio", required=False, default=15)
    parser.add_argument("--augment", type=bool, help="Set true if pollen image augmentation is needed", required=False, default=True)
    parser.add_argument("--length", type=int, help="Length of the animation [s]", required=False, default=30)
    parser.add_argument("--speed", type=int, help="Speed of the pollen grain movement", required=False, default=10)
    parser.add_argument("--fps", type=int, help="Frames per second of the animation", required=False, default=30)
    parser.add_argument("--frame_size", type=tuple, help="Size of the animation frame", required=False, default=(1920, 1080))

    parser.add_argument("--save_video", type=bool, required=False, default=True)
    parser.add_argument("--save_frames", type=bool, required=False, default=False)
    parser.add_argument("--save_labels", type=bool, required=False, default=False)
    parser.add_argument("--draw_bb", type=bool, required=False, default=True)
    args = parser.parse_args()


    sdc = SyntheticDatasetCreator(
        pollen_path=args.pollen_path,
        output_path=args.output_path,
        mode=args.mode,
        pollen_pos_mode = "continuous",
        num_pollens=args.num_pollens,
        pollen_to_frame_ratio=args.pollen_to_frame_ratio,
        augment=args.augment,
        length=args.length,
        speed=args.speed,
        fps=args.fps,
        frame_size=args.frame_size,
        save_video=args.save_video,
        save_frames=args.save_frames,
        save_labels=args.save_labels,
        draw_bb=args.draw_bb
    )
    sdc()


