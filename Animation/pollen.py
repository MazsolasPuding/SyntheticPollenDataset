"""
This module conatins the class Pollen, which is used in Animate.py.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

@dataclass
class Pollen:
    id: int
    # path: Path
    # y_position: int
    frame_size: tuple = (640, 480)
    # x_start_frame: int = 0
    # x_end_frame: int = 0
    # x_start_pollen: int = 0
    # x_end_pollen: int = 0
    # y_start_frame: int = 0
    # y_end_frame: int = 0
    # y_start_pollen: int = 0
    # y_end_pollen: int = 0


    def __init__(self, id: int, path: Path, y_position: int):
        self.id = id
        self.pollen_class = path.parent.name
        self.load_image(path)
        self.position = [-self.image.shape[1], y_position]

    def load_image(self, path: Path):
        self.image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if self.image.shape[2] == 4:  # Check for alpha channel
            # Extract BGR and Alpha channels
            self.bgr = self.image[:, :, :3]
            self.alpha = self.image[:, :, 3]
        else:
            self.bgr = self.image
            self.alpha = np.full(self.image.shape[:2], 255)  # Full opacity if no alpha channel

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]
    
    @property
    def x_start_frame(self):
        return max(self.position[0], 0)
    @property
    def x_end_frame(self):
        return min(self.position[0] + self.width, self.frame_size[0])

    @property
    def x_start_pollen(self):
        return max(0, -self.position[0])

    @property
    def x_end_pollen(self):
        return self.x_start_pollen + (self.x_end_frame - self.x_start_frame)

    @property
    def y_start_frame(self):
        return max(self.position[1], 0)

    @property
    def y_end_frame(self):
        return min(self.position[1] + self.height, self.frame_size[1])

    @property
    def y_start_pollen(self):
        return max(0, -self.position[1])

    @property
    def y_end_pollen(self):
        return self.y_start_pollen + (self.y_end_frame - self.y_start_frame)
