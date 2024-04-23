"""
This module conatins the class Pollen, which is used in Animate.py.

The Pollens position indexing starts in the upper left hand corner!
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
from opencv_transforms import functional as F


@dataclass
class Pollen:
    id: int
    path: Path
    position: list
    frame_size: tuple = (640, 480)
    pollen_to_frame_ratio: int = 10 # The pollens new hieght is 1/pollen_to_frame_ratio of the frame size
    annotate: bool = False
    augment: bool = False
    # List all attributes to be initialized here
    image: np.ndarray = None
    pollen_class: str = None
    pollen_class_index: int = None
    has_alpha: bool = False
    bgr: np.ndarray = None
    alpha: np.ndarray = None

    def __post_init__(self):
        self.load_image()
        if self.augment: self.augment_image()
        self.pollen_class = self.path.parent.name
        self.pollen_class_index = self.get_pollen_class_index()

        # Limit Y position offset, so at max only half of the pollen is outside the frame
        if self.position[1] < -self.image.shape[0] // 2:
            self.position[1] = -self.image.shape[0]
        if self.position[1] > self.frame_size[1] - self.image.shape[0] // 2:
            self.position[1] = self.frame_size[1] - self.image.shape[0]
        # If there is no X position, then the pollen selector is in continuos mode. Put the pollen right before the start of the frame.
        if not self.position[0]:
            self.position[0] = -self.image.shape[1]

    def load_image(self):
        # Load image
        self.image = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        # Resize image
        frame_height = self.frame_size[1]
        new_height = frame_height // self.pollen_to_frame_ratio
        scale_factor = new_height / self.image.shape[0]
        self.image = cv2.resize(self.image, (0, 0), fx=scale_factor, fy=scale_factor)
        
        if self.image.shape[2] == 4:  # Check for alpha channel
            # Extract BGR and Alpha channels
            self.has_alpha = True
            self.bgr = self.image[:, :, :3]
            self.alpha = self.image[:, :, 3]
        else:
            self.has_alpha = False
            self.bgr = self.image
            self.alpha = np.full(self.image.shape[:2], 255)  # Full opacity if no alpha channel

    def augment_image(self):
        # Apply color transformations
        self.bgr = F.adjust_hue(self.bgr, np.random.uniform(-0.01, 0.01))
        self.bgr = F.adjust_saturation(self.bgr, np.random.uniform(0.8, 1.2))
        self.bgr = F.adjust_brightness(self.bgr, np.random.uniform(0.8, 1.2))
        self.bgr = F.adjust_contrast(self.bgr, np.random.uniform(0.8, 1.2))
        self.bgr = F.adjust_gamma(self.bgr, np.random.uniform(0.8, 1.2))

        # Flip horizontally
        if np.random.rand() < 0.5: 
            self.bgr = cv2.flip(self.bgr, 1)
            self.alpha = cv2.flip(self.alpha, 1)
        # Flip vertically
        if np.random.rand() < 0.5:
            self.bgr = cv2.flip(self.bgr, 0)
            self.alpha = cv2.flip(self.alpha, 0)
        
        # Add noise
        mean = 0
        std_dev = 3
        noise = np.random.normal(mean, std_dev, self.bgr.shape)
        self.bgr = self.bgr + noise
        self.bgr = self.bgr.astype(np.uint8)
        # print(self.bgr.dtype, self.alpha.dtype, noise.dtype)

        # After applying all transformations, if there was an alpha channel, merge it back with the image
        if self.has_alpha:
            self.image = cv2.merge((self.bgr, self.alpha))


    def get_pollen_class_index(self):
        classes = sorted([cls.name for cls in self.path.parents[1].iterdir() if cls.is_dir()]) # get the grandparent of the path and get all classes (subdirectories inside)
        return classes.index(self.pollen_class)

    def check_annotation(self):
        # Set annotation flag to true in certaion visibility thresholds.
        half = self.position[0] + self.width / 2 # When Half of the image is inside the frame
        if half >= 0 and half <= self.frame_size[0]:
            self.annotate = True
        else:
            self.annotate = False

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
    
    @property
    def bounding_box(self):
        # [class index, all normalized x_center, y_center, width, height]
        if not self.annotate: return (None, None, None, None, None)

        x_center = (self.x_start_frame + (self.x_end_frame - self.x_start_frame) / 2) / self.frame_size[0]
        y_center = (self.position[1] + self.height / 2) / self.frame_size[1]
        normalized_width = (self.x_end_frame - self.x_start_frame) / self.frame_size[0]
        normalized_height = self.height / self.frame_size[1]

        return (self.pollen_class_index, x_center, y_center, normalized_width, normalized_height)
