"""
This module conatins the class Pollen, which is used in Animate.py.

The Pollens position indexing starts in the upper left hand corner!
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2
from torchvision.transforms import v2

@dataclass
class Pollen:
    id: int
    path: Path
    position: list
    frame_size: tuple = (640, 480)
    pollen_to_frame_ratio: int = 10 # The pollens new hieght is 1/pollen_to_frame_ratio of the frame size
    annotate: bool = False

    def __post_init__(self):
        self.load_image(self.path)
        self.pollen_class = self.path.parent.name
        self.pollen_class_index = self.get_pollen_class_index()
        # Limit Y position offset, so at max only half of the pollen is outside the frame
        if self.position[1] < -self.image.shape[0] // 2: self.position[1] = -self.image.shape[0]
        if self.position[1] > self.frame_size[1] - self.image.shape[0] // 2: self.position[1] = self.frame_size[1] - self.image.shape[0]
        if not self.position[0]: self.position[0] = -self.image.shape[1]

    def load_image(self, path: Path):
        # Load image
        self.image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        # Resize image
        frame_height = self.frame_size[1]
        new_height = frame_height // self.pollen_to_frame_ratio
        scale_factor = new_height / self.image.shape[0]
        self.image = cv2.resize(self.image, (0, 0), fx=scale_factor, fy=scale_factor)
        # self.image = self.augment_image(self.image)
        
        if self.image.shape[2] == 4:  # Check for alpha channel
            # Extract BGR and Alpha channels
            self.bgr = self.image[:, :, :3]
            self.alpha = self.image[:, :, 3]
        else:
            self.bgr = self.image
            self.alpha = np.full(self.image.shape[:2], 255)  # Full opacity if no alpha channel

    def augment_image(self, image):
        # Handle images with and without alpha channel
        if image.shape[2] == 4:  # Image has an alpha channel
            alpha_channel = image[:, :, 3]
            image = image[:, :, :3]  # Remove alpha channel for processing
        else:
            alpha_channel = None

        # Random Horizontal Flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)  # 1 means flipping around y-axis
        
        # Random Vertical Flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)  # 0 means flipping around x-axis

        # Color Jitter (adjusting brightness, contrast, saturation, and hue is more complex in OpenCV)
        # For simplicity, only demonstrating brightness and contrast adjustment here
        brightness_factor = np.random.uniform(0.5, 1.5)  # Adjust this range as needed
        contrast_factor = np.random.uniform(0.5, 1.5)  # Adjust this range as needed
        image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor * 255.0 - contrast_factor * 127.5)

        # Random Rotation
        angle = np.random.uniform(0, 360)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

        # Random Affine Transformation (Combines translation, rotation, and scaling)
        # Note: For simplicity, this example applies only rotation here. You can extend this with random translations and scaling.
        pts1 = np.float32([[10, 10], [20, 10], [10, 20]])
        pts2 = pts1 + np.random.uniform(-5, 5, pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Random Perspective Transformation
        pts1 = np.float32([[5, 5], [20, 5], [5, 20], [20, 20]])
        pts2 = pts1 + np.random.uniform(-10, 10, pts1.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

        # Gaussian Blur
        if np.random.rand() > 0.5:
            ksize = int(2 * round(np.random.uniform(1, 4)) + 1)  # Kernel size should be odd
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # Re-attach alpha channel if it was present
        if alpha_channel is not None:
            image = cv2.merge((image, alpha_channel))

        return image

    def get_pollen_class_index(self):
        classes = sorted([cls.name for cls in self.path.parents[1].iterdir() if cls.is_dir()]) # get the grandparent of the path and get all classes (subdire4ctories inside)
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


    
    
