# import cv2
# import numpy as np
# import os

# def find_dominant_color(image):
#     """
#     Find the dominant color in the image.
#     :param image: Input image.
#     :return: Dominant color (B, G, R).
#     """
#     # Resize the image for faster computation
#     resized_image = cv2.resize(image, (1, 1))
#     dominant_color = resized_image[0, 0].tolist()
#     return dominant_color


# def extend_image(img_path, border_size=50, edge_size=5):
#     """
#     Extends the image using the dominant color of its edges.
#     :param img_path: Path to the segmented pollen image.
#     :param border_size: Size of the border to be added.
#     :param edge_size: Size of the edge to be used for computing the dominant color.
#     :return: Extended image.
#     """
#     # Read the image
#     img = cv2.imread(img_path)

#     # Extract the outermost pixels from each edge
#     top_edge = img[0:edge_size, :]
#     bottom_edge = img[-edge_size:, :]
#     left_edge = img[:, 0:edge_size]
#     right_edge = img[:, -edge_size:]
    
#     # Reshape the edge regions to 1D arrays
#     top_edge = top_edge.reshape(-1, 3)
#     bottom_edge = bottom_edge.reshape(-1, 3)
#     left_edge = left_edge.reshape(-1, 3)
#     right_edge = right_edge.reshape(-1, 3)
    
#     # Concatenate all edge pixels
#     edges = np.concatenate((top_edge, bottom_edge, left_edge, right_edge), axis=0)

#     # Compute the dominant color by averaging the RGB values of the edge pixels
#     dominant_color = np.mean(edges, axis=0)

#     # Extend the image with the dominant color
#     extended_img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=dominant_color)
    
#     return extended_img



import os
import cv2
import numpy as np


def extend_image(img_path, border_size=50, patch_size=5):
    """
    Extends the image using PatchMatch-based inpainting.
    :param img_path: Path to the segmented pollen image.
    :param border_size: Size of the border to be added.
    :param patch_size: Size of the patch used for inpainting.
    :return: Extended image.
    """
    # Read the image
    img = cv2.imread(img_path)

    # Create a mask that includes the border region to be inpainted
    height, width = img.shape[:2]
    mask = np.ones((height + 2 * border_size, width + 2 * border_size), dtype=np.uint8)
    mask[border_size:-border_size, border_size:-border_size] = 0

    # Extend the original image by adding border
    extended_img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_REFLECT)

    # Enhance the details of the extended image
    extended_img = cv2.detailEnhance(extended_img)

    # Inpaint the border region using PatchMatch
    result = cv2.inpaint(extended_img, mask, inpaintRadius=patch_size, flags=cv2.INPAINT_TELEA)

    return result

input_folder = 'PollenDB\Kaggle\input'
output_folder = 'PollenDB\Kaggle\output'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        extended_image = extend_image(img_path, border_size=50, patch_size=5)
        
        # Create a complete path for the output image
        output_img_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_img_path, extended_image)
