from rembg import remove
import io
import cv2
import numpy as np
from PIL import Image

def apply_rembg_remove(image,
                       return_img: bool = True,
                       keep_bg: bool = False,
                       image_save_path: str = None, 
                       bg_save_path: str = None):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    result = remove(image)

    if keep_bg:
        mask_resized = cv2.resize(result, [image.shape[1], image.shape[0]])
        gray_image = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        inverted_mask = cv2.bitwise_not(binary_mask)
        mask = np.zeros(image.shape[:2], dtype="uint8")
        # print(f"image shape: {image.shape}, Mask Shape: {inverted_mask.shape}")
        background = cv2.bitwise_and(image, image, mask=inverted_mask)

    if image_save_path:
        cv2.imwrite(image_save_path, result)
    if bg_save_path:
        cv2.imwrite(bg_save_path, background)

    if return_img:
        image_rgb = cv2.cvtColor(background, cv2.COLOR_BGRA2RGB) if keep_bg else cv2.cvtColor(result, cv2.COLOR_BGRA2RGB)
        return image_rgb
  

if __name__ == '__main__':
    input_path = 'D:/UNI/PTE/Pollen/PollenDB/POLLEN73S/hyptis_sp/hyptis_sp (35).jpg'
    output_path = 'D:/UNI/PTE/Pollen/Segmentation/Analisis_Output/hyptis_sp (35).jpg_rembg.png'
    input = cv2.imread(input_path)


    output = apply_rembg_remove(input, output_path)
    cv2.imwrite(output_path, output)
