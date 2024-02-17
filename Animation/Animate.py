import cv2
import numpy as np

def create_animation(pollen_path, speed, output_path='animation.avi', frame_size=(640, 480), fps=24):
    pollen = cv2.imread(pollen_path, cv2.IMREAD_UNCHANGED)
    if pollen.shape[2] == 4:  # Check for alpha channel
        # Extract BGR and Alpha channels
        bgr = pollen[:, :, :3]
        alpha = pollen[:, :, 3]
    else:
        bgr = pollen
        alpha = np.full(pollen.shape[:2], 255)  # Full opacity if no alpha channel

    pollen_height, pollen_width = pollen.shape[:2]
    background = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255

    # Adjust num_frames to account for the entire movement across the frame
    num_frames = int(np.ceil((frame_size[0] + pollen_width) / speed))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame_idx in range(num_frames):
        x_offset = speed * frame_idx - pollen_width
        frame = background.copy()
        y_offset = (frame_size[1] - pollen_height) // 2

        # Calculate the parts of the image that are inside the frame boundaries
        x_start_frame = max(x_offset, 0)
        x_end_frame = min(x_offset + pollen_width, frame_size[0])
        x_start_pollen = max(0, -x_offset)
        x_end_pollen = x_start_pollen + (x_end_frame - x_start_frame)

        for c in range(3):  # For each color channel
            frame[y_offset:y_offset+pollen_height, x_start_frame:x_end_frame, c] = (
                bgr[:, x_start_pollen:x_end_pollen, c] * (alpha[:, x_start_pollen:x_end_pollen] / 255.0) +
                frame[y_offset:y_offset+pollen_height, x_start_frame:x_end_frame, c] * 
                (1 - alpha[:, x_start_pollen:x_end_pollen] / 255.0)
            ).astype(np.uint8)

        out.write(frame)

    out.release()
    print(f"Animation saved to {output_path}")



# Example usage
pollen_path = '/Users/horvada/Git/PERSONAL/PollenSegmentation/Analisis_Output/hyptis_sp (35).jpg_rembg.png'
create_animation(pollen_path, speed=5, output_path='pollen_animation.avi', fps=30)
