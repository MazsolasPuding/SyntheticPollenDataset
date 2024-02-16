import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet101

def remove_background(image_path):
    # Load the pre-trained DeepLab model
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

        # Create a mask for the foreground
    mask = output_predictions == 15  # In COCO dataset, 15 is the index for "person" class; adjust this for your needs
    mask = mask.cpu().numpy()

    # Apply the mask to the image
    foreground = np.array(image) * np.expand_dims(mask, axis=-1)
    foreground = Image.fromarray(foreground.astype(np.uint8))

    # Display the original and foreground images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(foreground)
    plt.title("Foreground Image")
    plt.axis("off")

    plt.show()


if __name__ == '__main__':
    # Provide the path to your image
    image_path = '/Users/horvada/Git/Personal/PollenDB/POLLEN73S/hyptis_sp/hyptis_sp (35).jpg'
    image_path = '/Users/horvada/Git/Personal/PollenDB/POLLEN73S/acrocomia_aculeta/Figura16.TIF'
    remove_background(image_path)