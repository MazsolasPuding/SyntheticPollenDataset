import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_average_histogram(image_paths):
    # Initialize sum of histograms for each channel
    sum_hist_hue = np.zeros((180,))
    sum_hist_saturation = np.zeros((256,))
    sum_hist_value = np.zeros((256,))
    num_images = len(image_paths)
    
    # Parameters for histogram calculation
    hist_bins = 256  # Number of bins for each channel
    hist_range = [0, 256]  # The range for the histogram
    
    for img_path in image_paths:
        # Read the image
        image = cv2.imread(str(img_path))
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        hist_hue = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        hist_saturation = cv2.calcHist([hsv_image], [1], None, [hist_bins], hist_range)
        hist_value = cv2.calcHist([hsv_image], [2], None, [hist_bins], hist_range)
        
        # Sum up the histograms
        sum_hist_hue += hist_hue.flatten()
        sum_hist_saturation += hist_saturation.flatten()
        sum_hist_value += hist_value.flatten()
    
    # Calculate the average histograms
    avg_hist_hue = sum_hist_hue / num_images
    avg_hist_saturation = sum_hist_saturation / num_images
    avg_hist_value = sum_hist_value / num_images
    
    return avg_hist_hue, avg_hist_saturation, avg_hist_value


def plot_histograms(histograms, title):
    # Plot the histograms
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b']
    labels = ['Hue', 'Saturation', 'Value']
    
    for i, hist in enumerate(histograms):
        plt.plot(hist, color=colors[i], label=labels[i])
    
    plt.title(title)
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_histograms_separate(histograms, title):
    # Define subplot layout
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    colors = ['r', 'g', 'b']
    labels = ['Hue', 'Saturation', 'Value']
    
    for i, (ax, hist) in enumerate(zip(axes, histograms)):
        ax.plot(hist, color=colors[i], label=labels[i])
        ax.set_title(f'{labels[i]} Channel')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make space for the suptitle
    plt.show()

if __name__ == '__main__':
    # Example usage
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']  # Update with your image paths
    avg_hist_hue, avg_hist_saturation, avg_hist_value = calculate_average_histogram(image_paths)
    plot_histograms_separate([avg_hist_hue, avg_hist_saturation, avg_hist_value], "Average HSV Histograms")