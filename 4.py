import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def enhance_contrast_color(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)  # Corrected 'applu' to 'apply'

    enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def segment_image_color(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(gray_image)
    return (gray_image > threshold).astype(np.uint8) * 255  

image = cv2.imread(r'aaaa.jpg')
if image is None:
    raise ValueError("Error: Could not load Image")

enhanced_image = enhance_contrast_color(image)
segmented_image = segment_image_color(enhanced_image)

# Display the images using Matplotlib
titles = ['Original Image', 'Enhanced Image', 'Segmented Image']
images = [image, enhanced_image, segmented_image]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.show()

