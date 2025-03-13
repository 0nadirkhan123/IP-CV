import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('HOE.jpg', cv2.IMREAD_GRAYSCALE) 
if image is None:
    print("Image not found")
    exit()

# Step 1: Enhance the image contrast using Histogram Equalization
enhanced_image = cv2.equalizeHist(image)

# Alternatively, you can use CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

# Step 2: Segment the image using simple thresholding
# Otsu's Thresholding (automatic thresholding)
ret, thresholded_image = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 3: Visualize the results
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Original low-contrast image
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# Enhanced image using Histogram Equalization
axs[0, 1].imshow(enhanced_image, cmap='gray')
axs[0, 1].set_title('Histogram Equalized')
axs[0, 1].axis('off')

# Enhanced image using CLAHE
axs[0, 2].imshow(clahe_image, cmap='gray')
axs[0, 2].set_title('CLAHE Enhanced')
axs[0, 2].axis('off')

# Thresholded image (segmentation result)
axs[1, 0].imshow(thresholded_image, cmap='gray')
axs[1, 0].set_title("Thresholding (Otsu's)")
axs[1, 0].axis('off')

# Original vs enhanced side-by-side for comparison
axs[1, 1].imshow(np.hstack([image, enhanced_image]), cmap='gray')
axs[1, 1].set_title('Original vs Enhanced')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
