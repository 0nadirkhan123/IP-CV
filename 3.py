import cv2
import numpy as np

# Function to apply rotation, scaling, and translation to an image
def transform_image(image_path, angle, scale, tx, ty):
    # Load the image
    image = cv2.imread('sunflower.jpg')
    
    if image is None:
        print("Error: Image not found!")
        return None
    
    # Get the image dimensions
    height, width = image.shape[:2]
    
    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, scale)
    
    # Translation matrix (shifting the image)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply rotation and scaling using cv2.warpAffine
    rotated_scaled_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    # Apply translation to the rotated and scaled image
    transformed_image = cv2.warpAffine(rotated_scaled_image, translation_matrix, (width, height))
    
    return transformed_image

# Path to the input image
image_path = 'D:/nk053 ipcv/sunflower.jpg'  # Replace with your image path

# Parameters
angle = 0  # Rotate by 45 degrees
scale = 1 # Scale by 1.2 times
tx, ty = 0, 0  # Translate by 100px horizontally and 50px vertically

# Perform the transformation
result = transform_image(image_path, angle, scale, tx, ty)

# If transformation was successful, display the result
if result is not None:
    cv2.imshow("Transformed Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Optionally save the result
    cv2.imwrite('transformed_image.jpg', result)






