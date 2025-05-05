import cv2

# Load the image
image_path = "rgb.jpg"  # Change this to your image path
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
output_path = "rgb_gray.jpg"  # Change this if needed
cv2.imwrite(output_path, gray_image)

print(f"Grayscale image saved as {output_path}")

from PIL import Image
import numpy as np

image = Image.open(output_path)
image = np.array(image)
print(image.shape)