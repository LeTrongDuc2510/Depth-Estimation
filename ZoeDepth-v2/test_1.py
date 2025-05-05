import numpy as np
import cv2
arr = np.ones((1200, 1200, 5), dtype=np.uint8)
cv2.resize(arr, (512, 512))