import cv2
import numpy as np

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

def bit_plane_slicing(img, k):
    res = (img >> k) & 1
    return np.uint8(res * 255)

cv2.imshow("Original", img)
for i in range(8):
    res = bit_plane_slicing(img, i)
    cv2.imshow(f"Image at {i}th bit plane", res)

cv2.waitKey(0)
cv2.destroyAllWindows()