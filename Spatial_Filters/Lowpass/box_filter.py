import numpy as np
import cv2
import sys
sys.path.append(r"g:\\projects\\IPCV-Labs")
from Spatial_Filters.correlation import filter

def box(kh:int, kw:int):
    w = 1 / (kh * kw)
    return w * np.ones((kh, kw))

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
kernel = box(kh=3, kw=3)

result_img = filter(img, kernel)

cv2.imshow("Original Image", img)
cv2.imshow("Blurred", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()