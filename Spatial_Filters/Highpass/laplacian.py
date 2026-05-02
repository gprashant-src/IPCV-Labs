import numpy as np
from math import comb
import cv2
import sys
sys.path.append(r"g:\\projects\\IPCV-Labs")
from Spatial_Filters.correlation import filter


def laplacian(diag:bool=False):
    # assert ks % 2 == 1, "Kernel size must be odd"
    if not bool:
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])




img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
kernel = laplacian(True)

result_img = filter(img, kernel)

cv2.imshow("Original Image", img)
cv2.imshow("Prewitt", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()