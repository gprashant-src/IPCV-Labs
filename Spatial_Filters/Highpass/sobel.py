import numpy as np
from math import comb
import cv2
import sys
sys.path.append(r"g:\\projects\\IPCV-Labs")
from Spatial_Filters.correlation import filter


def sobel(ks:int):
    assert ks % 2 == 1, "Kernel size must be odd"

    t = ks // 2
    x = ks - 1
    A = np.array([comb(x, i) for i in range(x + 1)])
    B = np.zeros(ks)

    B[:t] = -1
    B[t + 1:] = 1

    kernel = np.outer(A, B).T       # Vertical gradient
    # if horizontal, remove T
    return kernel


img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
kernel = sobel(ks=3)

result_img = filter(img, kernel)

cv2.imshow("Original Image", img)
cv2.imshow("Prewitt", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()