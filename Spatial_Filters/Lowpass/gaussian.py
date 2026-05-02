import numpy as np
import cv2
import sys
sys.path.append(r"g:\\projects\\IPCV-Labs")
from Spatial_Filters.correlation import filter

def gaussian(ks:int, stdev:float):
    assert ks % 2 == 1, "Kernel size must be odd"

    kernel = np.zeros((ks, ks))
    t = ks // 2

    A = 2 * stdev ** 2
    K = 1 / (A * np.pi)
    for i in range(-t, t + 1):
        for j in range(-t, t + 1):
            T = (i ** 2 + j ** 2) / A
            kernel[i + t, j + t] = K * np.exp(-T)
    
    kernel = kernel / kernel.sum()
    
    return kernel


# img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
# kernel = gaussian(ks=5, stdev=2)

# result_img = filter(img, kernel)

# cv2.imshow("Original Image", img)
# cv2.imshow("Gaussian Blurred", result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def g(x:int, y:int, stdev:float):
#     K = 1 / (2 * np.pi * stdev ** 2)
#     T = (x ** 2 + y ** 2) / (2 * stdev ** 2)
#     return K * np.exp(-T)