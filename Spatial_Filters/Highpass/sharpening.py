import numpy as np
import cv2
import sys
sys.path.append(r"g:\\projects\\IPCV-Labs")
from Spatial_Filters.correlation import filter
from Spatial_Filters.Lowpass.gaussian import gaussian


def sharpening(img: np.ndarray, ks:float, stdev:float, k:int):
    img = img.astype(np.float64)
    f1 = gaussian(ks, stdev)

    f = filter(img=img, kernel=f1)
    g_mask = img - f

    g = img + k * g_mask
    g = g.clip(0, 255).astype(dtype=np.uint8)

    return g

"""If k == 1: Unsharp masking, k > 1: Highboost filtering"""


# img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
# result_img = sharpening(img, 3, 0.5, 2)

# cv2.imshow("Original Image", img)
# cv2.imshow("Sharpened image", result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()