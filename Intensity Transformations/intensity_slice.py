import cv2
import numpy as np

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

def intensity_slice(img, p1, p2, suppress=True, high=255):
    if not suppress:
        res = img.copy()
        res[(img >= p1) & (img <= p2)] = high
        return res
    else:
        res = np.where((img >= p1) & (img <= p2), high, 0).astype(np.uint8)
        return res


res = intensity_slice(img, 40, 100, True)

cv2.imshow("Image", img)
cv2.imshow("Intensity_Slice", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
