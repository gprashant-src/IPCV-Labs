import cv2
import numpy as np

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

def contrast_stretch(img, x1=None, x2=None, y1=0, y2=255, auto=False, out_min=0, out_max=255):
    if auto:
        # Auto contrasting
        m, M = np.min(img), np.max(img)
        if M == m:
            return np.full_like(img, out_min, np.uint8)
        res = (img - m) * (out_max - out_min) / (M - m) + out_min
    else:
        res = np.piecewise(img, [img < x1, (img >= x1) & (img <= x2), img > x2], 
                        [lambda x: (y1 / x1) * x, 
                            lambda x: y1 + (x - x1) * (y2 - y1) / (x2 - x1),
                            lambda x: y2 + (x - x2) * (255 - y2) / (255 - x2)])
    res = np.uint8(res)
    return res

# print(np.min(img), np.max(img))

res = contrast_stretch(img, auto=True)

cv2.imshow("Image", img)
cv2.imshow("Contrast Stretched", res)
cv2.waitKey(0)
cv2.destroyAllWindows()