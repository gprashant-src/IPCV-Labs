import cv2
import numpy as np

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

def contrast_stretch(img, x1, x2, y1=0, y2=255):
    res = np.piecewise(img, [img < x1, (img >= x1) & (img <= x2), img > x2], 
                       [lambda x: (y1 / x1) * x, 
                        lambda x: y1 + (x - x1) * (y2 - y1) / (x2 - x1),
                        lambda x: y2 + (x - x2) * (255 - y2) / (255 - x2)])

    res = np.uint8(res)
    return res

# print(np.min(img), np.max(img))

res = contrast_stretch(img, 100, 55)

cv2.imshow("Image", img)
cv2.imshow("Contrast Stretched", res)
cv2.waitKey(0)
cv2.destroyAllWindows()