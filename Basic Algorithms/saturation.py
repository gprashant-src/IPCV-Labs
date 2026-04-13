import cv2
import numpy as np

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

h, w = img.shape

def saturate(a, b):
    res = img.copy()

    for i in range(h):
        for j in range(w):
            val = res[i][j]
            if val > b:
                res[i][j] = b
            elif val < a:
                res[i][j] = a
    
    return res

res_img = saturate(50, 100)
cv2.imshow("Image", img)
cv2.imshow("Saturation", res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()