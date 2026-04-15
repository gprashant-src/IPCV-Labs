import cv2
import numpy as np

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

temp = img.astype(np.float32)
c = 255 / np.log(1 + np.max(temp))

res = np.uint8(c * np.log(1 + temp))

cv2.imshow("Image", img)
cv2.imshow("NLog Transform Image", res)
cv2.waitKey(0)
cv2.destroyAllWindows()