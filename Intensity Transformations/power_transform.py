import cv2
import numpy as np

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

gamma = 1.5

temp = img.astype(np.float32)
temp = temp / 255.0

res = np.uint8(np.power(temp, gamma) * 255)

cv2.imshow("Image", img)
cv2.imshow("Power Transform Image", res)
cv2.waitKey(0)
cv2.destroyAllWindows()