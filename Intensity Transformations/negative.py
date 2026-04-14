import cv2

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

res = img.copy()

for i in range(h):
    for j in range(w):
        res[i][j] = 255 - res[i][j]

cv2.imshow("Image", img)
cv2.imshow("Negative", res)
cv2.waitKey(0)
cv2.destroyAllWindows()