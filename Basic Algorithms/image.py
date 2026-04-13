import cv2

# Load the image
img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

# Image properties
h, w = img.shape

print("Data type of each pixel value:", img.dtype)

# Access each pixel
x, y = 100, 100
print("Image at", (x, y), ":", img[100, 100])

# Modify pixel
# img[x, y] = 0

# Loop through image
h, w = img.shape
for i in range(h):
    for j in range(w):
        pixel = img[i, j]
        # print(pixel)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()