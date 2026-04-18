import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# hist = np.zeros(256)
# for p in img.flatten():
#     hist[p] += 1

hist = np.bincount(img.flatten(), minlength=256)

"""For normalized histogram"""
# hist = hist / (h * w)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

sns.barplot(x=range(256), y=hist)
plt.xticks([])
plt.show()

