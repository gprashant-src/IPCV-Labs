import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

hist = np.bincount(img.flatten(), minlength=256)

"""For normalized histogram"""
hist_1 = hist / (h * w)
T_r = np.round(np.cumsum(hist_1) * 255).astype(np.uint8)

# H = defaultdict(int)
# for r, t_r in enumerate(T_r):
#     H[t_r] += hist[r]
# res = np.array([H[i] for i in range(256)])

res = np.zeros(256, dtype=np.uint64)
np.add.at(res, T_r, hist)


"""Original and Final image display"""

cv2.imshow("Original", img)
cv2.imshow("After histogram eq.", T_r[img])
cv2.waitKey(0)
cv2.destroyAllWindows()


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(x=range(256), y=hist, ax=axes[0], color="blue")
axes[0].set_title("Before Equalization")
axes[0].set_xticks([])

sns.barplot(x=range(256), y=res, ax=axes[1], color="orange")
axes[1].set_title("After Equalization")
axes[1].set_xticks([])

plt.tight_layout()
plt.show()