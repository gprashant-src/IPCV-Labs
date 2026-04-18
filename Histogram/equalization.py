import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# hist = np.zeros(256)
# for p in img.flatten():
#     hist[p] += 1

hist = np.bincount(img.flatten(), minlength=256)

"""For normalized histogram"""
hist = hist / (h * w)

T_r = np.round(np.cumsum(hist) * 255).astype(int)

H = defaultdict(list)
for r, t_r in enumerate(T_r):
    H[t_r].append(hist[r])

res = np.array([sum(H[i]) for i in range(256)])

# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(x=range(256), y=hist, ax=axes[0])
axes[0].set_title("Before Equalization")
axes[0].set_xticks([])


sns.barplot(x=range(256), y=res, ax=axes[1])
axes[1].set_title("After Equalization")
axes[1].set_xticks([])

plt.tight_layout()
plt.show()