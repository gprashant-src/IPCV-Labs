import cv2
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

#img1 = Source image ; img2 = Target image
def hist_matching(img1, img2):
    hist_1 = np.bincount(img1.flatten(), minlength=256)
    hist_2 = np.bincount(img2.flatten(), minlength=256)

    r1, r2 = hist_1 / img1.size, hist_2 / img2.size

    cdf1 = np.round(255 * np.cumsum(r1))
    cdf2 = np.round(255 * np.cumsum(r2))


    # Binary search to find the index i of cdf2 that is closest of any e in cdf1
    H = np.searchsorted(cdf1, cdf2, side="left")    
    H = np.clip(H, 0, 255).astype(np.uint8)

    final_img = H[img1]

    return hist_1, hist_2, np.bincount(final_img.flatten(), minlength=256), final_img




src = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
target = cv2.imread("data/test2.jpg", cv2.IMREAD_GRAYSCALE)

# h, w = img.shape

"""Original , Target and Final image display"""
src_h, target_h, res_h, final_img = hist_matching(src, target)

cv2.imshow("Original", src)
cv2.imshow("Target image distribution.", target)
cv2.imshow("After histogram matching.", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


fig, axes = plt.subplots(1, 3, figsize=(18, 4))

x = np.arange(256)
axes[0].bar(x, src_h, color="blue")
axes[0].set_title("Source Histogram")
axes[0].set_xlim([0, 255])

axes[1].bar(x, target_h, color="green")
axes[1].set_title("Target Histogram")
axes[1].set_xlim([0, 255])

axes[2].bar(x, res_h, color="orange")
axes[2].set_title("Matched Histogram")
axes[2].set_xlim([0, 255])

plt.tight_layout()
plt.show()

