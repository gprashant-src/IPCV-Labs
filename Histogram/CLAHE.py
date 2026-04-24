import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def divide_image(img:np.ndarray, m:int=8, n:int=8):
    h, w = img.shape[:2]
    th = h // m
    tw = w // n
    # print("No of tiles = ", (h * w) // (m * n))

    tiles = []
    for i in range(m):
        arr = []
        for j in range(n):
            ys, xs = i * th, j * tw
            ye, xe = (i + 1) * th if i != m - 1 else h, (j + 1) * tw if j != n - 1 else w
            arr.append(img[ys:ye, xs:xe])
        tiles.append(arr)

    return tiles

def hist_clip(hist:np.ndarray, th:int, tw:int, clip_factor:float=3.0):
    clip_limit = int(clip_factor * (th * tw) / 256)
    # print(clip_limit)

    excess = np.zeros(256)
    for i in range(256):
        if hist[i] > clip_limit:
            excess[i] = hist[i] - clip_limit
            hist[i] = clip_limit
    # l = 1
    while excess.sum() != 0:
        # print(l)
        dis, rem = np.divmod(int(excess.sum()), 256)
        
        hist += dis
        hist[:rem] += 1

        excess = np.zeros(256)
        for i in range(256):
            if hist[i] > clip_limit:
                excess[i] = hist[i] - clip_limit
                hist[i] = clip_limit
        # l += 1

    return hist

        


# def hist_equalize(img):
#     hist = np.bincount(img.flatten(), minlength=256)

#     """For normalized histogram"""
#     hist_1 = hist / (h * w)
#     T_r = np.round(np.cumsum(hist_1) * 255).astype(np.uint8)

#     res = np.zeros(256, dtype=np.uint64)
#     np.add.at(res, T_r, hist)
#     return hist, res, T_r[img]


    

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

arr = divide_image(img)

test = arr[0][0]
th, tw = test.shape
hist = np.bincount(test.flatten(), minlength=256)

print(hist)
print(hist_clip(hist, th, tw))

# A = 0
# for i in range(8):
#     for j in range(8):
#         h, w = arr[i][j].shape
#         A += h * w

