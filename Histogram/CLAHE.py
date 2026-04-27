import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

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

def hist_equalize(tile:np.ndarray, hist:np.ndarray):
    hist_1 = hist / (tile.size)
    T_r = np.round(np.cumsum(hist_1) * 255).astype(np.uint8)
    return T_r, T_r[tile]

def pipeline1(tile:np.ndarray):
    test = tile
    th, tw = test.shape

    hist = np.bincount(test.flatten(), minlength=256)
    x1 = hist_clip(hist, th, tw)

    _, final_tile = hist_equalize(test, x1)
    return final_tile

def stitch_image(tiles:List[np.ndarray]):
    process = []
    m, n = len(tiles), len(tiles[0])
    for i in range(m):
        row = []
        for j in range(n):
            row.append(pipeline1(tiles[i][j]))

        process.append(row)
    
    return np.vstack([np.hstack(row) for row in process])

    


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
result_img = stitch_image(arr)

cv2.imshow("Original Image", img)
cv2.imshow("Transformed", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# A = 0
# for i in range(8):
#     for j in range(8):
#         h, w = arr[i][j].shape
#         A += h * w

