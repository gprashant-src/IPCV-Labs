import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

def divide_image(img:np.ndarray, m:int=16, n:int=16):
    h, w = img.shape[:2]
    th = h // m
    tw = w // n

    tiles = []
    y_bins, x_bins = [], []

    for i in range(m):
        y_bins.append(i * th)
    y_bins.append(h)

    for i in range(n):
        x_bins.append(i * tw)
    x_bins.append(w)

    for i in range(m):
        arr = []
        for j in range(n):
            ys, ye = y_bins[i], y_bins[i + 1]
            xs, xe = x_bins[j], x_bins[j + 1]
            arr.append(img[ys:ye, xs:xe])
        tiles.append(arr)

    return tiles, np.array(y_bins, dtype=int), np.array(x_bins, dtype=int)

def hist_clip(hist:np.ndarray, th:int, tw:int, clip_factor:float=3.0):
    clip_limit = int(clip_factor * (th * tw) / 256)

    excess = np.zeros(256)
    for i in range(256):
        if hist[i] > clip_limit:
            excess[i] = hist[i] - clip_limit
            hist[i] = clip_limit

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

    return hist

def hist_equalize(tile:np.ndarray, hist:np.ndarray):
    hist_1 = hist / (tile.size)
    T_r = np.round(np.cumsum(hist_1) * 255).astype(np.uint8)
    return T_r[tile], T_r

def pipeline1(tile:np.ndarray):
    test = tile
    th, tw = test.shape

    hist = np.bincount(test.flatten(), minlength=256)
    x1 = hist_clip(hist, th, tw)

    return hist_equalize(test, x1)


def stitch_image(tiles:List[np.ndarray]):
    process = []
    MAP = []
    m, n = len(tiles), len(tiles[0])
    for i in range(m):
        row = []
        M = []
        for j in range(n):
            IMG, map = pipeline1(tiles[i][j])
            row.append(IMG)
            M.append(map)

        process.append(row)
        MAP.append(M)
    
    return np.vstack([np.hstack(row) for row in process]), np.array(MAP, dtype=np.uint8)


def interpolate(img:np.ndarray, mappings:List[np.ndarray], y_bins:List[int], x_bins:List[int]):
    h, w = img.shape
    m, n = mappings.shape[:2]

    yc, xc = (y_bins[:-1] + y_bins[1:]) / 2, (x_bins[:-1] + x_bins[1:]) / 2

    y_idx, x_idx = np.indices((h, w))

    y0 = np.clip(np.searchsorted(yc, y_idx) - 1, 0, m - 2)
    x0 = np.clip(np.searchsorted(xc, x_idx) - 1, 0, n - 2)

    y1, x1 = y0 + 1, x0 + 1

    dy = np.clip((y_idx - yc[y0]) / (yc[y1] - yc[y0]), 0, 1)
    dx = np.clip((x_idx - xc[x0]) / (xc[x1] - xc[x0]), 0, 1)

    V = img
    q11, q12, q21, q22 = mappings[y0, x0, V], mappings[y1, x0, V], mappings[y0, x1, V], mappings[y1, x1, V]

    final_img = q11 * (1 - dx) * (1 - dy) + \
                q12 * (1 - dx) * dy + \
                q21 * dx * (1 - dy) + \
                q22 * dx * dy
    
    return final_img.astype(dtype=np.uint8)
    

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

arr, y_bins, x_bins = divide_image(img)
result_img, MAP = stitch_image(arr)
final_img = interpolate(img, MAP, y_bins, x_bins)

cv2.imshow("Original Image", img)
cv2.imshow("Process p1", result_img)
cv2.imshow("After interpolate", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




