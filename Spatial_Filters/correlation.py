import numpy as np

def filter(img:np.ndarray, kernel:np.ndarray):
    h, w = img.shape
    kh, kw = kernel.shape

    assert kh % 2 == 1 and kw % 2 == 1, "Kernel invalid"

    ph, pw = kh // 2, kw // 2

    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    output = np.zeros_like(img, dtype=np.float64)

    for i in range(h):
        for j in range(w):
            reg = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(reg * kernel)
    
    output = np.clip(output, 0, 255)
    return output.astype(dtype=np.uint8)

img = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])
# if convolution
# kernel = np.flip(kernel)
# print(filter(img, kernel))