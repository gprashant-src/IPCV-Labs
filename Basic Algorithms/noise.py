import cv2
import numpy as np

def saturate(img, a=0, b=255):
    res = img.copy()

    for i in range(h):
        for j in range(w):
            val = res[i][j]
            if val > b:
                res[i][j] = b
            elif val < a:
                res[i][j] = a
            else:
                res[i][j] = val
    
    
    return res

# add noise in image (Here we are using normal noise function)
def noise(img, mean, std):
    N = np.random.normal(loc=mean, scale=std, size=(h, w))
    res = img.copy()

    N_temp = res + N
    return saturate(N_temp).astype(np.uint8)


img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

res_img = noise(img, -400, 200)
cv2.imshow("Image", img)
cv2.imshow("Noise", res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()