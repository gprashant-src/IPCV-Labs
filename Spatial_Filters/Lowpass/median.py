import numpy as np
import cv2

def median(img:np.ndarray, ks:int):
    m, n = img.shape[:2]
    temp = np.pad(img, ((ks, ks), (ks, ks)), mode="reflect")
    
    output = np.zeros_like(img)
    
    for i in range(m):
        for j in range(n):
            output[i, j] = np.median(temp[i:i + ks, j:j + ks])
    
    return output
    

img = cv2.imread("data/test1.jpg", cv2.IMREAD_GRAYSCALE)

result_img = median(img, ks=7)

cv2.imshow("Original Image", img)
cv2.imshow("Median image", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()