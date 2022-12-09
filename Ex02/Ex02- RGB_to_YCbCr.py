import numpy as np
import cv2

def RGB_to_YCbCr(img):
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    Y  = 2.990*R + 0.5870*G + 0.1140*B
    Cb = 127.5 - 0.1687*R - 0.3313*G + 0.5*B
    Cr = 127.5 + 0.5*R - 0.4187*G -0.0813*B
    return np.stack([Y, Cr, Cb], axis=2)         
             
def YCbCr_to_RGB(img):
    Y, Cr, Cb = img[:,:,0], img[:,:,1], img[:,:,2]
    R = Y + 1.4020*(Cr-127.5)
    G = Y - 0.3441*(Cb-127.5) - 0.7141*(Cr-127.5)
    B = Y + 1.7720*(Cb-127.5) 
    return np.stack([B, G, R], axis=2)

###################################################

path = "./rgb.png"
img = cv2.imread(path)
print(img.shape)
# cv2.imwrite("./cvimg.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    

ycbcr = RGB_to_YCbCr(img)
rgb = YCbCr_to_RGB(ycbcr)

print(ycbcr.shape)
print(rgb.shape)

cv2.imwrite("./yuv_img.jpg", ycbcr)
cv2.imwrite("./rgb_img.jpg", rgb)
print("img saved")