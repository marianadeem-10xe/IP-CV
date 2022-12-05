import numpy as np
import cv2
import matplotlib.pyplot as plt
###################################################################################

def analyse_grey_values(img):
    """Compute stats of the image"""
    min = np.min(img)
    max = np.max(img)
    mean = np.mean(img)
    std = np.std(img)
    return [min,max,mean,std]

def quantisation(q, img):
    
    """ Quantize and image
        q: number of bits used to represent a value
        in the output image """
    
    d = 2**(8-q) 
    img = np.float32(img)
    imheight, imwidth = img.shape[0], img.shape[1]
    for y in range(imheight):
        for x in range(imwidth):
            img[y,x] = (np.floor(img[y,x]/d) + 0.5)*d 
    return img.astype("uint8")

def quantisation_with_noise(q, img):
    
    """ Quantize and image
        q: number of bits used to represent a value
        in the output image """
    
    d = 2**(8-q)
    img = np.float32(img)
    imheight, imwidth = img.shape[0], img.shape[1]
    for x in range(imwidth):
        for y in range(imheight):
            noise = np.round(np.random.uniform(-0.5,0.5),2)
            img[y,x] = (np.floor((img[y,x]/d)+noise) + 0.5)*d   
    return img.astype("uint8")

def convert(img, q):
  img1 = np.uint8(np.round((np.float32(img.copy())/255)*((2**q)-1)))

  return img1 

###################################################################################
print("\n")
print("QUANTISATION\n\n")
print("**************************************************\n\n")
print("    Copyright 2021 by Joachim Weickert            \n")
print("    Dept. of Mathematics and Computer Science     \n")
print("    Saarland University, Saarbruecken, Germany    \n\n")
print("    All rights reserved. Unauthorised usage,      \n")
print("    copying, hiring, and selling prohibited.      \n\n")
print("    Send bug reports to                           \n")
print("    weickert@mia.uni-saarland.de                  \n\n")
print("**************************************************\n\n")

# ---- What is a PGM file?? ----
# link: https://www.adobe.com/creativecloud/file-types/image/raster/pgm-file.html
"""PGM (Portable Gray Map) files store grayscale 2D images. 
Each pixel within the image contains only one or two bytes of information
(8 or 16 bits). While that might not sound like a lot of information, PGM 
files can hold tens of thousands of shades — ranging from pure black to white,
and every shade of gray in between."""

#  ---- Step1: read input image (pgm format P5) ----
path = "boat.pgm"
pgm_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
img_h, img_w = pgm_img.shape[0], pgm_img.shape[1]
# print(pgm_img.shape)

print("number of bits to represent quantised value:  ")
q = int(input())

print("goal of the program:\n") 
print(" (1) quantisation without noise\n")
print(" (2) quantisation with uniform noise\n")
print("your choice:                                  ")
flag = int(input())          

lst = analyse_grey_values(pgm_img)
print("original image:\n")
print("minimum:       {} \n".format(lst[0]))
print("maximum:       {} \n".format(lst[1]))
print("mean:          {} \n".format(round(lst[2],2)))
print("standard dev.: {} \n\n".format(round(lst[3],2)))

if flag == 1:
    # perform quantisation without noise
    out = quantisation (q, pgm_img.copy())
elif flag == 2:
    # perform quantisation with uniformly distributed noise
    out = quantisation_with_noise (q, pgm_img.copy())
else:
    print("option {} not available! \n\n\n".format(flag))
    exit()

# /* ---- analyse filtered image ---- */

img1  = convert(pgm_img.copy(), 8-q)
# lst1 = analyse_grey_values(img1)
lst = analyse_grey_values(out)

# print("image1:\n")
# print("minimum:       {} \n".format(lst1[0]))
# print("maximum:       {} \n".format(lst1[1]))
# print("mean:          {} \n".format(round(lst1[2],2)))
# print("standard dev.: {} \n\n".format(round(lst1[3],2)))


print("quantised image:\n")
print("minimum:       {} \n".format(lst[0]))
print("maximum:       {} \n".format(lst[1]))
print("mean:          {} \n".format(round(lst[2],2)))
print("standard dev.: {} \n\n".format(round(lst[3],2)))


# ---- write output image (pgm format P5) ---- 
filename = "quantised_image_flag{}_q{}.pgm".format(flag, q)
cv2.imwrite(filename, out, (cv2.IMWRITE_PXM_BINARY, 0))
# cv2.imwrite("img1.pgm", img1, (cv2.IMWRITE_PXM_BINARY, 0))
##########################################################################
