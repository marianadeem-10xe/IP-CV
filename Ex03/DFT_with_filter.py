import numpy as np
import scipy 
import matplotlib.pyplot as plt
import cv2

def mirror_img(path):
    org_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mirror_right = org_img[:, ::-1]
    img = np.concatenate((org_img,mirror_right), axis=1)
    mirror_down = img[::-1, :]
    img = np.concatenate((img, mirror_down), axis=0)
    
    cv2.imwrite("./Ex03/Extended_img.png", img)
    # plt.set_cmap("gray")
    
    # plt.subplot(121)
    # plt.imshow(org_img)
    # plt.title("Original image")
    # plt.axis("off")

    # plt.subplot(122)
    # plt.imshow(img)
    # plt.title("Extended image")
    # plt.axis("off")
    # plt.show()


def apply_filter(ft_img, radius, filter_type):
    
    mask = np.ones(ft_img.shape)
    center_row, center_col = int(ft_img.shape[0]/2) , int(ft_img.shape[1]/2)
    y, x = np.arange(ft_img.shape[0]), np.arange(ft_img.shape[1]) 
    X, Y = np.meshgrid(x, y)
        
    if filter_type=="HPF":
        # filter out evreything inside the circle i.e all indices inside the 
        # circle in spatial domain will have zero pixel value
        filter_out_cond = (X-center_col)**2 + (Y-center_row)**2 <= radius**2

    elif filter_type=="LPF":
        # filter out evreything outside the circle i.e all indices outside the 
        # circle in spatial domain will have zero pixel value
        filter_out_cond = (X-center_col)**2 + (Y-center_row)**2 >= radius**2
    else: 
        filter_out_cond = (X-center_col)**2 + (Y-center_row)**2 < 0

    mask[filter_out_cond] = 0
    return ft_img*mask

def filter_dft_grayscale(path, filter=True):
    # Read img
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # apply 2D Fourier Transform
    dft_img = scipy.fft.fft2(img)

    # shift center for better visualization
    shifted_dft = scipy.fft.fftshift(dft_img)
    shifted_dft_mag = np.absolute(shifted_dft)
    # print(np.max(shifted_dft_mag))
    # print(np.min(shifted_dft_mag))
    
    # Take size and type of filter from user  
    radius = int(input("Enter Radius: "))
    filter = input("Enter Filter: ")
    
    # Apply filter
    filtered_dft = apply_filter(shifted_dft.copy(), radius, filter)
    filtered_dft_mag = np.absolute(filtered_dft)

    # shift back the center       
    filtered_bshifted = scipy.fft.ifftshift(filtered_dft)

    # Apply inverse Fourier Transform
    filtered_img = scipy.fft.ifft2(filtered_bshifted)
    
    # plot 
    plt.set_cmap("gray")
    
    plt.subplot(221)
    plt.imshow(img)
    plt.title("Original image")
    plt.axis("off")
    
    plt.subplot(222)
    plt.imshow(np.log(shifted_dft_mag+1))
    plt.title("FT of Original image")
    plt.axis("off")
    
    plt.subplot(223)
    plt.imshow(abs(filtered_img))
    plt.title("Filtered image")
    plt.axis("off")

    plt.subplot(224)
    plt.imshow(np.log(filtered_dft_mag+1))
    plt.title("FT of Filtered image")
    plt.axis("off")
    plt.show()

filter_dft_grayscale("./Ex03/smoke_img.jpg")# bird_150x200.jpeg, Extended_img.png , smoke_zoomed_in.jpg
# mirror_img("./Ex03/smoke_zoomed_in.jpg")