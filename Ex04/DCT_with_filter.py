import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
import pathlib


class DCT:
    def __init__(self, path, filename):
        # Read img
        self.path = path
        self.img = cv2.imread(path+filename, cv2.IMREAD_GRAYSCALE)

    def mirror_img(self):
        mirror_right = self.img[:, ::-1]
        img = np.concatenate((self.img, mirror_right), axis=1)
        mirror_down = img[::-1, :]
        self.ext_img = np.concatenate((img, mirror_down), axis=0)

        cv2.imwrite(self.path+"Extended_img.png", self.ext_img)

    def plot_extended(self):

        plt.set_cmap("gray")

        plt.subplot(121)
        plt.imshow(self.img)
        plt.title("Original image")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(self.ext_img)
        plt.title("Extended image")
        plt.axis("off")
        plt.show()

    def apply_filter(self, ft_img, perc_coeff, filter_type):

        mask = np.ones(ft_img.shape)
        exclude_coeff_h, exclude_coeff_w  = int((perc_coeff/100)*ft_img.shape[0]), int((perc_coeff/100)*ft_img.shape[1])
        # center_row, center_col = int(ft_img.shape[0]/2), int(ft_img.shape[1]/2)
        # y, x = np.arange(ft_img.shape[0]), np.arange(ft_img.shape[1])
        # X, Y = np.meshgrid(x, y)

        if filter_type == "HPF":
            # filter out evreything inside the circle i.e all indices inside the
            # circle in spatial domain will have zero pixel value
            # filter_out_cond = (X-center_col)**2 + \
            #     (Y-center_row)**2 <= radius**2
            mask[0:exclude_coeff_h, 0:exclude_coeff_w] = 0

        if filter_type == "LPF":
            # filter out evreything outside the circle i.e all indices outside the
            # circle in spatial domain will have zero pixel value
            # filter_out_cond = (X-center_col)**2 + \
            #     (Y-center_row)**2 >= radius**2
            mask[-exclude_coeff_h:, -exclude_coeff_w:] = 0
            print(np.count_nonzero(mask))
        # mask[filter_out_cond] = 0
        return ft_img*mask

    def filter_dct_8x8(self):
        """The function assumes that the size of input image is multiple of 8 as no padding is applied."""
        h, w = self.img.shape
        
        # create 8x8 blocks
        total_blocks = h//8*w//8

        block_h_idx = list(np.arange(h)[::8])
        block_w_idx = list(np.arange(w)[::8])

        dct_8x8 = np.zeros((h, w), dtype=np.float64)

        # loop over blocks and compute dct for each block
        for start_h_idx in block_h_idx:
            for start_w_idx in block_w_idx:
                
                block = self.img[start_h_idx:start_h_idx+8, start_w_idx:start_w_idx +8]
                block_dct = scipy.fft.dct(block, type=2, norm="ortho")
                dct_8x8[start_h_idx:start_h_idx+8, start_w_idx:start_w_idx +8] = block_dct
        
        # compute magnitude 
        dct_mag = np.absolute(dct_8x8)

        # plot
        plt.set_cmap("gray")

        plt.subplot(121)
        plt.imshow(self.img)
        plt.title("Original image")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(np.log(dct_mag+1))
        plt.title("FT of Original image")
        plt.axis("off")
        plt.show()

    def filter_dct_grayscale(self):

        # apply 2D Fourier Transform
        dct_img = scipy.fft.dct(self.img, type=2, norm="ortho")

        # No need to shift the center in case of DCT
        # shifted_dct = scipy.fft.fftshift(dct_img)
        dct_mag = np.absolute(dct_img)

        # Take size and type of filter from user
        perc_coeff = int(input("Enter the percentage of the coefficients to be excluded: "))
        filter = input("Enter Filter: ")

        # Apply filter
        filtered_dct = self.apply_filter(dct_img.copy(), perc_coeff, filter)
        filtered_dct_mag = np.absolute(filtered_dct)

        # Apply inverse Fourier Transform
        filtered_img = scipy.fft.idct(filtered_dct)

        # plot
        plt.set_cmap("gray")

        plt.subplot(221)
        plt.imshow(self.img)
        plt.title("Original image")
        plt.axis("off")

        plt.subplot(222)
        plt.imshow(np.log(dct_mag+1))
        plt.title("FT of Original image")
        plt.axis("off")

        plt.subplot(223)
        plt.imshow(abs(filtered_img))
        plt.title("Filtered image")
        plt.axis("off")

        plt.subplot(224)
        plt.imshow(np.log(filtered_dct_mag+1))
        plt.title("FT of Filtered image")
        plt.axis("off")
        plt.show()


# ==================================================
dct = DCT("./Ex04/", "smoke_img.jpg")
# dct.filter_dct_grayscale()
dct.filter_dct_8x8()
# dct.mirror_img()
# dct.plot_extended()
# ==================================================
