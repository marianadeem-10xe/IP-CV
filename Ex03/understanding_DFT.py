import numpy as np
import cv2
import matplotlib.pyplot as plt

# Used in Assignment P3Q1 (see Assignment Notes google doc)

def gaussian_2D(peak, std_x,std_y, center = (0,0), theta=0):
    """https://en.wikipedia.org/wiki/Gaussian_function"""
    x = np.arange(-500, 501, 1)
    X, Y = np.meshgrid(x, x)
    # gaussian = peak*np.exp(-((X-center[0])**2/(2*std_x**2))+((Y-center[1])**2/(2*std_y**2)))
    a = np.cos(theta)**2 / (2 * std_x**2) + np.sin(theta)**2 / (2 * std_y**2)
    b = np.sin(2 * theta) / (4 * std_x**2) - np.sin(2 * theta) / (4 * std_y**2)
    c = np.sin(theta)**2 / (2 * std_x**2) + np.cos(theta)**2 / (2 * std_y**2)

    gaussian = peak * np.exp(-(a * (X - center[0])**2 + 2 * b * (X - center[0])*(Y - center[1]) + c * (Y - center[1])**2))
    return gaussian

def sinusoidal_grating(wavelength, angle, amp, phase):
    x = np.arange(-500, 501, 1)
    X, Y = np.meshgrid(x, x)        # Cartesian grid
    # y = A sin(2pi(xcos(theta)+ysin(theta)))
    grating = amp*(np.sin((2 * np.pi * (X*np.cos(angle) + Y*np.sin(angle)) / wavelength)+phase))
    return grating    

def cosine_grating(wavelength, angle):
    x = np.arange(-500, 501, 1)
    X, Y = np.meshgrid(x, x)        # Cartesian grid
    grating = np.cos(2 * np.pi * (X*np.cos(angle) + Y*np.sin(angle)) / wavelength)
    return grating

def plot_DFT(grating):

    plt.set_cmap("gray")
    plt.subplot(121)
    plt.imshow(grating)

    ft = np.fft.ifftshift(grating)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    
    plt.subplot(122)
    plt.imshow(abs(ft))
    plt.xlim([480, 520])
    plt.ylim([520, 480])  # Note, order is reversed for y
    plt.show() 

def plot_dft_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.set_cmap("gray")

    ft = np.fft.fftshift(img)
    ft = np.fft.fft2(ft)
    ft = np.fft.ifftshift(ft)

    plt.subplot(121)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(np.log(abs(ft)+1))
    plt.axis("off")
    plt.show()


# increasing the frequency with have more sin waves in the same region and more black stripes are seen 
# theta is the orientation, results in rotation of the stripes
# 
wavelength = 200
theta  = 0
amplitude =1
phase = np.pi

gaussian = gaussian_2D(peak = 1,std_x=50,std_y=50,theta =0)
sin_grating = sinusoidal_grating(wavelength,theta, amplitude, phase)
cos_grating = cosine_grating(wavelength,theta)
# plot_DFT(gaussian)
# plot_DFT(sin_grating)
# plot_DFT(cos_grating)

plot_dft_grayscale("./Ex03/bird_150x200.jpeg")