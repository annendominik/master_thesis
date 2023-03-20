# Python packages
import numpy as np
import scipy
from skimage.color import rgb2gray, rgba2rgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.io
from skimage.data import camera


if __name__ == "__main__":
    # Set resolution of output images
    resolution = 500

    # Plot original image for comparison
    image = camera()
    # image = plt.imread('shearogramm.png')
    # image = rgba2rgb(image)
    # image = rgb2gray(image)
    plt.figure(frameon=False)
    plt.imshow(image, cm.gray)
    plt.axis('off')
    plt.savefig('fftoriginal',
                dpi=resolution, bbox_inches='tight', pad_inches=0)

    # Transform into greyscale
    # image_gray = rgb2gray(image)

    # Plot grayscale image
    # plt.figure(frameon=False)
    # plt.imshow(image_gray, cmap=cm.gray)
    # plt.axis('off')
    # plt.savefig('fftgrayscale',
    #             dpi=resolution, bbox_inches='tight', pad_inches=0)

    # Compute FFT of image
    image_fft = scipy.fft.fft2(image)

    # Sort frequencies by magnitude
    fft_magsort = np.sort(np.abs(image_fft.reshape(-1)))

    # Zero out all small coefficients and inverse transform
    for keep in (0.5, 0.25, 0.1, 0.05, 0.01, 0.002):
        # Set thresholds
        threshold = fft_magsort[int(np.floor((1-keep)*len(fft_magsort)))]
        # Keep only frequencies above threshold
        ind = np.abs(image_fft)>threshold
        fft_low = image_fft * ind

        # Compute inverse FFT
        im_low = scipy.fft.ifft2(fft_low).real

        # Plot compressed images
        plt.figure(frameon=False)
        plt.imshow(im_low, cmap=cm.gray)
        plt.axis('off')

        # Define file names
        keep_per = str(keep*100) + 'percent'
        if keep_per == '0.2percent':
            keep_per = '0dot2percent'
        else:
            keep_per = str(int(keep*100)) + 'percent'
        plt.savefig('fft' + f'{keep_per}',
                    dpi=resolution, bbox_inches='tight', pad_inches=0)

    plt.show()
