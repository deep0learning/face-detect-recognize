"""
====================
Denoising a picture
====================

In this example, we denoise a noisy version of a picture using the total
variation, bilateral, and wavelet denoising filters.

Total variation and bilateral algorithms typically produce "posterized" images
with flat domains separated by sharp edges. It is possible to change the degree
of posterization by controlling the tradeoff between denoising and faithfulness
to the original image.

Total variation filter
----------------------

The result of this filter is an image that has a minimal total variation norm,
while being as close to the initial image as possible. The total variation is
the L1 norm of the gradient of the image.

Bilateral filter
----------------

A bilateral filter is an edge-preserving and noise reducing filter. It averages
pixels based on their spatial closeness and radiometric similarity.

Wavelet denoising filter
------------------------

A wavelet denoising filter relies on the wavelet representation of the image.
The noise is represented by small values in the wavelet domain which are set to
0.

In color images, wavelet denoising is typically done in the `YCbCr color
space`_ as denoising in separate color channels may lead to more apparent
noise.

.. _`YCbCr color space`: https://en.wikipedia.org/wiki/YCbCr

"""
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from scipy import fftpack
import argparse
import cv2
import numpy as np
import mxnet as mx

def parm():
    parser = argparse.ArgumentParser(description='img denoise')
    parser.add_argument('--img-dir',dest='img_dir',type=str,default=None,\
                        help="image saved dir")
    return parser.parse_args()

def denoise_img(file_name):
    if file_name is not None:
        image = cv2.imread(file_name)
        original = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        noisy = original
    else:
        original = img_as_float(data.chelsea()[100:250, 50:300])
        sigma = 0.155
        noisy = random_noise(original, var=sigma**2)
    print("img shape",np.shape(noisy))
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                        sharex=True, sharey=True)
    plt.gray()
    # Estimate the average noise standard deviation across color channels.
    sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
    # Due to clipping in random_noise, the estimate will be a bit smaller than the
    # specified sigma.
    print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))
    ax[0, 0].imshow(noisy)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Noisy')
    ax[0, 1].imshow(denoise_tv_chambolle(noisy, weight=0.1, multichannel=True))
    ax[0, 1].axis('off')
    ax[0, 1].set_title('TV')
    ax[0, 2].imshow(denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,
                    multichannel=True))
    ax[0, 2].axis('off')
    ax[0, 2].set_title('Bilateral')
    ax[0, 3].imshow(denoise_wavelet(noisy, multichannel=True))
    ax[0, 3].axis('off')
    ax[0, 3].set_title('Wavelet denoising')

    ax[1, 1].imshow(denoise_tv_chambolle(noisy, weight=0.02, multichannel=True))
    ax[1, 1].axis('off')
    ax[1, 1].set_title('(more) TV')
    ax[1, 2].imshow(denoise_bilateral(noisy, sigma_color=0.001, sigma_spatial=3,
                    multichannel=True))
    ax[1, 2].axis('off')
    ax[1, 2].set_title('(more) Bilateral')
    ax[1, 3].imshow(denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True))
    ax[1, 3].axis('off')
    ax[1, 3].set_title('Wavelet denoising\nin YCbCr colorspace')
    ax[1, 0].imshow(original)
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Original')

    fig.tight_layout()

    plt.show()

def opencv_denoise(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    dst = cv2.fastNlMeansDenoisingColored(img,None,5,5,3,7)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.show()

def fft_denoise(file_name):
    """
    ======================
    Image denoising by FFT
    ======================
    Denoise an image (:download:`../../../../data/moonlanding.png`) by
    implementing a blur with an FFT.

    Implements, via FFT, the following convolution:

    .. math::

        f_1(t) = \int dt'\, K(t-t') f_0(t')

    .. math::

        \tilde{f}_1(\omega) = \tilde{K}(\omega) \tilde{f}_0(\omega)

    """
    ############################################################
    # Read and plot the image
    ############################################################
    im = plt.imread(file_name).astype(float)
    im = im[:,:,0]
    plt.figure()
    plt.imshow(im, plt.cm.gray)
    plt.title('Original image')
    ############################################################
    # Compute the 2d FFT of the input image
    ############################################################
    im_fft = fftpack.fft2(im)
    # Show the results
    def plot_spectrum(im_fft):
        from matplotlib.colors import LogNorm
        # A logarithmic colormap
        plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
        plt.colorbar()
    plt.figure()
    plot_spectrum(im_fft)
    plt.title('Fourier transform')
    ############################################################
    # Filter in FFT
    ############################################################
    # In the lines following, we'll make a copy of the original spectrum and
    # truncate coefficients.

    # Define the fraction of coefficients (in each direction) we keep
    keep_fraction = 0.1
    # Call ff a copy of the original transform. Numpy arrays have a copy
    # method for this purpose.
    im_fft2 = im_fft.copy()
    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft2.shape
    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    # Similarly with the columns:
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    plt.figure()
    plot_spectrum(im_fft2)
    plt.title('Filtered Spectrum')
    ############################################################
    # Reconstruct the final image
    ############################################################
    # Reconstruct the denoised image from the filtered spectrum, keep only the
    # real part for display.
    im_new = fftpack.ifft2(im_fft2).real
    plt.figure()
    plt.imshow(im_new, plt.cm.gray)
    plt.title('Reconstructed Image')
    ############################################################
    # Easier and better: :func:`scipy.ndimage.gaussian_filter`
    ############################################################
    #
    # Implementing filtering directly with FFTs is tricky and time consuming.
    # We can use the Gaussian filter from :mod:`scipy.ndimage`
    from scipy import ndimage
    im_blur = ndimage.gaussian_filter(im, 4)
    plt.figure()
    plt.imshow(im_blur, plt.cm.gray)
    plt.title('Blurred image')
    plt.show()

def test(file_name):
    img = cv2.imread(file_name)
    img_org = img
    '''
    nd_ar = mx.nd.array(img)
    print(nd_ar[0,:20,0])
    img_f = mx.ndarray.transpose(nd_ar,axes=(2,0,1))
    img_fl = mx.ndarray.flip(data=img_f,axis=2)
    img_o = mx.ndarray.transpose(img_fl,axes=(1,2,0))
    img_s = img_o.asnumpy()
    img_s.astype(np.uint8)
    '''
    img_f = np.transpose(img,(2,0,1))
    img_fl = np.flip(img_f,axis=2)
    img_s = np.transpose(img_fl,(1,2,0))
    cv2.imshow("show",img_s)
    cv2.imshow("org",img_org)
    cv2.waitKey(0)



if __name__ == '__main__':
    args = parm()
    file_name = args.img_dir
    #denoise_img(file_name)
    #opencv_denoise(file_name)
    #fft_denoise(file_name)
    test(file_name)
