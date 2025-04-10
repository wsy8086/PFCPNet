from .common import get_image_paths,AverageMeter, ListAverageMeter, read_img, hwc_to_chw, chw_to_hwc
from .syn import ISP_implement
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error
from skimage import img_as_ubyte
import sys
from .common import DWT,IWT
def batch_PSNR(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    for i in range(Img.shape[0]):
             PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=255)
    return (PSNR/Img.shape[0])

def batch_MSE(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += mean_squared_error(Iclean[i,:,:,:], Img[i,:,:,:])
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += ssim_index(Iclean[i,:,:,:].transpose((1,2,0)), Img[i,:,:,:].transpose((1,2,0)))
    return (SSIM/Img.shape[0])
def ssim_index(im1, im2):
    '''
    Input:
        im1, im2: np.uint8 format
    '''
    if im1.ndim == 2:
        out = compare_ssim(im1, im2, data_range=255, gaussian_weights=True,
                                                    use_sample_covariance=False, multichannel=False)
    elif im1.ndim == 3:
        out = compare_ssim(im1, im2, data_range=255, gaussian_weights=True,
                           use_sample_covariance=False, multichannel=True)
    else:
        sys.exit('Please input the corrected images')
    return out
