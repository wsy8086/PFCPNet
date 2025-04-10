import glob, cv2, os
import numpy as np
from skimage import io,util
#from skimage import io, img_as_ubyte
#from model.INR_v28_NOdc import DN
#from model.INR_V28_NOCOR import DN
# from Amodel.PFCPN.INR_v28_0 import DN
from Amodel.PFCPN.INR_V28_K import DN

import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn as nn
# from utils import AverageMeter, batch_SSIM, batch_PSNR
from scipy.io import loadmat
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint = torch.load(r'F:\Pan\results\Amodel\PFCPN\checkpoint_epoch_0010_16_41.48_0.9752.pth.tar', map_location=device)
# model = DN().to(device)
#
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()
checkpoint = torch.load(r'F:\Pan\wsy_results\K2\checkpoint_epoch_0011_15_41.30_0.9730.pth.tar', map_location=device)
model = DN().to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

psnr = 0
ssim = 0
loss_val = 0
criterion = nn.L1Loss()
model.eval()

mat_path = r'F:\Pan\results\visual\Nam\1patch0100.mat'
data_dict = loadmat(mat_path)
noisy_img = data_dict['noisy_pchs'] / 255.0
gt_img = data_dict['gt_pchs'] / 255.0
noisy_img = np.transpose(noisy_img, axes=[0, 3, 1, 2]).astype('float32')
gt_img = np.transpose(gt_img, axes=[0, 3, 1, 2]).astype('float32')
iter_pch = gt_img.shape[0]
noisy_img = torch.from_numpy(noisy_img).cuda()
gt_img = torch.from_numpy(gt_img).cuda()

for i in range(int(iter_pch)):
        input = Variable(noisy_img[i, :, :, :].unsqueeze(0))
        gt = Variable(gt_img[i, :, :, :].unsqueeze(0))

        # input = model_h(input)
        with torch.no_grad():
                _,test_out = model(input)
        # test_out = input - test_out
        im_denoise = test_out.data
        im_denoise.clamp_(0.0, 1.0)

        # print(psnr_iter,ssim_iter)
        im_denoise = util.img_as_ubyte(im_denoise[0, ...].cpu().numpy().transpose([1, 2, 0]))
        io.imsave(r'F:\Pan\results\visual\Nam\cfnet' + str(i + 1) + '}.png', im_denoise)
