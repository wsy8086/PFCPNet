import glob, cv2, os
import numpy as np
from skimage import io,util
# from Amodel.PFCPN.INR_v28_0 import DN
from Amodel.PFCPN.INR_V28_K import DN
from Amodel.dncnn.zk_dncnn import DnCNN
from Amodel.ridnet.TestCode.code.model import  Model
from Amodel.CBDnet.cbdnet import Network
#from Amodel.GrencNet.GrencNet import DN
from Amodel.casapunet.CasaPuNet import Network
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(r'F:\Pan\wsy_results\K2\checkpoint_epoch_0011_15_41.30_0.9730.pth.tar', map_location=device)
model = DN().to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# checkpoint = torch.load(r'F:\代码需要学习的需要实验的\results\Amodel\CBDnet\checkpoint.pth.tar', map_location=device)
# model = Network().to(device)
# model = torch.nn.DataParallel(model)
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()
#
# model = DnCNN().to(device)
# w = torch.load(r'F:\代码需要学习的需要实验的\results\Amodel\dncnn\dncnn_color_blind.pth')
# model.load_state_dict(w)
# model.eval()

# checkpoint = torch.load(r'F:\代码需要学习的需要实验的\results\Amodel\GrencNet\checkpoint_39.45_0.9188.pth', map_location=device)
# model = DN().to(device)
# model.load_state_dict(checkpoint)
# model.eval()
# model = Network()
# model =torch.nn.DataParallel(model).cuda()
# weight = torch.load(r'F:\代码需要学习的需要实验的\results\Amodel\casapunet\checkpoint.pth.tar')
# model.load_state_dict(weight['state_dict'])
# model.eval()

image_list = glob.glob(r'F:/Pan/results/visual/SIDD/2-SIDD_noise' + "/*")
#image_list = glob.glob('F:/work/319low-light/0.5lux/low'+"/*")
save_denoised_image = True

psnr = 0
ssim = 0
img_nums = len(image_list)

for i in range(img_nums):
    image_name = image_list[i]
    print('Image: {:02d}, path: {:s}'.format(i + 1, image_name))
    path_label = os.path.split(image_name)
    gt_name = path_label[1].split('.')[0]#.replace('_noise', '')
    im_input = io.imread(image_name)
    #3840x2160太大要切块
    im_input = im_input


    im_input = np.transpose(im_input, axes=[2, 0, 1]).astype('float32') / 255.0
    noisy_img = torch.from_numpy(im_input).to(device)
    noisy_img = noisy_img.unsqueeze(0)
    # print(noisy_img.shape)
    with torch.no_grad():
        #noise_map, test_out = model(noisy_img)
        _ ,test_out = model(noisy_img)
    im_denoise = test_out.data.cpu()
    im_denoise.clamp_(0.0, 1.0)
    im_denoise = util.img_as_ubyte(im_denoise.squeeze(0).numpy().transpose([1, 2, 0]))

    #gt = io.imread('./noise/siddgt/'+'/' + gt_name + '.jpg')
    # gt = io.imread('F:/work/319low-light/0.5lux/high' + '/' + gt_name + '.png')
    # psnr_iter = compare_psnr(im_denoise, gt, data_range=255)
    # ssim_iter = compare_ssim(im_denoise, gt, channel_axis=2)
    # psnr += psnr_iter
    # ssim += ssim_iter
    noise, save_name = os.path.split(image_name)
    if save_denoised_image:
        #io.imsave(os.path.join('./v/SIDD/sec1/zk5/', gt_name + '_{:.2f}.jpg'.format(psnr_iter)), im_denoise)
        io.imsave(os.path.join('./visual/SIDD/9-CFnet/', gt_name + '.png'), im_denoise)
   #  for i in range(64):
   #   noises = _[4][:,i:i+1,:,:].data.cpu()
   # # noises = noises.mean(dim=1)
   #   noises.clamp_(0.0,1.0)
   #   noises = util.img_as_ubyte(noises.squeeze(0).numpy().transpose([1,2,0]))  # 256,256,1
   #   cv2.imwrite(os.path.join('./noise/m4/', str(i) + '_-1.jpg'.format(psnr_iter)), noises)

#
# print("///////////////////////////////////////")
# print('PSNR: {:.2f}, SSIM: {:.4f}'.format(psnr / img_nums, ssim / img_nums))