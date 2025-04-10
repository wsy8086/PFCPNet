import glob, cv2, os
import numpy as np
from skimage import io,util
from Amodel.PFCPN.INR_v28_0 import DN
from Amodel.dncnn.zk_dncnn import DnCNN
from Amodel.ridnet.TestCode.code.model import  Model
# from Amodel.GrencNet.GrencNet import DN
from  Amodel.casapunet.CasaPuNet import Network
# from Amodel.CBDnet.cbdnet import Network
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint = torch.load(r'F:\代码需要学习的需要实验的\results\Amodel\PFCPN\checkpoint_epoch_0010_16_41.48_0.9752.pth.tar', map_location=device)
# model = DN().to(device)
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()

model = Network()
model =torch.nn.DataParallel(model).cuda()
weight = torch.load(r'F:\代码需要学习的需要实验的\results\Amodel\casapunet\checkpoint.pth.tar')
model.load_state_dict(weight['state_dict'])
model.eval()

# checkpoint = torch.load(r'F:\代码需要学习的需要实验的\results\Amodel\CBDnet\checkpoint.pth.tar', map_location=device)
# model = Network().to(device)
# model = torch.nn.DataParallel(model)
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()

# model = DnCNN().to(device)
# w = torch.load(r'F:\代码需要学习的需要实验的\results\Amodel\dncnn\dncnn_color_blind.pth')
# model.load_state_dict(w)
# model.eval()
# checkpoint = torch.load(r'F:\代码需要学习的需要实验的\results\Amodel\GrencNet\checkpoint_39.45_0.9188.pth', map_location=device)
# model = DN().to(device)
# model.load_state_dict(checkpoint)
# model.eval()


image_list = glob.glob('./v/real/IM' + "/*")

save_denoised_image = True

psnr = 0
ssim = 0
img_nums = len(image_list)

for i in range(img_nums):
    image_name = image_list[i]
    print('Image: {:02d}, path: {:s}'.format(i + 1, image_name))
    path_label = os.path.split(image_name)
    gt_name = path_label[1].split('.')[0]#.replace('_noise', '')
    #im_input = io.imread(image_name)          #RGB
    im_input = io.imread(image_name)[:, :, :3] #RGBA
    im_input = im_input
    im_input = np.transpose(im_input, axes=[2, 0, 1]).astype('float32') / 255.0
    noisy_img = torch.from_numpy(im_input).to(device)
    noisy_img = noisy_img.unsqueeze(0)
    with torch.no_grad():
        #noise_map, test_out = model(noisy_img)
        _,test_out = model(noisy_img)
    im_denoise = test_out.data.cpu()
    im_denoise.clamp_(0.0, 1.0)
    im_denoise = util.img_as_ubyte(im_denoise.squeeze(0).numpy().transpose([1, 2, 0]))

    _, save_name = os.path.split(image_name)
    if save_denoised_image:
        io.imsave(('./v/real/IM_grenc/'+ gt_name + '.png'), im_denoise)

print("///////////////////////////////////////")
print('PSNR: {:.2f}, SSIM: {:.4f}'.format(psnr / img_nums, ssim / img_nums))