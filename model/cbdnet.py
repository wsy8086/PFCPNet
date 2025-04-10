
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from math import exp


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.fcn(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            single_conv(4, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 3)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()
    
    def forward(self, x):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        return noise_level, out


# class fixed_loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
#         l2_loss = F.mse_loss(out_image, gt_image)
#
#         asym_loss = torch.mean(if_asym * torch.abs(0.3 - F.relu(gt_noise - est_noise)) * torch.pow(est_noise - gt_noise, 2))
#
#         h_x = est_noise.size()[2]
#         w_x = est_noise.size()[3]
#         count_h = self._tensor_size(est_noise[:, :, 1:, :])
#         count_w = self._tensor_size(est_noise[:, :, : ,1:])
#         h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
#         w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
#         tvloss = h_tv / count_h + w_tv / count_w
#
#         loss = l2_loss +  0.5 * asym_loss + 0.05 * tvloss
#
#         return loss
#
#     def _tensor_size(self,t):
#         return t.size()[1]*t.size()[2]*t.size()[3]

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class log_SSIM_loss(nn.Module):
    def __init__(self, window_size=11, channel=3, is_cuda=True, size_average=True):
        super(log_SSIM_loss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = create_window(window_size, channel)
        if is_cuda:
            self.window = self.window.cuda()


    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return -torch.log10(ssim_map.mean())

class fixed_loss_est(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight,est_noise, gt_noise, if_asym,criterionL1):
        # l1_loss = criterionL1(out_image,gt_image)
        # loss_ssim = self.loss_dict(out_image, gt_image)
        # l2_loss = F.mse_loss(out_image, gt_image)
        asym_loss = 0
        tvloss = 0


        asym_loss += criterionL1(if_asym*est_noise,if_asym*gt_noise)
                         # torch.mean(if_asym*torch.abs(mask - 0.25) * (torch.pow(est_noise[i] - gt_noise, 2)))
            # asym_loss = torch.mean(
            #     if_asym * torch.abs(0.3 - F.relu(gt_noise - est_noise)) * torch.pow(est_noise - gt_noise, 2))
         ##############################################################
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
        tvloss += h_tv / count_h + w_tv / count_w
        ###########################################
        # print(asym_loss)
        loss = asym_loss + tvloss*0.0005

        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class fixed_loss_est_gray(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, est_noise, gt_noise, criterionL1):
        # l1_loss = criterionL1(out_image,gt_image)
        # loss_ssim = self.loss_dict(out_image, gt_image)
        # l2_loss = F.mse_loss(out_image, gt_image)
        asym_loss = 0
        tvloss = 0


        asym_loss += criterionL1(est_noise, gt_noise)
        loss = asym_loss

        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class fixed_loss_est2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight,est_noise, gt_noise, if_asym,criterionL1):
        # l1_loss = criterionL1(out_image,gt_image)
        # loss_ssim = self.loss_dict(out_image, gt_image)
        # l2_loss = F.mse_loss(out_image, gt_image)
        asym_loss = 0
        tvloss = 0


        asym_loss += criterionL1(if_asym*est_noise,if_asym*gt_noise)
                         # torch.mean(if_asym*torch.abs(mask - 0.25) * (torch.pow(est_noise[i] - gt_noise, 2)))
            # asym_loss = torch.mean(
            #     if_asym * torch.abs(0.3 - F.relu(gt_noise - est_noise)) * torch.pow(est_noise - gt_noise, 2))
         ##############################################################
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow(torch.abs(est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 1).sum()
        w_tv = torch.pow(torch.abs(est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 1).sum()
        tvloss += h_tv / count_h + w_tv / count_w
        ###########################################
        # print(asym_loss)
        loss = asym_loss + tvloss*0.001

        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class fixed_loss_est3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight,est_noise, gt_noise, if_asym,criterionL1):
        # l1_loss = criterionL1(out_image,gt_image)
        # loss_ssim = self.loss_dict(out_image, gt_image)
        # l2_loss = F.mse_loss(out_image, gt_image)
        asym_loss = 0
        tvloss = 0


        asym_loss += criterionL1(if_asym*est_noise,if_asym*gt_noise)
                         # torch.mean(if_asym*torch.abs(mask - 0.25) * (torch.pow(est_noise[i] - gt_noise, 2)))
            # asym_loss = torch.mean(
            #     if_asym * torch.abs(0.3 - F.relu(gt_noise - est_noise)) * torch.pow(est_noise - gt_noise, 2))
         ##############################################################
        # h_x = est_noise.size()[2]
        # w_x = est_noise.size()[3]
        # count_h = self._tensor_size(est_noise[:, :, 1:, :])
        # count_w = self._tensor_size(est_noise[:, :, :, 1:])
        # h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
        # w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
        # tvloss += h_tv / count_h + w_tv / count_w
        ###########################################
        # print(asym_loss)
        loss = asym_loss

        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]




class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_dict = log_SSIM_loss()

    def forward(self, weight,out_image, gt_image, est_noise, gt_noise, if_asym,criterionL1):
        l1_loss = criterionL1(out_image,gt_image)
        loss_ssim = self.loss_dict(out_image, gt_image)
        # l2_loss = F.mse_loss(out_image, gt_image)
        asym_loss = 0
        tvloss = 0

        for i in range(len(est_noise)):
            # mask = torch.FloatTensor(torch.zeros(gt_noise.size()))
            # mask[(gt_noise - est_noise[i]) < 0] = 1
            # mask = torch.autograd.Variable(mask.cuda())
            asym_loss += weight[i]*criterionL1(if_asym*est_noise[i],if_asym*gt_noise)
                         # torch.mean(if_asym*torch.abs(mask - 0.25) * (torch.pow(est_noise[i] - gt_noise, 2)))
            # asym_loss = torch.mean(
            #     if_asym * torch.abs(0.3 - F.relu(gt_noise - est_noise)) * torch.pow(est_noise - gt_noise, 2))
         #############################################################
            #h_x = est_noise[i].size()[2]
            #w_x = est_noise[i].size()[3]
            #count_h = self._tensor_size(est_noise[i][:, :, 1:, :])
            #count_w = self._tensor_size(est_noise[i][:, :, :, 1:])
            #h_tv = torch.pow(torch.abs(est_noise[i][:, :, 1:, :] - est_noise[i][:, :, :h_x - 1, :]), 1).sum()
            #w_tv = torch.pow(torch.abs(est_noise[i][:, :, :, 1:] - est_noise[i][:, :, :, :w_x - 1]), 1).sum()
            #tvloss +=weight[i]*( h_tv / count_h + w_tv / count_w)
        ##########################################
        # print(asym_loss)
        loss = l1_loss + 0.1*asym_loss + 0.25*loss_ssim

        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class fixed_loss_gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_dict = log_SSIM_loss()

    def forward(self, weight,out_image, gt_image, est_noise, gt_noise,criterionL1):
        l1_loss = criterionL1(out_image,gt_image)
        loss_ssim = self.loss_dict(out_image, gt_image)
        # l2_loss = F.mse_loss(out_image, gt_image)
        asym_loss = 0
        tvloss = 0

        for i in range(len(est_noise)):
            # mask = torch.FloatTensor(torch.zeros(gt_noise.size()))
            # mask[(gt_noise - est_noise[i]) < 0] = 1
            # mask = torch.autograd.Variable(mask.cuda())
            asym_loss += weight[i]*criterionL1(est_noise[i], gt_noise)
                         # torch.mean(if_asym*torch.abs(mask - 0.25) * (torch.pow(est_noise[i] - gt_noise, 2)))
            # asym_loss = torch.mean(
            #     if_asym * torch.abs(0.3 - F.relu(gt_noise - est_noise)) * torch.pow(est_noise - gt_noise, 2))
         #############################################################
            #h_x = est_noise[i].size()[2]
            #w_x = est_noise[i].size()[3]
            #count_h = self._tensor_size(est_noise[i][:, :, 1:, :])
            #count_w = self._tensor_size(est_noise[i][:, :, :, 1:])
            #h_tv = torch.pow(torch.abs(est_noise[i][:, :, 1:, :] - est_noise[i][:, :, :h_x - 1, :]), 1).sum()
            #w_tv = torch.pow(torch.abs(est_noise[i][:, :, :, 1:] - est_noise[i][:, :, :, :w_x - 1]), 1).sum()
            #tvloss +=weight[i]*( h_tv / count_h + w_tv / count_w)
        ##########################################
        # print(asym_loss)
        loss = l1_loss + 0.1*asym_loss + 0.25*loss_ssim

        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]