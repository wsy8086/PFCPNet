import os, time, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg
from skimage.measure import compare_psnr, compare_ssim
from utils import AverageMeter, batch_SSIM, batch_PSNR
from dataset.loader import Syn_JPEG, BenchmarkTest, BenchmarkTrain, DatasetFromFolder_JPEG
from model.cbdnet import Network, fixed_loss
from model.INR_v28_0 import DN
# from benchmark_test.Net_v9_ab import DN, print_network
import numpy as np
import time
from scipy.io import loadmat
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--ps', default=64, type=int, help='patch size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=5000, type=int, help='sum of epochs')
parser.add_argument('--val_path', type=str, default="/media/sr617/新加卷2/test_2021/data/val/small_imgs_test.hdf5",
                    help='validating dataset path, if not, set None')
parser.add_argument('--src_path', type=str, default="/media/sr617/新加卷2/test_2021/DIV2K_HR_256_JPEG",
                    help='training dataset path')
parser.add_argument('--src_path_real', type=str, default="/media/sr617/新加卷2/test_2021/train.h5",
                    help='training dataset path')
parser.add_argument('--src_path_JPEG', type=str, default="./PolyU/OriginalImages/",
                    help='training dataset path')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='save model per every N epochs')
# parser.add_argument('--finetune', type=bool, default=False,
#                     help='if finetune model, set True')
parser.add_argument('--init_epoch', type=int, default=19000,
                    help='if finetune model, set the initial epoch')
parser.add_argument('--gpu', type=str, default="1",
                    help='GPUs')
parser.add_argument('--save_val_img', type=bool, default=True,
                    help='save the last validated image for comparison')
parser.add_argument('--gray', type=bool, default=False,
                    help='if gray image, set True; if color image, set False')
#parser.add_argument('--pretrained_1', type=str, default='/media/sr617/77C45040838A76BE/pyz/abc/save_model_v8/checkpoint_epoch_0009_5_39.13_0.9162.pth.tar',
 #                    help='training loss')#/media/sr617/77C45040838A76BE/pyz/abc/save_model_v6/checkpoint_epoch_0005_3_38.16_0.9108.pth.tar
# parser.add_argument('--transform_weights', default='/media/sr617/77C45040838A76BE/pyz/abc/save_model_v6/checkpoint_epoch_0000_3_36.94_0.9011.pth.tar',
#                      help='training loss')

# parser.add_argument('--pretrained_1', type=str, default='/media/sr617/新加卷2/test_2021/abc/不同残差块数/save_model_crb_8/checkpoint_epoch_0009_1_39.03_0.9156.pth.tar',
#                    help='training loss')#checkpoint_epoch_0030_4_39.48_0.9187.pth.tar
# parser.add_argument('--pretrained_1', type=str, default='./benchmark_test/test_model/checkpoint_epoch_0032_4_39.48_0.9186.pth.tar',
#                    help='training loss')#checkpoint_epoch_0030_4_39.48_0.9187.pth.tar
# parser.add_argument('--pretrained_1', type=str, default='./save_model_map_est/checkpoint_epoch_0008_5_0.00037682.pth.tar',
#                      help='training loss')#checkpoint_epoch_0003_4_38.79_0.9140
parser.add_argument('--pretrained_1', type=str, default='./save_model_P_JPEG/v12/checkpoint_epoch_0001_3_39.23_0.9174.pth.tar',
                     help='training loss')
# parser.add_argument('--pretrained_1', type=str, default='./save_model_P_JPEG/v1/checkpoint_epoch_0002_13_41.27_0.9773.pth.tar',
#                      help='training loss')
# parser.add_argument('--pretrained_1', type=str, default='./benchmark_test/test_model_p/checkpoint_epoch_0080_22_39.47_0.9189.pth.tar',
#                      help='training loss')
parser.add_argument('--pretrained_2', type=str, default='1./save_model_P_JPEG/est/checkpoint_epoch_0031_35.27_0.00307676.pth.tar',
                     help='training loss')

args = parser.parse_args()
psnr_max = 36

def train(train_loader, model, criterion, optimizer, epoch, epoch_total,criterionL1):
    loss_sum = 0
    losses = AverageMeter()
    model.train()
    start_time = time.time()
    # weight = [1, 1, 1, 1]
    global psnr_max
    weight = [3 / 255, 12 / 255, 48 / 255, 192 / 255]
    # weight = [1 / 21, 4 / 21, 16 / 21]
    # weight = [1 / 5, 4 / 5]
    # weight = [1]
    # weight = [1 / 341, 4 / 341, 16 / 341, 64 / 341, 256 / 341]

    for i, (noise_img, clean_img, sigma_img, flag) in enumerate(train_loader):
        # t1 = time.time()
        input_var = Variable(noise_img.cuda())
        target_var = Variable(clean_img.cuda())
        sigma_var = Variable(sigma_img.cuda())
        flag_var = Variable(flag.cuda()).float()

        noise_level_est, output = model(input_var)

        loss = criterion(weight,output, target_var, noise_level_est, sigma_var, flag_var,criterionL1)

        loss_sum+=loss.item()
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(time.time() - t1)
        if (i % 10 == 0) and (i != 0):
            loss_avg = loss_sum / 10
            loss_sum = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f} Time: {:4.4f}s".format(
                epoch + 1, epoch_total, i + 1, len(train_loader), loss_avg, time.time() - start_time))
            start_time = time.time()
            # psnr, ssim = test_Nam(model)

        if (i % 2000 == 0) and (i != 0):
            psnr = 0
            ssim = 0
            # psnr, ssim = test_Nam(model)
            psnr, ssim = test(dataloader_val, model, criterionL1)
            if psnr > psnr_max - 0.2:
                psnr_max = max(psnr, psnr_max)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.save_dir,
                                 'checkpoint_epoch_{:0>4}_{}_{:.2f}_{:.4f}.pth.tar'.format(epoch, i//2000, psnr, ssim)))
            else:
                torch.save({}, os.path.join(args.save_dir,
                                            'checkpoint_epoch_{:0>4}_{}_{:.2f}_{:.4f}.pth.tar'.format(epoch, i//2000, psnr, ssim)))
        # Record train loss
        # writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
        # Record learning rate
        # writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
        # writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

    return losses.avg


# def test(dataloader_val, model, criterion, epoch,args):
#     psnr = 0
#     loss_val = 0
#     model.eval()
#     for i, data in enumerate(dataloader_val):
#         input, label = data
#         if torch.cuda.is_available():
#             input, label = input.cuda(), label.cuda()
#         input, label = Variable(input), Variable(label)
#         with torch.no_grad():
#
#
#             noise_est,test_out = model(input)
#         test_out.detach_()
#
#         # 计算loss
#         loss_val += criterion(test_out, label).item()
#         rgb_out = test_out.cpu().numpy().transpose((0, 2, 3, 1))
#         clean = label.cpu().numpy().transpose((0, 2, 3, 1))
#         for num in range(rgb_out.shape[0]):
#             denoised = np.clip(rgb_out[num], 0, 1)
#             psnr += compare_psnr(clean[num], denoised)
#     img_nums = rgb_out.shape[0] * len(dataloader_val)
#     # img_nums = batch_size * len(dataloader_val)
#     psnr = psnr / img_nums
#     loss_val = loss_val / len(dataloader_val)
#     print('Validating: {:0>3} , loss: {:.8f}, PSNR: {:4.4f}'.format(img_nums, loss_val, psnr))
#     # mpimg.imsave(ckpt_dir+"img/%04d_denoised.png" % epoch, rgb_out[0])
#     # if args.save_val_img:
#     #     if args.gray:
#     #         mpimg.imsave(args.save_dir + "img/%04d_denoised.png" % epoch, denoised[:, :, 0])
#     #     else:
#     #         mpimg.imsave(args.save_dir + "img/%04d_denoised.png" % epoch, denoised)
#     return psnr

def test(dataloader_val, model, criterion):
    psnr = 0
    ssim = 0
    loss_val = 0
    model.eval()
    for i, data in enumerate(dataloader_val):
        input, label = data
        if torch.cuda.is_available():
            input, label = input.cuda(), label.cuda()
        input, label = Variable(input), Variable(label)
        with torch.no_grad():

            noisy_map, test_out= model(input)


        # test_out = test_out.detach_()
        loss_val += criterion(test_out, label).item()

        im_denoise = test_out.data
        im_denoise.clamp_(0.0, 1.0)
        # 计算loss

        psnr_iter = batch_PSNR(im_denoise, label)
        ssim_iter = batch_SSIM(im_denoise, label)
        psnr+=psnr_iter
        ssim+=ssim_iter
        # rgb_out = test_out.cpu().numpy().transpose((0, 2, 3, 1))
        # clean = label.cpu().numpy().transpose((0, 2, 3, 1))
        # for num in range(rgb_out.shape[0]):
        #     denoised = np.clip(rgb_out[num], 0, 1)
        #     psnr += compare_psnr(clean[num], denoised)
    img_nums = len(dataloader_val)
    # img_nums = batch_size * len(dataloader_val)
    psnr = psnr / img_nums
    ssim = ssim / img_nums

    loss_val = loss_val / len(dataloader_val)
    print('Validating: {:0>3} , loss: {:.8f}, PSNR: {:4.4f}, SSIM: {:5.4f}, PSNR_max: {:4.4f}'.format(img_nums, loss_val, psnr, ssim, psnr_max))
    # mpimg.imsave(ckpt_dir+"img/%04d_denoised.png" % epoch, rgb_out[0])
    # if args.save_val_img:
    #     if args.gray:
    #         mpimg.imsave(args.save_dir + "img/%04d_denoised.png" % epoch, denoised[:, :, 0])
    #     else:
    #         mpimg.imsave(args.save_dir + "img/%04d_denoised.png" % epoch, denoised)
    return psnr,ssim

def test_Nam(model):
    psnr = 0
    ssim = 0
    loss_val = 0
    criterion = nn.L1Loss()
    model.eval()

    mat_path = '/media/sr617/新加卷2/test_2021/abc/Nam/1patch0100.mat'
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
        label = Variable(gt_img[i, :, :, :].unsqueeze(0))
        if torch.cuda.is_available():
            input, label = input.cuda(), label.cuda()

        with torch.no_grad():

            noisy_map, test_out= model(input)

        # test_out = test_out.detach_()
        loss_val += criterion(test_out, label).item()

        im_denoise = test_out.data
        im_denoise.clamp_(0.0, 1.0)
        # 计算loss

        psnr_iter = batch_PSNR(im_denoise, label)
        ssim_iter = batch_SSIM(im_denoise, label)
        psnr+=psnr_iter
        ssim+=ssim_iter
        # rgb_out = test_out.cpu().numpy().transpose((0, 2, 3, 1))
        # clean = label.cpu().numpy().transpose((0, 2, 3, 1))
        # for num in range(rgb_out.shape[0]):
        #     denoised = np.clip(rgb_out[num], 0, 1)
        #     psnr += compare_psnr(clean[num], denoised)
    img_nums = iter_pch
    # img_nums = batch_size * len(dataloader_val)
    psnr = psnr / img_nums
    ssim = ssim / img_nums

    loss_val = loss_val / img_nums
    print('Validating: {:0>3} , loss: {:.8f}, PSNR: {:4.4f}, SSIM: {:5.4f}, PSNR_max: {:4.4f}'.format(img_nums, loss_val, psnr, ssim, psnr_max))
    # mpimg.imsave(ckpt_dir+"img/%04d_denoised.png" % epoch, rgb_out[0])
    # if args.save_val_img:
    #     if args.gray:
    #         mpimg.imsave(args.save_dir + "img/%04d_denoised.png" % epoch, denoised[:, :, 0])
    #     else:
    #         mpimg.imsave(args.save_dir + "img/%04d_denoised.png" % epoch, denoised)
    return psnr,ssim

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.save_dir = './save_model_P_JPEG/v13/'
    model = DN()
    # model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_0120_dict.pth')))
    criterion = fixed_loss()
    criterionL1 = nn.L1Loss()
    # criterion.cuda()


    model.cuda()
    # model = nn.DataParallel(model)
    if os.path.exists(os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch))):
        # load existing model
        model_info = torch.load(os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch)))
        print('==> loading existing model:',
              os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch)))
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler.load_state_dict(model_info['scheduler'])
        cur_epoch = model_info['epoch']
    else:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        # create model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        cur_epoch = 0

    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            #model = torch.nn.DataParallel(modelte, device_ids=[0]).cuda()
            model = torch.nn.DataParallel(model)
            criterion = criterion.cuda()
            criterionL1 = criterionL1.cuda()
        else:
            model = model.cuda()
            criterion = criterion.cuda()
            criterionL1 = criterionL1.cuda()

    # print_network(model)
    if args.pretrained_1:
        if os.path.isfile(args.pretrained_1):
            print("=> loading model '{}'".format(args.pretrained_1))
            # model_pretrained = torch.load(opt.pretrained_1, map_location=lambda storage, loc: storage)
            model_pretrained = torch.load(args.pretrained_1)
            # print(model_pretrained['state_dict'])
            # for param in list(model_pretrained.parameters()):
            #     param.requires_grad = False
            pretrained_dict = model_pretrained['state_dict']
            model_dict = model.state_dict()
            # 筛除不加载的层结构
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新当前网络的结构字典
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print("=> no model found at '{}'".format(args.pretrained_1))

    if args.pretrained_2:
        if os.path.isfile(args.pretrained_2):
            print("=> loading model '{}'".format(args.pretrained_2))
            # model_pretrained = torch.load(opt.pretrained_1, map_location=lambda storage, loc: storage)
            model_pretrained = torch.load(args.pretrained_2)
            # print(model_pretrained['state_dict'])
            # for param in list(model_pretrained.parameters()):
            #     param.requires_grad = False
            pretrained_dict = model_pretrained['state_dict']
            model_dict = model.state_dict()
            # 筛除不加载的层结构
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "P." in k}
            # 更新当前网络的结构字典
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    print(k)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print("=> no model found at '{}'".format(args.pretrained_2))

    for name, param in model.named_parameters():
        if  "P." in name:
            param.requires_grad = False
            print(name)
        # else:
        #     param.requires_grad = True
    # Real('./data/SIDD_train/', 320, args.ps) +
    train_dataset = BenchmarkTrain(args.src_path_real,pch_size = args.ps) \
                    + Syn_JPEG(args.src_path, 100, args.ps) + DatasetFromFolder_JPEG(args.src_path_JPEG, args.ps)
    # train_dataset = Syn(args.src_path, 100, args.ps)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    dataset_val = BenchmarkTest(args.val_path)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=0,
                                                 drop_last=True)

    print('load dataset')
    for epoch in range(cur_epoch, args.epochs + 1):
        loss = train(train_loader, model, criterion, optimizer, epoch, args.epochs + 1,criterionL1)
        scheduler.step()

        if epoch % args.save_epoch == 0:
            psnr = 0
            ssim = 0
            # psnr,ssim = test_Nam(model)
            psnr, ssim = test(dataloader_val, model, criterionL1)

            if psnr > psnr_max - 0.2:
                psnr_max = max(psnr, psnr_max)
                torch.save({
                  'epoch': epoch + 1,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()},
                  os.path.join(args.save_dir, 'checkpoint_epoch_{:0>4}_19_{:.2f}_{:.4f}.pth.tar'.format(epoch, psnr, ssim)))
            else:
                torch.save({},os.path.join(args.save_dir, 'checkpoint_epoch_{:0>4}_19_{:.2f}_{:.4f}.pth.tar'.format(epoch, psnr, ssim)))

        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}'
            .format(
            epoch,
            lr=optimizer.param_groups[-1]['lr'],
            loss=loss))
