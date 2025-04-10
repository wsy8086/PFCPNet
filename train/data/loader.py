import math
import os
import random
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
import h5py
import time
from scipy.io import loadmat, savemat

from skimage import img_as_float32 as img_as_float
import cv2
from utils import read_img, hwc_to_chw

from utils import ISP_implement
from utils import get_image_paths
import torch.nn.functional as F

def get_patch(imgs, patch_size):
    H = imgs[0].shape[0]
    W = imgs[0].shape[1]

    ps_temp = min(H, W, patch_size)

    xx = np.random.randint(0, W - ps_temp) if W > ps_temp else 0
    yy = np.random.randint(0, H - ps_temp) if H > ps_temp else 0

    for i in range(len(imgs)):
        imgs[i] = imgs[i][yy:yy + ps_temp, xx:xx + ps_temp, :]

    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i])
    return imgs


def data_augmentation(image, mode):
    '''
    Performs dat augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        # pass
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation_pami(*args):
    out = []
    # if random.randint(0, 7) == 1:
    flag_aug = random.randint(0, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    # else:
    #     for data in args:
    #         out.append(data)
    return out


def random_augmentation(*args):
    out = []
    if random.randint(0, 1) == 1:
        flag_aug = random.randint(1, 7)
        for data in args:
            out.append(data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data)
    return out

class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, patch_size, if_gray=False, data_augmentation = True):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        self.repeat = 40
        self.if_gray = if_gray

    def __getitem__(self, index):
        index = self._get_index(index)

        target = read_img(self.image_filenames[index],if_gray=self.if_gray)

        # input = target.resize((int(target.size[0]/self.upscale_factor),int(target.size[1]/self.upscale_factor)), Image.BICUBIC)
        #         # bicubic = rescale_img(input, self.upscale_factor)

        [target] = get_patch([target], self.patch_size)

        # if self.data_augmentation:
        #     [target] = random_augmentation([target])

        return hwc_to_chw(target)

    def __len__(self):
        return len(self.image_filenames) * self.repeat

    def _get_index(self, idx):

        return idx % len(self.image_filenames)

class DatasetFromFolder_2(Dataset):
    def __init__(self, image_dir1, image_dir2, patch_size, if_gray=False, data_augmentation = True):
        super(DatasetFromFolder_2, self).__init__()
        self.image_filenames = [os.path.join(image_dir1, x) for x in os.listdir(image_dir1)]
        self.image_dir1 = image_dir1
        self.image_dir2 = image_dir2
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        self.repeat = 40
        self.if_gray = if_gray

    def __getitem__(self, index):
        index = self._get_index(index)
        name1 = self.image_filenames[index]
        name2 = name1.replace(self.image_dir1, self.image_dir2)

        if random.randint(0, 1) == 1:
            input = read_img(name1, if_gray=self.if_gray)
            target = read_img(name2, if_gray=self.if_gray)
        else:
            input = read_img(name2, if_gray=self.if_gray)
            target = read_img(name1, if_gray=self.if_gray)

        # input = target.resize((int(target.size[0]/self.upscale_factor),int(target.size[1]/self.upscale_factor)), Image.BICUBIC)
        #         # bicubic = rescale_img(input, self.upscale_factor)

        [input, target] = get_patch([input, target], self.patch_size)

        # if self.data_augmentation:
        #     [target] = random_augmentation([target])

        return hwc_to_chw(input), hwc_to_chw(target)

    def __len__(self):
        return len(self.image_filenames) * self.repeat

    def _get_index(self, idx):

        return idx % len(self.image_filenames)


class DatasetFromFolder_JPEG(Dataset):
    def __init__(self, image_dir, patch_size, data_augmentation = True):
        super(DatasetFromFolder_JPEG, self).__init__()
        self.image_filenames_gt = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if 'mean' in x]
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        self.repeat = 400#原始400

    def __getitem__(self, index):
        index = self._get_index(index)
        gt_name = self.image_filenames_gt[index]
        target = read_img(gt_name)
        noise =  read_img(gt_name.replace('mean.JPG', 'Real.JPG'))

        # input = target.resize((int(target.size[0]/self.upscale_factor),int(target.size[1]/self.upscale_factor)), Image.BICUBIC)
        #         # bicubic = rescale_img(input, self.upscale_factor)

        [target, noise] = get_patch([target, noise], self.patch_size)

        # if self.data_augmentation:
        #     [target] = random_augmentation([target])

        return hwc_to_chw(noise), hwc_to_chw(target), np.zeros((1, self.patch_size, self.patch_size)), np.zeros(
            (1, self.patch_size, self.patch_size))#.astype("float32")

    def __len__(self):
        return len(self.image_filenames_gt) * self.repeat

    def _get_index(self, idx):
        return idx % len(self.image_filenames_gt)

class DatasetFromFolderEval(Dataset):
    def __init__(self, lr_dir, if_gray=False):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [os.path.join(lr_dir, x) for x in os.listdir(lr_dir)]
        self.if_gray = if_gray

    def __getitem__(self, index):
        input = read_img(self.image_filenames[index], if_gray=self.if_gray)
        # _, file = os.path.split(self.image_filenames[index])
        return hwc_to_chw(input)

    def __len__(self):
        return len(self.image_filenames)

class Real(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size

        folders = glob.glob(root_dir + '/*')
        folders.sort()

        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = random.choice(self.clean_fns[idx])

        clean_img = read_img(clean_fn)
        noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))

        if self.patch_size > 0:
            [clean_img, noise_img] = get_patch([clean_img, noise_img], self.patch_size)

        return hwc_to_chw(noise_img), hwc_to_chw(clean_img), np.zeros((1, self.patch_size, self.patch_size)), np.zeros(
            (1, self.patch_size, self.patch_size))


class Syn(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size

        self.clean_fns = get_image_paths(root_dir, data_type='img')


        # folders = glob.glob(root_dir + '/*')
        # clean_imgs = glob.glob(root_dir + '/*.png')
        # clean_imgs.sort()
        self.ISP = ISP_implement.ISP(curve_path='./utils/syn')
        # self.blur, self.pad = get_gaussian_kernel(kernel_size=5, sigma=1)

        # self.clean_fns = [None] * sample_num
        # for i in range(sample_num):
        #     self.clean_fns[i] = []

        # for ind, folder in enumerate(folders):
        #     # clean_imgs = glob.glob(folder + '/*GT_SRGB*')
        #     clean_imgs = glob.glob(folder + '/*GT_SRGB*')
        #     clean_imgs.sort()
        #
        #
        #     for ind, clean_img in enumerate(clean_imgs):
        #         self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):

        # start_time = time.time()
        clean_fn = (self.clean_fns[idx])

        clean_img = read_img(clean_fn)

        # noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))
        # sigma_img = read_img(clean_fn.replace('GT_SRGB', 'SIGMA_SRGB'), True) / 15.	# inverse scaling

        if self.patch_size > 0:
            [clean_img] = get_patch([clean_img], self.patch_size)
            [clean_img] = random_augmentation(clean_img)

        # clean_img = torch.from_numpy(np.ascontiguousarray(np.transpose(clean_img, (2, 0, 1))))
        # clean_img = F.pad(clean_img.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        # clean_img = self.blur(clean_img).squeeze(0)
        # clean_img = np.transpose(clean_img.numpy(), (1, 2, 0))
        clean_img, noise_img, sigma_img = self.ISP.noise_generate_srgb(clean_img)
        # [clean_img, noise_img, sigma_img] = get_patch([clean_img, noise_img, sigma_img], self.patch_size)


        # print("读取数据Time: {:4.4f}s".format(time.time() - start_time))
        # return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(np.expand_dims(sigma_img, 2)), np.ones(
        #     (1, self.patch_size, self.patch_size)).astype("float32")
        return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(np.expand_dims(sigma_img, 2)),  np.ones(
            (1, self.patch_size, self.patch_size)).astype("float32")


class Syn_JPG_NEW(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size

        self.clean_fns = get_image_paths(root_dir, data_type='img')


        # folders = glob.glob(root_dir + '/*')
        # clean_imgs = glob.glob(root_dir + '/*.png')
        # clean_imgs.sort()
        self.ISP = ISP_implement.ISP(curve_path='./utils/syn')
        # self.blur, self.pad = get_gaussian_kernel(kernel_size=5, sigma=1)

        # self.clean_fns = [None] * sample_num
        # for i in range(sample_num):
        #     self.clean_fns[i] = []

        # for ind, folder in enumerate(folders):
        #     # clean_imgs = glob.glob(folder + '/*GT_SRGB*')
        #     clean_imgs = glob.glob(folder + '/*GT_SRGB*')
        #     clean_imgs.sort()
        #
        #
        #     for ind, clean_img in enumerate(clean_imgs):
        #         self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):

        # start_time = time.time()
        clean_fn = (self.clean_fns[idx])

        clean_img = read_img(clean_fn)

        # noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))
        # sigma_img = read_img(clean_fn.replace('GT_SRGB', 'SIGMA_SRGB'), True) / 15.	# inverse scaling

        if self.patch_size > 0:
            [clean_img] = get_patch([clean_img], self.patch_size)
            [clean_img] = random_augmentation(clean_img)

        # clean_img = torch.from_numpy(np.ascontiguousarray(np.transpose(clean_img, (2, 0, 1))))
        # clean_img = F.pad(clean_img.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        # clean_img = self.blur(clean_img).squeeze(0)
        # clean_img = np.transpose(clean_img.numpy(), (1, 2, 0))
        clean_img, noise_img, sigma_img = self.ISP.noise_generate_srgb(clean_img)
        # [clean_img, noise_img, sigma_img] = get_patch([clean_img, noise_img, sigma_img], self.patch_size)

        quality = np.random.uniform(60, 100)
        encode_q = [int(cv2.IMWRITE_JPEG_CHROMA_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', noise_img*255, encode_q)
        noise_img = cv2.imdecode(encimg, 1)/255.
        #print(noise_img)

        # print("读取数据Time: {:4.4f}s".format(time.time() - start_time))
        # return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(np.expand_dims(sigma_img, 2)), np.ones(
        #     (1, self.patch_size, self.patch_size)).astype("float32")
        return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(np.expand_dims(sigma_img, 2)),  np.ones(
            (1, self.patch_size, self.patch_size)).astype("float32")

class Syn_JPEG(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size

        self.clean_fns = get_image_paths(os.path.join(root_dir, 'GT_SRGB'), data_type='img')


        # folders = glob.glob(root_dir + '/*')
        # clean_imgs = glob.glob(root_dir + '/*.png')
        # clean_imgs.sort()
        # self.ISP = ISP_implement.ISP(curve_path='./utils/syn')
        # self.blur, self.pad = get_gaussian_kernel(kernel_size=5, sigma=1)

        # self.clean_fns = [None] * sample_num
        # for i in range(sample_num):
        #     self.clean_fns[i] = []

        # for ind, folder in enumerate(folders):
        #     # clean_imgs = glob.glob(folder + '/*GT_SRGB*')
        #     clean_imgs = glob.glob(folder + '/*GT_SRGB*')
        #     clean_imgs.sort()
        #
        #
        #     for ind, clean_img in enumerate(clean_imgs):
        #         self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):

        # start_time = time.time()
        clean_fn = (self.clean_fns[idx])

        clean_img = read_img(clean_fn)
        noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB').replace('.png', '.jpg'))
        sigma_img = loadmat(clean_fn.replace('GT_SRGB', 'SIGMA_SRGB').replace('.png', '.mat'))['A']
        sigma_img = np.expand_dims(sigma_img, 2)

        if self.patch_size > 0:
            [clean_img, noise_img, sigma_img] = get_patch([clean_img, noise_img, sigma_img], self.patch_size)

            # [clean_img] = random_augmentation(clean_img)

        # clean_img = torch.from_numpy(np.ascontiguousarray(np.transpose(clean_img, (2, 0, 1))))
        # clean_img = F.pad(clean_img.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        # clean_img = self.blur(clean_img).squeeze(0)
        # clean_img = np.transpose(clean_img.numpy(), (1, 2, 0))
        # clean_img, noise_img, sigma_img = self.ISP.noise_generate_srgb(clean_img)
        # [clean_img, noise_img, sigma_img] = get_patch([clean_img, noise_img, sigma_img], self.patch_size)


        # print("读取数据Time: {:4.4f}s".format(time.time() - start_time))
        # return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(np.expand_dims(sigma_img, 2)), np.ones(
        #     (1, self.patch_size, self.patch_size)).astype("float32")
        return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(sigma_img),  np.ones(
            (1, self.patch_size, self.patch_size)).astype("float32")

class Syn_JPEG_PAMI(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size

        self.clean_fns = get_image_paths(os.path.join(root_dir), data_type='img')


        # folders = glob.glob(root_dir + '/*')
        # clean_imgs = glob.glob(root_dir + '/*.png')
        # clean_imgs.sort()
        self.ISP = ISP_implement.ISP(curve_path='./utils/syn')
        # self.blur, self.pad = get_gaussian_kernel(kernel_size=5, sigma=1)

        # self.clean_fns = [None] * sample_num
        # for i in range(sample_num):
        #     self.clean_fns[i] = []

        # for ind, folder in enumerate(folders):
        #     # clean_imgs = glob.glob(folder + '/*GT_SRGB*')
        #     clean_imgs = glob.glob(folder + '/*GT_SRGB*')
        #     clean_imgs.sort()
        #
        #
        #     for ind, clean_img in enumerate(clean_imgs):
        #         self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):

        # start_time = time.time()
        clean_fn = (self.clean_fns[idx])

        clean_img = read_img(clean_fn)
        # noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB').replace('.png', '.jpg'))
        # sigma_img = loadmat(clean_fn.replace('GT_SRGB', 'SIGMA_SRGB').replace('.png', '.mat'))['A']
        # sigma_img = np.expand_dims(sigma_img, 2)

        if self.patch_size > 0:
            [clean_img] = get_patch([clean_img], self.patch_size)

            # [clean_img] = random_augmentation(clean_img)

        # clean_img = torch.from_numpy(np.ascontiguousarray(np.transpose(clean_img, (2, 0, 1))))
        # clean_img = F.pad(clean_img.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        # clean_img = self.blur(clean_img).squeeze(0)
        # clean_img = np.transpose(clean_img.numpy(), (1, 2, 0))
        clean_img, noise_img, sigma_img = self.ISP.noise_generate_srgb(clean_img)
        quality = np.random.uniform(60, 100)
        save_name = clean_fn.replace('target', 'NOISY_JPEG').replace('.png', '.jpg')
        cv2.imwrite(save_name, noise_img[:, :, ::-1]*255, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        noise_img = read_img(save_name)
        # [clean_img, noise_img, sigma_img] = get_patch([clean_img, noise_img, sigma_img], self.patch_size)


        # print("读取数据Time: {:4.4f}s".format(time.time() - start_time))
        # return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(np.expand_dims(sigma_img, 2)), np.ones(
        #     (1, self.patch_size, self.patch_size)).astype("float32")
        return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(np.expand_dims(sigma_img, 2)),  np.ones(
            (1, self.patch_size, self.patch_size)).astype("float32")


class BaseDataSetH5(Dataset):
    def __init__(self, h5_path, length=None):
        '''
        Args:
            h5_path (str): path of the hdf5 file
            length (int): length of Datasets
        '''
        super(BaseDataSetH5, self).__init__()
        self.h5_path = h5_path
        self.length = length
        with h5py.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)

    def __len__(self):
        if self.length == None:
            return self.num_images
        else:
            return self.length

    def crop_patch(self, imgs_sets):
        H, W, C2 = imgs_sets.shape
        C = int(C2 / 2)
        ind_H = random.randint(0, H - self.pch_size)
        ind_W = random.randint(0, W - self.pch_size)
        im_noisy = np.array(imgs_sets[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :C])
        im_gt = np.array(imgs_sets[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, C:])
        return im_gt, im_noisy

# 训练集控制合成集比例函数
# 用法：random.random() > self.ratio。修改self.ratio = 0.1处即可。
class Tra(BaseDataSetH5):
    def __init__(self, root_dir, h5_file, length=None, pch_size=128, radius=5, eps2=1e-6, noise_estimate=True):
        super(Tra, self).__init__(h5_file, length)

        self.pch_size = pch_size
        self.clean_fns = get_image_paths(root_dir, data_type='img')
        self.ISP = ISP_implement.ISP(curve_path='./utils/syn')
        self.ratio = 0.1

    # def __len__(self):
    #     if self.length == None:
    #         return len(self.clean_fns) + self.num_images
    #     else:
    #         return len(self.clean_fns) + self.length
    def __len__(self):
        if self.length == None:
            return math.ceil(self.num_images/(1-self.ratio))
        else:
            return math.ceil(self.length/(1-self.ratio))

    def __getitem__(self, index):
        if random.random() > self.ratio:#执行SIDD数据集读取
            num_images = self.num_images
            ind_im = random.randint(0, num_images - 1)

            with h5py.File(self.h5_path, 'r') as h5_file:
                imgs_sets = h5_file[self.keys[ind_im]]
                im_gt, im_noisy = self.crop_patch(imgs_sets)
            im_gt = img_as_float(im_gt)
            im_noisy = img_as_float(im_noisy)

            # data augmentation
            im_gt, im_noisy = random_augmentation(im_gt, im_noisy)
            return hwc_to_chw(im_noisy), hwc_to_chw(im_gt), np.zeros((1, self.pch_size, self.pch_size)).astype(
                "float32"), np.zeros(
                (1, self.pch_size, self.pch_size)).astype("float32")
        else:#执行数据集合成
            # start_time = time.time()
            idx = random.randint(0, len(self.clean_fns) - 1)
            clean_fn = (self.clean_fns[idx])

            clean_img = read_img(clean_fn)


            if self.pch_size > 0:
                [clean_img] = get_patch([clean_img], self.pch_size)
                [clean_img] = random_augmentation(clean_img)

            clean_img, noise_img, sigma_img = self.ISP.noise_generate_srgb(clean_img)
            return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(np.expand_dims(sigma_img, 2)), np.ones(
                (1, self.pch_size, self.pch_size)).astype("float32")

class BenchmarkTest(BaseDataSetH5):
    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2 / 2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy,im_gt

class BenchmarkTest(BaseDataSetH5):
    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2 / 2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy,im_gt

import torch.nn as nn
import math
def get_gaussian_kernel(kernel_size=21, sigma=5, channels=3):
    #if not kernel_size: kernel_size = int(2*np.ceil(2*sigma)+1)
    #print("Kernel is: ",kernel_size)
    #print("Sigma is: ",sigma)
    padding = kernel_size//2
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter, padding

class BenchmarkTrain(BaseDataSetH5):
    def __init__(self, h5_file, length=None, pch_size=64, radius=5, eps2=1e-6, noise_estimate=True):
        super(BenchmarkTrain, self).__init__(h5_file, length)
        # self.win = 2*radius + 1
        # self.sigma_spatial = radius
        # self.noise_estimate = noise_estimate
        # self.eps2 = eps2

        self.pch_size = pch_size

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        with h5py.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[ind_im]]
            im_gt, im_noisy = self.crop_patch(imgs_sets)
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)
        # im_gt, im_noisy = random_augmentation_pami(im_gt, im_noisy)
        return hwc_to_chw(im_noisy), hwc_to_chw(im_gt), np.zeros((1, self.pch_size, self.pch_size)).astype("float32"), np.zeros(
            (1, self.pch_size, self.pch_size)).astype("float32")


        # im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        # im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))
        #
        #
        # return im_noisy, im_gt,


class Dataset_h5_real(Dataset):

    def __init__(self, src_path, length = None,patch_size=128,  gray=False, train=True):

        self.path = src_path
        h5f = h5py.File(self.path, 'r')
        self.length = length
        self.keys = list(h5f.keys())
        self.num_images = len(self.keys)
        if train:
            random.shuffle(self.keys)
        h5f.close()

        self.patch_size = patch_size
        self.train = train
        self.gray = gray

    def __getitem__(self, index):

        num_images = self.num_images
        ind_im = random.randint(0, num_images - 1)

        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]
        data = np.array(h5f[key]).reshape(h5f[key].shape)
        h5f.close()

        if self.train:
            (H, W, C) = data.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch = data[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            p = 0.5
            if random.random() > p: #RandomRot90
                patch = patch.transpose(1, 0, 2)
            if random.random() > p: #RandomHorizontalFlip
                patch = patch[:, ::-1, :]
            if random.random() > p: #RandomVerticalFlip
                patch = patch[::-1, :, :]
        else:
            patch = data

        patch = np.clip(patch.astype(np.float32)/255.0, 0.0, 1.0)
        if self.gray:
            noisy = np.expand_dims(patch[:, :, 0], -1)
            clean = np.expand_dims(patch[:, :, 1], -1)
        else:
            noisy = patch[:, :, 0:3]
            clean = patch[:, :, 3:6]

        # noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(noisy, (2, 0, 1)))).float()
        # clean = torch.from_numpy(np.ascontiguousarray(np.transpose(clean, (2, 0, 1)))).float()

        return hwc_to_chw(noisy), hwc_to_chw(clean),np.zeros((1, self.patch_size, self.patch_size)).astype("float32"), np.zeros(
            (1, self.patch_size, self.patch_size)).astype("float32")

    def __len__(self):


        if self.length == None:
            return self.num_images
        else:
            return self.length
        # return len(self.keys)