import numpy as np
import cv2

BINARY_EXTENSIONS = ['.npy']
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count


def read_img(filename, if_gray=False):
	if if_gray:
		img = cv2.imread(filename, 0)
		img = np.expand_dims(img, 2) / 255.0
	else:
		img = cv2.imread(filename)
		img = img[:,:,::-1] / 255.0

	img = np.array(img).astype('float32')

	return img


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).astype('float32')


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).astype('float32')


def data_augmentation(image, mode):
	'''
	Performs data augmentation of the input image
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


def inverse_data_augmentation(image, mode):
	'''
	Performs inverse data augmentation of the input image
	'''
	if mode == 0:
		# original
		out = image
	elif mode == 1:
		out = np.flipud(image)
	elif mode == 2:
		out = np.rot90(image, axes=(1,0))
	elif mode == 3:
		out = np.flipud(image)
		out = np.rot90(out, axes=(1,0))
	elif mode == 4:
		out = np.rot90(image, k=2, axes=(1,0))
	elif mode == 5:
		out = np.flipud(image)
		out = np.rot90(out, k=2, axes=(1,0))
	elif mode == 6:
		out = np.rot90(image, k=3, axes=(1,0))
	elif mode == 7:
		# rotate 270 degree and flip
		out = np.flipud(image)
		out = np.rot90(out, k=3, axes=(1,0))
	else:
		raise Exception('Invalid choice of image transformation')

	return out

import os
def get_image_paths(dataroot,data_type='npy'):
    paths = None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        elif data_type == 'npy':
            paths = sorted(_get_paths_from_binary(dataroot))
        else:
            raise NotImplementedError("[Error] Data_type [%s] is not recognized." % data_type)
    return paths

def _get_paths_from_binary(path_src,total = 9):
    assert os.path.isdir(path_src), '[Error] [%s] is not a valid directory' % path_src
    files = []
    for i in range(0,1):
        path =os.path.join(path_src)
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if is_binary_file(fname):
                    binary_path = os.path.join(dirpath, fname)
                    files.append(binary_path)
        assert files, '[%s] has no valid binary file' % path
    return files
def is_binary_file(filename):
    return any(filename.endswith(extension) for extension in BINARY_EXTENSIONS)

def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '[%s] has no valid image file' % path
    return images

import torch
import torch.nn as nn
def dwt_init(x):
    in_batch, in_channel, in_height, in_width = x.size()

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2



    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4


    return h
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)