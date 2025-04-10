import os
import random, math
import torch
import numpy as np
import glob
import cv2
from tqdm import tqdm
from skimage import io
from scipy.io import loadmat, savemat
from utils.syn.ISP_implement import ISP


if __name__ == '__main__':
	isp = ISP()

	source_dir = '/mnt/data/617/wsy/Dataset/DIV2K_denoise/div2k_gt'
	target_dir1 = '/mnt/data/617/wsy/Dataset/DIV2K_denoise/div2k_gt'
	target_dir2 = '/mnt/data/617/wsy/Dataset/DIV2K_denoise/noise'
	target_dir3 = '/mnt/data/617/wsy/Dataset/DIV2K_denoise/sigma'

	# if not os.path.isdir(target_dir):
	# 	os.makedirs(target_dir)
	# if not os.path.isdir(target_dir+'/GT_SRGB'):
	# 	os.makedirs(os.path.join(target_dir,'/GT_SRGB'))

	if not os.path.isdir(target_dir1):
		os.makedirs(target_dir1)
	if not os.path.isdir(target_dir2):
		os.makedirs(target_dir2)
	if not os.path.isdir(target_dir3):
		os.makedirs(target_dir3)

	# if not os.path.isdir(os.path.join(target_dir,'/SIGMA_SRGB_')):
	# 	os.makedirs(os.path.join(target_dir,'/SIGMA_SRGB_'))

	fns = sorted(glob.glob(os.path.join(source_dir, '*.png')))

	patch_size = 256
	count = 0
	save_mat = np.empty(len(fns), dtype=object)

	for fn in tqdm(fns):
		img_rgb = cv2.imread(fn)[:, :, ::-1] / 255.0

		# H = img_rgb.shape[0]
		# W = img_rgb.shape[1]
		#
		# H_s = H // patch_size
		# W_s = W // patch_size
		#
		# patch_id = 0

		# for i in range(H_s):
		# 	for j in range(W_s):
		#
		# 		yy = i * patch_size
		# 		xx = j * patch_size
		#
		# 		patch_img_rgb = img_rgb[yy:yy+patch_size, xx:xx+patch_size, :]
		#
		# 		gt, noise, sigma = isp.noise_generate_srgb(patch_img_rgb)
		#
		# 		sigma = np.uint8(np.round(np.clip(sigma * 15 , 0, 1) * 255))	# store in uint8
		#
		# 		filename = os.path.basename(fn)
		# 		foldername = filename.split('.')[0]
		#
		# 		# out_folder = os.path.join(target_dir, foldername)
		#
		# 		# if not os.path.isdir(out_folder):
		# 		# 	os.makedirs(out_folder)
		# 		# os.path.join(target_dir, '/GT_SRGB')
		#
		#
		# 		io.imsave(os.path.join(target_dir1, foldername+'_%d.png' % (count)), gt)
		# 		io.imsave(os.path.join(target_dir2,   foldername + '_%d.png' % (count)), noise)
		# 		io.imsave(os.path.join(target_dir3,  foldername + '_%d.png' % (count)), sigma)
		# 		# io.imsave(os.path.join(out_folder, 'NOISY_SRGB_%d_%d.png' % (i, j)), noise)
		# 		# io.imsave(os.path.join(out_folder, 'SIGMA_SRGB_%d_%d.png' % (i, j)), sigma)
		count += 1
		# random.seed(1)
		# np.random.seed(1)
		gt, noise, sigma = isp.noise_generate_srgb(img_rgb)

		# sigma = np.uint8(np.round(np.clip(sigma * 15, 0, 1) * 255))  # store in uint8

		# if(count == 3): exit(1)

		filename = os.path.basename(fn)
		foldername = filename.split('.')[0]

		# out_folder = os.path.join(target_dir, foldername)

		# if not os.path.isdir(out_folder):
		# 	os.makedirs(out_folder)
		# os.path.join(target_dir, '/GT_SRGB')
		noise = noise*255
		# gt *= 255
		quality = int(np.random.uniform(60, 100))

		# io.imsave(os.path.join(target_dir1, '%d.png' % (count)), gt)
		# cv2.imwrite(os.path.join(target_dir1, '%d.jpg' % (count)), gt[:, :, ::-1])
		cv2.imwrite(os.path.join(target_dir2, filename), noise[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])


		# io.imsave(os.path.join(target_dir3, '%d.png' % (count)), sigma)
		# savemat(os.path.join(target_dir3, '%d.mat' % (count)), {"A":sigma})

