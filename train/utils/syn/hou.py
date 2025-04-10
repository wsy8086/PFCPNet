import os
import cv2

# 指定源文件夹和目标文件夹路径
source_folder = r'F:\Pan\test_2021\10.29\DIV2K_color_HR_NOISY_JPEG_copy'  # 替换为你的源文件夹路径
target_folder = r'F:\Pan\test_2021\10.29\noise'  # 替换为你的目标文件夹路径

# 创建目标文件夹（如果不存在）
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.jpg'):
        # 构造完整的文件路径
        jpg_path = os.path.join(source_folder, filename)

        # 读取 JPG 图片
        image = cv2.imread(jpg_path)

        # 构造新的 PNG 文件名
        png_filename = filename[:-4] + '.png'  # 去掉 .jpg 后缀，改为 .png
        png_path = os.path.join(target_folder, png_filename)

        # 保存为 PNG 格式
        cv2.imwrite(png_path, image)

print("转换完成！")
