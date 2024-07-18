import os  # 用于处理文件路径、创建目录等操作
import random  # 用于生成随机数种子、打乱列表等操作
import shutil  # 用于生成随机数种子、打乱列表等操作

# 设置随机数种子
random.seed(123)

# 定义文件夹路径
image_dir = 'D:\Aaysydataset\YOLOv5my\yolov5-6.1\DIYDATASET\VOC2007\VOCdevkit\AugpotoJPEGimages\JPEGImages'  # 原始图像所在的子目录
label_dir = 'D:\Aaysydataset\YOLOv5my\yolov5-6.1\DIYDATASET\VOC2007\VOCdevkit\AugpotoAnnoto\AugMSCOCO_txt'  # 原始标签所在的子目录
output_dir = 'D:\Aaysydataset\YOLOv5my\yolov5-6.1\DIYDATASET\VOC2007\VOCdevkit\AUGSoybean_Weed'  # 处理后的数据集输出目录

# 定义训练集、验证集和测试集比例
train_ratio = 0.6  # 训练集比例
valid_ratio = 0.2  # 验证集比例
test_ratio = 0.2  # 测试集比例

# 获取所有图像文件和标签文件的文件名（不包括文件扩展名）
image_filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]  # 提取所有图像文件的文件名列表
label_filenames = [os.path.splitext(f)[0] for f in os.listdir(label_dir)]  # 提取所有标签文件的文件名列表

# 随机打乱文件名列表
random.shuffle(image_filenames)  # 打乱图像文件的文件名列表

# 计算训练集、验证集和测试集的数量
total_count = len(image_filenames)  # 总文件数
train_count = int(total_count * train_ratio)  # 训练集文件数
valid_count = int(total_count * valid_ratio)  # 验证集文件数
test_count = total_count - train_count - valid_count  # 测试集文件数

# 定义输出文件夹路径
train_image_dir = os.path.join(output_dir, 'images', 'train')  # 训练集图像输出目录
train_label_dir = os.path.join(output_dir, 'labels', 'train')  # 训练集标签输出目录
valid_image_dir = os.path.join(output_dir, 'images', 'valid')  # 验证集图像输出目录
valid_label_dir = os.path.join(output_dir, 'labels', 'valid')  # 验证集标签输出目录
test_image_dir = os.path.join(output_dir, 'images', 'test')  # 测试集图像输出目录
test_label_dir = os.path.join(output_dir, 'labels', 'test')  # 测试集标签输出目录

# 创建输出文件夹
os.makedirs(train_image_dir, exist_ok=True)  # 创建训练集图像输出目录
os.makedirs(train_label_dir, exist_ok=True)  # 创建训练集标签输出目录
os.makedirs(valid_image_dir, exist_ok=True)  # 创建验证集图像输出目录
os.makedirs(valid_label_dir, exist_ok=True)  # 创建验证集标签输出目录
os.makedirs(test_image_dir, exist_ok=True)  # 创建测试集图像输出目录
os.makedirs(test_label_dir, exist_ok=True)  # 创建测试集标签输出目录

# 将图像和标签文件划分到不同的数据集中
for i, filename in enumerate(image_filenames):
    # 如果文件数量小于训练数据集大小，则将文件复制到训练数据集目录中
    if i < train_count:
        output_image_dir = train_image_dir
        output_label_dir = train_label_dir
    # 如果文件数量小于训练数据集大小+验证数据集大小，则将文件复制到验证数据集目录中
    elif i < train_count + valid_count:
        output_image_dir = valid_image_dir
        output_label_dir = valid_label_dir
    # 否则，将文件复制到测试数据集目录中
    else:
        output_image_dir = test_image_dir
        output_label_dir = test_label_dir

    # 复制图像文件
    src_image_path = os.path.join(image_dir, filename + '.jpg')  # 获取图像文件的源路径
    dst_image_path = os.path.join(output_image_dir, filename + '.jpg')  # 获取图像文件的目标路径
    shutil.copy(src_image_path, dst_image_path)  # 复制图像文件到目标路径

    # 复制标签文件
    src_label_path = os.path.join(label_dir, filename + '.txt')  # 获取标签文件的源路径
    dst_label_path = os.path.join(output_label_dir, filename + '.txt')  # 获取标签文件的目标路径
    shutil.copy(src_label_path, dst_label_path)  # 复制标签文件到目标路径