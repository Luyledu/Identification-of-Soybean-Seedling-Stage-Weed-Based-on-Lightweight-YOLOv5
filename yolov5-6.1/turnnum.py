import os

path = r"D:\Aaysydataset\YOLOv5my\yolov5-6.1\Apply dataset\detect True\Image"
filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
count = 1
for file in filelist:
    Olddir = os.path.join(path, file)  # 原来的文件路径
    if os.path.isdir(Olddir):  # 如果是文件夹则跳过
        continue
    filename = os.path.splitext(file)[0]  # 文件名
    filetype = os.path.splitext(file)[1]  # 文件扩展名
    print(filename)
    Newdir = os.path.join(path, str(count).zfill(3) + filetype)  # 用字符串函数zfill 以0补全所需位数
    os.rename(Olddir, Newdir)  # 重命名
    count += 1
# os.path.splitext(“文件路径”)    分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作

print('END')
count = str(count)
print("共有" + count + "张图片尺寸被修改")