import os
if __name__ == '__main__':
    path = "D:\Aaysydataset\DIYDATASET\VOC2007\VOCdevkit\MSCOCO_txt"
    filelist = os.listdir(path)
    for filename in filelist:
        if "_"in filename:
            continue
        else:
            os.remove(os.path.join(path, filename))
