import os


def count_yolo_data(path, classList):
    '''统计yolo数据集目录格式下各个类别的数量
    Parameters
    ----------
    path : str
        yolo数据集路径
    classList: list
        数据集类别名组成的列表， 名字顺序与Yolo标签的顺序一致
    Returns : none
    -------
    '''
    train_path = path + "\labels\ train"
    valid_path = path + "\labels\ valid"
    train_labels = os.listdir(train_path)
    valid_labels = os.listdir(valid_path)
    train_dir = {}
    train_count = [0 for _ in range(len(classList))]
    for file in train_labels:
        # print(train_path + "/" + file)
        with open(train_path + "/" + file) as o:
            for line in o:
                index = int(line.split()[0])
                train_count[index] += 1
    for i in range(len(classList)):
        train_dir[classList[i]] = train_count[i]
    print(f"train_dir:{train_dir}")
    valid_dir = {}
    valid_count = [0 for _ in range(len(classList))]
    for file in valid_labels:
        # print(valid_path + "/" + file)
        with open(valid_path + "/" + file) as o:
            for line in o:
                index = int(line.split()[0])
                valid_count[index] += 1
    for i in range(len(classList)):
        valid_dir[classList[i]] = valid_count[i]
    print(f"valid_dir:{valid_dir}")


data_path = "D:\Aaysydataset\YOLOv5my\yolov5-6.1\Soybean_weed_Detcetion"
classList = ['soybean','machixian','matang','tiexiancai','ciercai','li','dawanhua']
count_yolo_data(data_path, classList)