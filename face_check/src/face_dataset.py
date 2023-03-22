import os
import cv2
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader

path_name = "../train_face_image"

# 读取图片数据
# image = []
# for dir_item in os.listdir(path_name):
#     if dir_item is None:
#         break
#     else:
#         image_path = os.path.relpath(os.path.join(path_name, dir_item))  # 只要它的相对路径
#         image = cv2.imread(image_path)


class face_dataset(Dataset):  # 需要继承Dataset类
    def __init__(self, img_dir, csv_dir, transform):
        super().__init__()
        # dir_item = os.listdir(img_dir)
        self.image_dir = img_dir
        self.img_labels = pd.read_csv(csv_dir,header=None)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir + self.img_labels.iloc[index, 0])  # 只要它的相对路径文件夹
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[index,1]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    dataset = face_dataset(img_dir="../train_face_image/",
                           csv_dir="../person_csv/person_train.csv",
                           transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, drop_last=True)
    for data in dataloader:
        imgs , target = data
        print(imgs.shape)
        # print(target)
# if __name__ == '__main__':
#     dataset = face_dataset(img_dir="../test_face_image/",
#                            csv_dir="../person_csv/person_test.csv",
#                            transform=torchvision.transforms.ToTensor())
#     dataloader = DataLoader(dataset, batch_size=32, drop_last=True)
#     for data in dataloader:
#         imgs , target = data
#         print(imgs.shape)
#         print(target)