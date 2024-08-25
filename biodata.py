import torch
import os
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
import torch.utils.data as data

#数据加载器，把文件里的图片以及他的标签按照顺序写到一个npy文件里。

def find_classes(dir):
    # 得到指定目录下的所有文件，并将其名字和指定目录的路径合并
    # 以数组的形式存在classes中
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    # 使用sort()进行简单的排序
    classes.sort()
    # 将其保存的路径排序后简单地映射到 0 ~ [ len(classes)-1] 的数字上
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # 返回存放路径的数组和存放其映射后的序号的数组
    return classes, class_to_idx

class BioData(torch.utils.data.Dataset):
    def __init__(self, root, dataname,
                 transform=None):
        self.root = root
        self.transform = transform
        #self.target_transform = target_transform
        self.dataname = dataname
        self.train_data_with_label= []

        # now load the picked numpy arrays
        self.train_data = []
        self.train_labels = []
        file_root = self.root +self.dataname +'/'
        classes, class_to_idx = find_classes(file_root)
        data_len = 0
        for class_ in classes:
            file_folder = file_root+class_+'/'
            files = os.listdir(file_folder)
            class_idx = class_to_idx[class_]
            for file in files:
                image = Image.open(file_folder+file).convert('RGB')
                transform_ = transforms.Resize([227,227])
                image = transform_(image)
                image_arr = np.array(image) 
                self.train_data.append(image_arr)
                self.train_labels.append(class_idx)
                data_len += 1

        self.train_data = np.array(self.train_data)
        print(data_len)
        self.train_data = self.train_data.reshape((data_len, 3, 227, 227))
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.train_data[index], self.train_labels[index]
        

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

            return img, target, index #这一组图片它在所有的图片里是第几组

    def __len__(self):
            return len(self.train_data)
        
def get_biodata(dataset_name, batch_size, datafolder = '/root/autodl-tmp/'):
    """Get Office datasets loader."""
    pre_process = transforms.Compose([
            transforms.Resize([227,227]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((30,60)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    datafolder = datafolder
    dataset_ = BioData(datafolder, dataset_name, transform=pre_process)

    dataloader_ = torch.utils.data.DataLoader(
        dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader_, dataset_