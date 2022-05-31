from torch.utils.data import Dataset
import functions
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class RecolaDataset(Dataset):
    def __init__(self, names_txt_dir, audio_path, txt_path, label_path, img_shape=(3, 90, 90)) -> None:
        self.imgs_dir_all, self.audio_all, self.arousal_all, self.valence_all = functions.read_all_data(
            names_txt_dir, audio_path, txt_path, label_path
        )
        self.transforms = A.Compose([
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),   # 标准化，归一化
            ToTensorV2()        # 转为tensor
        ])
        self.img_shape = img_shape

    def __len__(self):
        return len(self.imgs_dir_all)

    def __getitem__(self, index):
        imgs_dirs = self.imgs_dir_all[index]
        img_sq = self.__read_pics__(imgs_dirs)
        audio_sq = torch.tensor(self.audio_all[index])
        arousal = self.arousal_all[index]
        valence = self.valence_all[index]
        return img_sq, audio_sq, arousal, valence
    
    def __read_pics__(self, imgs_dirs):     # 对一条图像序列进行处理（batchsize=1）
        l = len(imgs_dirs)      # 序列长度
        imgs = torch.zeros((l, *self.img_shape))
        for idx, img_dir in enumerate(imgs_dirs):   # 解析图像文件
            img = np.array(Image.open(img_dir).convert("RGB").resize((90, 90)))
            augmentations = self.transforms(image=img)      # 图像增强
            img = augmentations["image"]
            imgs[idx] = img
        
        return imgs

if __name__ == "__main__":
    names_txt_dir = "/home/MyServer/data/RECOLA/names.txt"
    frames_path = "/home/MyServer/data/RECOLA/video_frames"
    audio_path = "/home/MyServer/data/RECOLA/recordings_audio/recordings_audio"
    txt_path = "/home/MyServer/data/RECOLA/txt_data"
    label_path = "/home/MyServer/data/RECOLA/labels/gs_1"
    dataset = RecolaDataset(names_txt_dir, audio_path, txt_path, label_path)

    loader = DataLoader(dataset, 7, shuffle=True)

    for data in loader:
        img_sq, audio_sq, arousal, valence = data
        img_sq = img_sq.permute(1, 0, 2, 3, 4)      # 维度转换
        audio_sq = audio_sq.permute(1, 0, 2)
        print(img_sq.shape, audio_sq.shape)

