from RecolaData import *
import torch
from torch.utils.data import DataLoader
from model import *
from tqdm import tqdm
from model import *
from loss import *
from torch.utils.tensorboard import SummaryWriter

names_txt_dir = "/home/MyServer/data/RECOLA/names.txt"
frames_path = "/home/MyServer/data/RECOLA/video_frames"
audio_path = "/home/MyServer/data/RECOLA/recordings_audio/recordings_audio"
txt_path = "/home/MyServer/data/RECOLA/txt_data"
label_path = "/home/MyServer/data/RECOLA/labels/gs_1"
epoch = 20
lr = 1e-4
batchsize = 128
model_path = "/home/MyServer/model_file/RECOLA"
writer = SummaryWriter("/home/MyServer/MyCode/logs")      # 指定过程数据保存路径

def train_fn(auto_encoder, dataset, epoch):
    loss_fn = AutoEncoderLoss()
    loss_fn.to("cuda")
    optim = torch.optim.Adam(auto_encoder.parameters(), lr)

    for i in range(epoch):
        loader = DataLoader(dataset, batchsize, shuffle=True)
        auto_encoder.train()

        loop = tqdm(loader)
        for idx, data in enumerate(loop):
            img_sq, audio_sq, arousal_gt, valence_gt = data
            img_sq = img_sq.to("cuda")
            audio_sq = audio_sq.to("cuda")
            arousal_gt = arousal_gt.to("cuda")
            valence_gt = valence_gt.to("cuda")
            img_sq = img_sq.permute(1, 0, 2, 3, 4)      # 维度转换
            audio_sq = audio_sq.permute(1, 0, 2)
            audio_sq = audio_sq.unsqueeze(2)

            img_pr, sig_pr, arousal_pr, valence_pr = auto_encoder(img_sq, audio_sq)
            loss = loss_fn(img_pr, img_sq, sig_pr, audio_sq, arousal_pr, arousal_gt, valence_pr, valence_gt)
            optim.zero_grad()   # 设定优化的方向
            loss.backward()     # 从最后一层损失反向计算到所有层的损失     
            optim.step()        # 更新权重

            loop.set_postfix(loss=loss.item())
            writer.add_scalar("the loss", loss, idx)
    
    writer.close()


if __name__ == "__main__":
    auto_encoder = AutoEncoderNet([3, 96, 96], [1, 640], 32, 40).to("cuda")
    dataset = RecolaDataset(names_txt_dir, audio_path, txt_path, label_path)

    train_fn(auto_encoder, dataset, epoch)
    