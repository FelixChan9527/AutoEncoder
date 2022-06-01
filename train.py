from RecolaData import *
import torch
from torch.utils.data import DataLoader
from model import *
from tqdm import tqdm
from model import *
from loss import *
from torch.utils.tensorboard import SummaryWriter
import sys
import requests
import torchvision

def send_msg(msg):      # 用于发送信息给微信
    """
    ### 该函数用于发送信息给微信，以提醒用户程序跑到什么阶段，出现什么问题
    ### 输入：需要发送的信息字符串
    """
    app_token = 'AT_pXOBwvbqSPML2B9MlPt1HO8LG7QvEOoD'   # 本处改成自己的应用 APP_TOKEN
    uid_myself = 'UID_JhqgW6oGSyxgFCsiRL7xLH5jD2EA'  # 本处改成自己的 UID

    """利用 wxpusher 的 web api 发送 json 数据包，实现微信信息的发送"""
    webapi = 'http://wxpusher.zjiecode.com/api/send/message'
    data = {
        "appToken":app_token,
        "content":msg,
        "summary":msg[:99], # 该参数可选，默认为 msg 的前10个字符
        "contentType":1,
        "uids":[ uid_myself, ],
        }
    result = requests.post(url=webapi,json=data)

names_txt_dir = "/home/MyServer/data/RECOLA/names.txt"
frames_path = "/home/MyServer/data/RECOLA/video_frames"
audio_path = "/home/MyServer/data/RECOLA/recordings_audio/recordings_audio"
txt_path = "/home/MyServer/data/RECOLA/txt_data"
label_path = "/home/MyServer/data/RECOLA/labels/gs_1"
epochs = 200
lr = 0.0001
batchsize = 128
model_path = "/home/MyServer/model_file/RECOLA"
writer = SummaryWriter("/home/MyServer/MyCode/logs")      # 指定过程数据保存路径

def train_fn(auto_encoder, dataset, epochs, checkpoint):
    loss_fn = AutoEncoderLoss()
    loss_fn.to("cuda")
    optim = torch.optim.Adam(auto_encoder.parameters(), lr, weight_decay=0.001)
    auto_encoder.load_state_dict(checkpoint['model_state_dict'])
    # optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    count = 0
    for epoch in range(epoch+1, epochs):
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

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=loss.item())
            writer.add_scalar("the loss", loss, count)
            count += 1

            # view_pic = torchvision.transforms.ToPILImage()(img_pr[0, 0] * 255)
            # view_pic2 = torchvision.transforms.ToPILImage()(img_sq[0, 0] * 255)
            # view_pic.save("/home/MyServer/pics/"+str(idx)+"r.jpg")
            # view_pic2.save("/home/MyServer/pics/"+str(idx)+"n.jpg")

        torch.save({"epoch": epoch, 'model_state_dict': auto_encoder.state_dict(), 
        'optimizer_state_dict': optim.state_dict(), 'loss': loss}, model_path+"/RECOLA_{}.pth.tar".format(epoch))
        # torch.save(auto_encoder.state_dict(), model_path+"/RECOLA_{}.pth".format(epoch))
        send_msg("完成一次 loss={:.5f}".format(loss))

    writer.close()


if __name__ == "__main__":
    auto_encoder = AutoEncoderNet([3, 96, 96], [1, 640], 32, 40).to("cuda")
    dataset = RecolaDataset(names_txt_dir, audio_path, txt_path, label_path)
    # auto_encoder.load_state_dict(torch.load("/home/MyServer/model_file/RECOLA/RECOLA_2.pth"))
    
    checkpoint = torch.load("/home/MyServer/model_file/RECOLA/RECOLA_19.pth.tar")
    train_fn(auto_encoder, dataset, epochs, checkpoint)
    