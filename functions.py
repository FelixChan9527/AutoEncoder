import cv2
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import sys

import requests

def send_msg(msg):

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

sys.path.append("/home/MyServer/MyCode/Retainface/Pytorch_Retinaface/")
import me

video_frames_path = "/home/MyServer/data/RECOLA/video_frames"
video_path = "/home/MyServer/data/RECOLA/recordings_video/recordings_video"
audio_frames_path = "/home/MyServer/data/RECOLA/audio_frames"
audio_path = "/home/MyServer/data/RECOLA/recordings_audio/recordings_audio"
filename = "/home/MyServer/data/RECOLA/recordings_audio/recordings_audio/dev_5.wav"
pic_path = "/home/MyServer/MyCode/AutoEncoder/test.jpg"
frame_path = "/home/MyServer/data/RECOLA/video_frames/train_3/7_280.jpg"
xml_path = "/usr/local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"

def extract_imgs(video_path, video_frames_path, model_path):
    """
    ### 这个函数用于视频抽帧
    """
    device = torch.device("cuda")
    net = me.model_config(device, model_path)

    videos_names = os.listdir(video_path)
    for name in videos_names:
        full_name = os.path.join(video_path, name)
        save_path = os.path.join(video_frames_path, name[:-4])
        
        video = cv2.VideoCapture(full_name)
        frame_exit = video.isOpened()
        frame_num = 0
        frame_name = 0
        acc = 100

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        frame_buff = 0
        while frame_exit:
            frame_exit, frame = video.read()            # 获取下一帧
            
            if frame_exit:
                frame, acc = me.pic_predict(device, net, frame)
                time = video.get(cv2.CAP_PROP_POS_MSEC)     # 获取当前时间戳

                ex = (time // 40) -  frame_num  # 正常来说，ex应该为0
                # 执行以下循环，说明发生了跳帧
                for i in range(int(ex)):
                    if acc <= 70:
                        frame_name = str(frame_num)+"_"+str(int(time-i*40)) + "_ac_" + "ex_" + ".jpg"
                    else:
                        frame_name = str(frame_num)+"_"+str(int(time-i*40)) + "_ex_" + ".jpg"
                    frame_path = os.path.join(save_path, frame_name)
                    frame_num += 1
                    cv2.imwrite(frame_path, frame_buff)
                    # print(frame_path)
                frame_buff = frame  # 保留本次的图像，供下一次如果掉帧时使用

                if acc <= 70:
                    frame_name = str(frame_num)+"_"+str(int(time)) + "_ac_" + ".jpg"
                else:
                    frame_name = str(frame_num)+"_"+str(int(time)) + ".jpg"
                frame_path = os.path.join(save_path, frame_name)

                cv2.imwrite(frame_path, frame)
                frame_num += 1
            
            else:
                send_msg("完成一次")
                break
            

def extract_audio(audio_path, audio_frames_path):
    pass

def torchaudio_test():
    waveform, sample_rate = torchaudio.load(filename)   # 加载音频
    # 音频重采样，将音频变为16kHZ，最终使得一帧为0.04s，每帧为长度640的时间序列
    transformed = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    transformed = (transformed - transformed.mean()) / transformed.std()
    

if __name__ == "__main__":
    model_path = "/home/MyServer/model_file/RetinaFace/Retinaface_model_v2/Resnet50_Final.pth"
    extract_imgs(video_path, video_frames_path, model_path)
    # img_path = "/home/MyServer/data/RECOLA/video_frames/train_3/7074_282960.jpg"
    # save_path = "/home/MyServer/MyCode/AutoEncoder/"
    # device = torch.device("cuda")
    # net = me.model_config(device, model_path)
    # frame = cv2.imread(img_path)
    # frame, acc = me.pic_predict(device, net, frame)
    # print(acc)
    # cv2.imwrite(save_path+"hh.jpg", frame)

    # frame = cv2.imread(frame_path)
    # frame = extract_face(frame)
    # cv2.imwrite(pic_path, frame)
    # torchaudio_test()
    # path = "/home/MyServer/data/RECOLA/recordings_video/recordings_video/dev_5.mp4"
    # cap = cv2.VideoCapture(path)
    # print(cap.get(7)/cap.get(5)/60)

