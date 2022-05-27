import cv2
import os
import torch
import torchaudio
import sys
import requests
import numpy as np

def send_msg(msg):      # 用于发送信息给微信

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
        frame_buff = 0

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        while frame_exit:
            frame_exit, frame = video.read()            # 获取下一帧
            
            if frame_exit:
                frame, acc = me.pic_predict(device, net, frame)
                time = video.get(cv2.CAP_PROP_POS_MSEC)     # 获取当前时间戳
                
                ex = (time // 40) -  frame_num  # 正常来说，ex应该为0
                # 执行以下循环，说明发生了跳帧
                for i in range(int(ex)):
                    if acc <= 70:       # 误差过大，就不要了
                        frame_num += 1
                        print("No face, skip!")
                    else:
                        frame_name = str(frame_num) + "_ex_" + ".jpg"
                        frame_path = os.path.join(save_path, frame_name)
                        frame_num += 1
                        cv2.imwrite(frame_path, frame_buff)
                        print(frame_name)
                frame_buff = frame  # 保留本次的图像，供下一次如果掉帧时使用

                if acc <= 70:
                    frame_num += 1
                    print("No face, skip!")
                else:
                    frame_name = str(frame_num) + ".jpg"
                    frame_path = os.path.join(save_path, frame_name)
                    print(frame_name)
                    try:
                        cv2.imwrite(frame_path, frame)
                        frame_num += 1
                    except cv2.error:       # 会出现这个奇怪的错误
                        send_msg("出现异常")
                        print("No face, skip!")
                        frame_num += 1
            else:
                send_msg("完成一次")
                break
    
    send_msg("人脸提取完成！")

def extract_audio(filename, Hz=16000):
    """
    ### 将一条音频数据加载为标准化后的数组
    """
    waveform, sample_rate = torchaudio.load(filename)   # 加载音频
    # 音频重采样，将音频变为16kHZ，最终使得一帧为0.04s，每帧为长度640的时间序列
    audio = torchaudio.transforms.Resample(sample_rate, Hz)(waveform)
    audio = (audio - audio.mean()) / audio.std()    # 标准化
    return audio.squeeze(0)

def seperate(name, frames_file, sequen_lenth, txt_folder):
    """
    ### 一个用于根据得到的视频的帧来将其划分为特定时间点的序列数据集
    """  
    pic_files = np.array(os.listdir(frames_file))
    pic_index = np.array([int(pic_files[i].split("_")[0]) for i in range(len(pic_files))])
    sorted_index = np.argsort(pic_index)    # 获取图像对应时间顺序的索引
    sorted_index = sorted_index[0: len(sorted_index)-1]     # 去掉最后一帧（因为图像有可能会多出一帧）
    
    pic_index = pic_index[sorted_index]     # 按顺序排列的图像帧索引
    pic_files = pic_files[sorted_index]     # 按顺序排列的图像名
    pic_fullname = np.array([os.path.join(frames_file, pic_files[i]) for i in range(len(pic_files))])
    
    # 图片路径按顺序写入文本做保存
    write_txt_pic(pic_fullname, name, txt_folder)

    pics_len = len(pic_index)               # 获取图像长度

    # 生成用于划分有重叠的
    index_metric = [[i+j for j in range(sequen_lenth)] for i in range(pics_len-sequen_lenth+1)]
    file_sperated_int = np.array([[pic_index[i] for i in indexs] for indexs in index_metric])

    # 去除不是连续的
    index = []
    for i in range(len(file_sperated_int)):
        # 判断是不是等差数列，使用等差数列求和公式
        if sum(file_sperated_int[i]) == (max(file_sperated_int[i])+min(file_sperated_int[i]))*sequen_lenth/2:
            index.append(i)

    seuqence_index = file_sperated_int[index]

    return seuqence_index

def write_txt_pic(pic_fullname, name, txt_folder):
    txt_dir = os.path.join(txt_folder, name+".txt")
    txt_file = open(txt_dir,'w')
    txt_file.truncate(0)    # 对原文件内容直接清空

    for name in pic_fullname:
        txt_file.write(name+"\n")

def write_txt_index(seuqence_index, name, txt_folder):
    txt_dir = os.path.join(txt_folder, name+"_index.txt")
    txt_file = open(txt_dir,'w')    # 用于建立这个文件
    txt_file.truncate(0)    # 对原文件内容直接清空
    txt_file.close()

    # 保存索引数组，保存为整数
    np.savetxt(txt_dir, seuqence_index, fmt="%d")

def get_audio_sequence(seuqence_index, audio_path, audio_frame_len):
    # 对音频每audio_frame_len长度作为一帧与图像对应
    audio = extract_audio(audio_path).numpy()   # 必须要转化为numpy格式才好做转换
    audio_frame_num = len(audio) // audio_frame_len
    # 将音频进行截断分组
    audio = np.array(np.split(audio, audio_frame_num))  # 将音频平均分段
    audio = audio[seuqence_index]

    return audio

def get_pic_sequence(seuqence_index, image_dirs_txt):
    with open(image_dirs_txt, "r") as f:
        img_dirs = np.array([line.strip("\n") for line in f.readlines()])
        img_dirs = img_dirs[seuqence_index]
    
    f.close()

    return img_dirs

def get_seuqence_index(seuqence_index_txt):
    with open(seuqence_index_txt, "r") as f:
        seuqence_index = np.loadtxt(seuqence_index_txt, dtype=int)
        f.close()
    
    return seuqence_index

def seperqte_all(frames_files, sequen_lenth, txt_folder):
    names = os.listdir(frames_files)
    for name in names:
        frames_file = os.path.join(frames_files, name)
        seuqence_index = seperate(name, frames_file, sequen_lenth, txt_folder)
        write_txt_index(seuqence_index, name, txt_folder)

def name_txt(names_path, save_path):
    names = os.listdir(names_path)

    txt_dir = os.path.join(save_path, "names.txt")
    txt_file = open(txt_dir,'w')
    txt_file.truncate(0)    # 对原文件内容直接清空

    for name in names:
        txt_file.write(name+"\n")
    
    txt_file.close()


if __name__ == "__main__":

    name_txt("/home/MyServer/data/RECOLA/video_frames", "/home/MyServer/data/RECOLA")