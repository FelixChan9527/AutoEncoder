import cv2
import os
import torch
import torchaudio
import sys
import requests
import numpy as np
import pandas as pd
from PIL import Image
import time


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

sys.path.append("/home/MyServer/MyCode/Retainface/Pytorch_Retinaface/")
import me
    
def extract_imgs(video_path, video_frames_path, model_path):
    """
    ### 这个函数用于视频抽帧并将每帧图像进行人脸检测并截取
    ### 输入：视频路径、帧保存路径、人脸检测模型文件
    ### 人脸检测模型：RetinaFace
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
    ### 输入：音频路径、采样频率（可选）
    ### 输出：采样后的音频信息数组
    """
    waveform, sample_rate = torchaudio.load(filename)   # 加载音频
    # 音频重采样，将音频变为16kHZ，最终使得一帧为0.04s，每帧为长度640的时间序列
    audio = torchaudio.transforms.Resample(sample_rate, Hz)(waveform)
    audio = (audio - audio.mean()) / audio.std()    # 标准化
    return audio.squeeze(0)

def seperate(name, frames_file, sequen_lenth, txt_folder):
    """
    ### 该函数用于根据视频帧时间顺序来生成划分策略
    ### 输入：数据名称、对应名称的帧保存路径、序列长度、txt保存路径
    ### 输出：序列数据划分策略、图像划分策略
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
    index_metric = np.array([[i+j for j in range(sequen_lenth)] for i in range(pics_len-sequen_lenth+1)])
    file_sperated_int = np.array([[pic_index[i] for i in indexs] for indexs in index_metric])

    # 去除不是连续的
    index = []
    for i in range(len(file_sperated_int)):
        # 判断是不是等差数列，使用等差数列求和公式
        if sum(file_sperated_int[i]) == (max(file_sperated_int[i])+min(file_sperated_int[i]))*sequen_lenth/2:
            index.append(i)

    seuqence_index = file_sperated_int[index]   # 这个是针对音频的索引和标签的索引
    index_metric = index_metric[index]          # 

    return seuqence_index, index_metric

def write_txt_pic(pic_fullname, name, txt_folder):
    """
    ### 该函数用于将排好顺序的图像数据帧路径写入到txt
    ### 输入：排好顺序的图像数据帧路径数组、数据名称、txt保存路径
    """
    txt_dir = os.path.join(txt_folder, name+".txt")
    txt_file = open(txt_dir,'w')
    txt_file.truncate(0)    # 对原文件内容直接清空

    for name in pic_fullname:
        txt_file.write(name+"\n")
    txt_file.close()

def write_txt_index(seuqence_index, index_metric, name, txt_folder):
    """
    ### 该函数用于将划分策略保存为txt文件
    ### 输入：序列索引数组、图像划分数组、数据名称、txt保存路径
    """
    txt_dir = os.path.join(txt_folder, name+"_index.txt")
    txt_file = open(txt_dir,'w')    # 用于建立这个文件
    txt_file.truncate(0)    # 对原文件内容直接清空
    txt_file.close()

    # 保存索引数组，保存为整数
    np.savetxt(txt_dir, seuqence_index, fmt="%d")

    txt_dir = os.path.join(txt_folder, name+"_pics.txt")
    txt_file = open(txt_dir,'w')    # 用于建立这个文件
    txt_file.truncate(0)    # 对原文件内容直接清空
    txt_file.close()

    # 保存索引数组，保存为整数
    np.savetxt(txt_dir, index_metric, fmt="%d")

def get_audio_sequence(seuqence_index, audio_path, audio_frame_len):
    """
    ### 该函数用于根据划分策略来生成音频数据数组
    ### 输入：音频划分策略索引数组、音频保存路径
    ### 输出：划分好的音频数据数组
    """
    # 对音频每audio_frame_len长度作为一帧与图像对应
    audio = extract_audio(audio_path).numpy()   # 必须要转化为numpy格式才好做转换
    audio_frame_num = len(audio) // audio_frame_len
    # 将音频进行截断分组
    audio = np.array(np.split(audio, audio_frame_num))  # 将音频平均分段
    audio = audio[seuqence_index]

    return audio

def get_pic_sequence(pic_index, image_dirs_txt):
    """
    ### 该函数用于根据划分策略来生成图像路径数组
    ### 输入：图像划分策略索引数组、视频帧保存路径
    ### 输出：划分好的图像路径数组
    """
    with open(image_dirs_txt, "r") as f:
        # img_dirs = np.array([line.strip("\n") for line in f.readlines()])
        imgs = []
        for img_dir in f:
            img = Image.open(img_dir.strip("\n")).convert("RGB").resize((90, 90))
            imgs.append(img)
        imgs = np.array(imgs, dtype=object)[pic_index]
    
    f.close()

    return imgs

def get_index(index_txt):
    """
    ### 该函数用于在读取数据时方便将划分策略（即索引）的txt转换成数组
    """
    with open(index_txt, "r") as f:
        index = np.loadtxt(index_txt, dtype=int)
        f.close()
    
    return index

def seperqte_all(frames_files, sequen_lenth, txt_folder):
    """
    ### 该函数用于将所有的数据集按视频帧顺序来划分，并将划分策略（即索引）保存于txt
    ### 输入：视频帧的文件夹路径、序列长度、txt保存路径
    """
    names = os.listdir(frames_files)
    for name in names:
        frames_file = os.path.join(frames_files, name)
        seuqence_index, index_metric = seperate(name, frames_file, sequen_lenth, txt_folder)
        write_txt_index(seuqence_index, index_metric, name, txt_folder)

def names_txt(names_path, save_path):
    """
    ### 用于生成数据集名称的txt的函数，以便程序可按此txt的名称顺序来读取数据集
    ### 输入：包含数据集名称的文件夹路径、名称txt保存路径
    """
    names = os.listdir(names_path)

    txt_dir = os.path.join(save_path, "names.txt")
    txt_file = open(txt_dir,'w')
    txt_file.truncate(0)    # 对原文件内容直接清空

    for name in names:
        txt_file.write(name+"\n")
    
    txt_file.close()

def read_all_data(names_txt_dir, audio_path, txt_path, label_path):
    """
    ### 这个函数用于读取所有划分好的图像路径以及音频数据，以及得到对应序列的标签
    ### 输入：数据集名称txt，音频路径，预划分策略的文本路径，标签路径
    ### 输出：按顺序划分好的图像路径数组、音频数据集数组、标签数组
    """
    start_time = time.time()

    names_txt = open(names_txt_dir, 'r')
    names = names_txt.readlines()
    
    frist = True
    imgs_all = None
    audio_all = None
    arousal_all = None
    valence_all = None

    for name in names:
        name = name.strip("\n")
        video_txt_dir = os.path.join(txt_path, name+".txt")
        audio_txt_dir = os.path.join(audio_path, name+".wav")
        index_txt_dir = os.path.join(txt_path, name+"_index.txt")
        pic_txt_dir = os.path.join(txt_path, name+"_pics.txt")
        
        seuqence_index = get_index(index_txt_dir)
        pic_index = get_index(pic_txt_dir)
        imgs = get_pic_sequence(pic_index, video_txt_dir)
        audio = get_audio_sequence(seuqence_index, audio_txt_dir, audio_frame_len=640)

        arousal, valence = read_all_label(name, label_path, seuqence_index)
        
        if frist:
            imgs_all = imgs
            audio_all = audio
            arousal_all = arousal
            valence_all = valence
            frist = False
        else:
            imgs_all = np.concatenate((imgs_all, imgs), axis=0)
            audio_all = np.concatenate((audio_all, audio), axis=0)
            arousal_all = np.concatenate((arousal_all, arousal), axis=0)
            valence_all = np.concatenate((valence_all, valence), axis=0)
        
        print(name, " finish reading ! ")

    end_time = time.time()
    print(imgs_all.shape, audio_all.shape, arousal_all.shape, valence_all.shape)
    print("take time: ", int(end_time-start_time), " S. ")
    names_txt.close()

    return imgs_all, audio_all, arousal_all, valence_all

def read_arff(arff_path):
    """
    ### 该函数用于读取arff文件
    ### 输入：arff文件路径
    ### 输出：标签的numpy.ndarray数组
    """
    with open(arff_path, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        data_pd = pd.read_csv(f, header=None)
        data_pd.columns = header

        f.close()
    
    return data_pd["GoldStandard"].values   # .value可以直接转化为numpy数组
    
def read_all_label(name, label_path, seuqence_index):
    """
    ### 该函数用于读取对应每个视频文件的标签信息
    ### 输入：文件名、标签路径、序列索引
    ### 输出：根据序列索引来整理后的arousal和valence数据
    """
    arousal_dirs = os.path.join(label_path, "arousal")
    valence_dirs = os.path.join(label_path, "valence")
    
    arousal_arff = os.path.join(arousal_dirs, name+".arff")
    valence_arff = os.path.join(valence_dirs, name+".arff")
    arousal = read_arff(arousal_arff)
    valence = read_arff(valence_arff)

    last_frame_index = seuqence_index[:, -1]    # 取最后一帧的标签作为这一个序列的标签
    arousal = arousal[last_frame_index]
    valence = valence[last_frame_index]

    return arousal, valence


if __name__ == "__main__":
    names_txt_dir = "/home/MyServer/data/RECOLA/names.txt"
    frames_path = "/home/MyServer/data/RECOLA/video_frames"
    audio_path = "/home/MyServer/data/RECOLA/recordings_audio/recordings_audio"
    txt_path = "/home/MyServer/data/RECOLA/txt_data"
    label_path = "/home/MyServer/data/RECOLA/labels/gs_1"
    read_all_data(names_txt_dir, audio_path, txt_path, label_path)
    # read_all_label(names_txt_dir, label_path, txt_path)
    # seperqte_all(frames_path, 5, txt_path)

    # read_all_label(names_txt_dir, label_path, txt_path)
    # data = read_arff("/home/MyServer/data/RECOLA/labels/gs_1/arousal/dev_1.arff")
    # data = data["GoldStandard"].values
    # print(data)
    
    
