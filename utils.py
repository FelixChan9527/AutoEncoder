import cv2
import os
import torch
import torchaudio
import matplotlib.pyplot as plt

video_frames_path = "/home/MyServer/data/RECOLA/video_frames"
video_path = "/home/MyServer/data/RECOLA/recordings_video/recordings_video"
audio_frames_path = "/home/MyServer/data/RECOLA/audio_frames"
audio_path = "/home/MyServer/data/RECOLA/recordings_audio/recordings_audio"
filename = "/home/MyServer/data/RECOLA/recordings_audio/recordings_audio/dev_5.wav"
pic_path = "/home/MyServer/MyCode/AutoEncoder/test.jpg"
frame_path = "/home/MyServer/data/RECOLA/video_frames/train_3/7_280.jpg"
xml_path = "/usr/local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"

def extract_face(frame, frame_buff):    # 提取单帧的人脸数据
    face_detector = cv2.CascadeClassifier(xml_path)
    face = face_detector.detectMultiScale(frame)
    if len(face) != 0:
        x, y, w, h = face[0]
        frame = frame[x: x+w, y: y+h]
        return frame
    else:   # 检测不到人脸，则返回上一帧的人脸
        return frame_buff

def extract_imgs(video_path, video_frames_path):
    """
    ### 这个函数用于视频抽帧
    """
    videos_names = os.listdir(video_path)
    for name in videos_names:
        full_name = os.path.join(video_path, name)
        save_path = os.path.join(video_frames_path, name[:-4])
        
        video = cv2.VideoCapture(full_name)
        frame_exit = video.isOpened()
        frame_num = 0

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        frame_buff = 0
        while frame_exit:
            frame_exit, frame = video.read()            # 获取下一帧
            frame = extract_face(frame, frame_buff)                 # 截取人脸
            time = video.get(cv2.CAP_PROP_POS_MSEC)     # 获取当前时间戳

            ex = (time // 40) -  frame_num  # 正常来说，ex应该为0
            # 执行以下循环，说明发生了跳帧
            for i in range(int(ex)):
                frame_name = str(frame_num)+"_"+str(int(time-i*40)) + ".jpg"
                frame_path = os.path.join(save_path, frame_name)
                frame_num += 1
                cv2.imwrite(frame_path, frame_buff)
                print(frame_path)
            frame_buff = frame  # 保留本次的图像，供下一次如果掉帧时使用

            frame_name = str(frame_num)+"_"+str(int(time)) + ".jpg"
            frame_path = os.path.join(save_path, frame_name)

            if frame_exit:
                cv2.imwrite(frame_path, frame)
                pass
            else:
                break

            print(frame_path)
            frame_num += 1

def extract_audio(audio_path, audio_frames_path):
    pass

def torchaudio_test():
    waveform, sample_rate = torchaudio.load(filename)   # 加载音频
    # 音频重采样，将音频变为16kHZ，最终使得一帧为0.04s，每帧为长度640的时间序列
    transformed = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    transformed = (transformed - transformed.mean()) / transformed.std()
    

if __name__ == "__main__":
    extract_imgs(video_path, video_frames_path)
    # frame = cv2.imread(frame_path)
    # frame = extract_face(frame)
    # cv2.imwrite(pic_path, frame)
    # torchaudio_test()
    # path = "/home/MyServer/data/RECOLA/recordings_video/recordings_video/dev_5.mp4"
    # cap = cv2.VideoCapture(path)
    # print(cap.get(7)/cap.get(5)/60)

