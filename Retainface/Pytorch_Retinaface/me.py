from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

save_path = "/home/MyServer/MyCode/Retainface/Pytorch_Retinaface/"
handsom_path = "/home/MyServer/data/RECOLA/video_frames/train_7/1_40.jpg"

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def model_config(device, model_path):
    torch.set_grad_enabled(False)
    net = RetinaFace(cfg_re50, phase = 'test')

    div = torch.cuda.current_device()
    pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(div))
    
    # 加载参数
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(net, pretrained_dict)
    net.load_state_dict(pretrained_dict, strict=False)
    net.eval()
    cudnn.benchmark = True
    net = net.to(device)

    return net

def pic_predict(device, net, img_raw):
    # img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # _t['forward_pass'].tic()
    loc, conf, landms = net(img)  # forward pass 模型预测****
    # _t['forward_pass'].toc()
    # _t['misc'].tic()
    priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_re50['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    # order = scores.argsort()[::-1][:args.top_k]
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)

    dets = dets[keep, :]
    landms = landms[keep]
    
    dets = np.concatenate((dets, landms), axis=1)
    
    max_idx = 0
    try:
        max_idx = np.argmax(dets[:, 4])     # 挑出可能性最大的人脸
    except:     # 出现问题一般是没检测出人脸
        acc = 0
        img_raw = 0
        return acc, img_raw
        
    # print(int(dets[max_idx, 1]), int(dets[max_idx, 3]), int(dets[max_idx, 0]), int(dets[max_idx, 2]))
    acc = dets[max_idx, 4]*100
    if dets[max_idx, 1] < 0:
        dets[max_idx, 1] = 0
    if dets[max_idx, 0] < 0:
        dets[max_idx, 0] = 0
    img_raw = img_raw[int(dets[max_idx, 1]): int(dets[max_idx, 3]), int(dets[max_idx, 0]): int(dets[max_idx, 2])]
    # print(acc, "%")

    return img_raw, acc

if __name__ == "__main__":
    model_path = "/home/MyServer/model_file/RetinaFace/Retinaface_model_v2/Resnet50_Final.pth"
    img_raw = cv2.imread(handsom_path, cv2.IMREAD_COLOR)
    device = torch.device("cuda")
    net = model_config(device, model_path)
    pic_predict(device, net, img_raw)
#####################################################################
    




