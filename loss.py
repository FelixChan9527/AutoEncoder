import torch
import torch.nn as nn

class AutoEncoderLoss(nn.Module):

    def img_sig_loss(self, pre, gt):
        # img_pre和img_gt大小为[batch, length, 3, 96, 96]
        batch_size = pre.shape[0]
        length = gt.shape[1]
        # 需要注意，对于图像和音频的预测，是基于一个时间步的，而不是所有时间步
        # 因此需要对所有batch和length的样本的总loss进行取平均
        loss = torch.sum(pre - gt) / (batch_size*length)
        return loss
    
    def CCC(self, pre, gt):
        pass


if __name__ == "__main__":
    loss = AutoEncoderLoss()
    x1 = torch.randn([1, 20, 3, 96, 96])     # 序列长度为50，batch大小为4
    x2 = torch.ones([5, 20, 1, 640])
    loss.CCC()

    

