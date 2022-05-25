import torch
import torch.nn as nn

class AutoEncoderLoss(nn.Module):
    def __init__(self, altha, beta, gama) -> None:
        super().__init__()
        self.altha = altha
        self.beta = beta
        self.gama = gama

    def img_sig_loss(self, pr, gt):
        # img_pre和img_gt大小为[batch, length, 3, 96, 96]
        batch_size = pr.shape[0]
        length = gt.shape[1]
        # 需要注意，对于图像和音频的预测，是基于一个时间步的，而不是所有时间步
        # 因此需要对所有batch和length的样本的总loss进行取平均
        loss = torch.sum(torch.square(pr - gt)) / (batch_size*length)
        return loss
    
    def CCC(self, pr, gt):
        mean_pr = torch.mean(pr)    # 均值
        mean_gt = torch.mean(gt)
        std_pr = torch.std(pr)  # 标准差
        std_gt = torch.std(gt)
        var_pr = torch.var(pr)  # 方差
        var_gt = torch.var(gt)   
        res_pr = pr - mean_pr     # 残差
        res_gt = gt - mean_gt

        cor = torch.sum (res_pr * res_gt) / ((torch.sqrt(torch.sum(res_pr ** 2)) * torch.sqrt(torch.sum(res_gt ** 2))) + 0.0000001)
        numerator=2*cor*std_pr*std_gt   # 协方差
        denominator=var_gt+var_pr+(mean_gt-mean_pr)**2

        ccc = numerator/(denominator+0.0000001)

        return ccc
    
    def rec_loss(self, a_pr, a_gt, v_pr, v_gt):
        a_ccc = self.CCC(a_pr, a_gt)
        v_ccc = self.CCC(v_pr, v_gt)
        loss = 1 - 0.5 * (a_ccc + v_ccc)

        return loss
    
    def forward(self, image_pr, image_gt, signal_pr, 
                        signal_gt, a_pr, a_gt, v_pr, v_gt):
        image_loss = self.img_sig_loss(image_pr, image_gt)
        signal_loss = self.img_sig_loss(signal_pr, signal_gt)
        rec_loss = self.rec_loss(a_pr, a_gt, v_pr, v_gt)

        loss = self.altha*image_loss + self.beta*signal_loss + self.gama*rec_loss

        return loss


if __name__ == "__main__":
    loss = AutoEncoderLoss()
    x1 = torch.randn([1, 20, 3, 96, 96])     # 序列长度为50，batch大小为4
    x2 = torch.ones([5, 20, 1, 640])
    a_pr = torch.tensor([12.3, 1.1, 4.5, 9.0])    # batchsize为20
    a_gt = torch.tensor([11.1, 23.5, 6.1, 8.4])
    ccc = loss.CCC(a_pr, a_gt)
    print(ccc)

    

