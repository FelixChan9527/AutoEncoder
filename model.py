import torch
import torch.nn as nn

class EncoderResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(EncoderResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel//2)
        self.LReLU1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channel//2, out_channel//2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel//2)
        self.LReLU2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(out_channel//2, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.LReLU3 = nn.LeakyReLU()
        self.identity = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2)
    
    def forward(self, x):
        out = self.LReLU1(self.bn1(self.conv1(x)))
        out = self.LReLU2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.identity(x)
        out += x
        out = self.LReLU3(out)

        return out

class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(DecoderResidualBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=1, stride=1, padding=0)
        self.LReLU1 = nn.LeakyReLU()
        self.deconv2 = nn.ConvTranspose2d(in_channel//2, in_channel//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.LReLU2 = nn.LeakyReLU()
        self.deconv3 = nn.ConvTranspose2d(in_channel//2, out_channel, kernel_size=1, stride=1, padding=0)
        self.LReLU3 = nn.LeakyReLU()
        self.identity = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=2, output_padding=1)
    
    def forward(self, x):
        out = self.LReLU1(self.deconv1(x))
        out = self.LReLU2(self.deconv2(out))
        out = self.deconv3(out)
        x = self.identity(x)
        out += x
        out = self.LReLU3(out)

        return out

class Encoder2D(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(Encoder2D, self).__init__()
        self.block1 = EncoderResidualBlock(in_channel, out_channel//2)
        self.block2 = EncoderResidualBlock(out_channel//2, out_channel)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out

class Decoder2D(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(Decoder2D, self).__init__()
        self.block1 = DecoderResidualBlock(in_channel, in_channel//2)
        self.block2 = DecoderResidualBlock(in_channel//2, out_channel)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out

class Encoder1D(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(Encoder1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=21, stride=1, padding=10)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=1, stride=2)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=41, stride=1, padding=20)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=1, stride=10)
    
    def forward(self, x):
        out = self.Maxpool1(self.conv1(x))
        out = self.Maxpool2(self.conv2(out))
        return out

class Decoder1D(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(Decoder1D, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channel, in_channel*10, kernel_size=21, stride=1, padding=10)
        self.upsampling = nn.Upsample(size=640)
        self.deconv2 = nn.ConvTranspose1d(in_channel*10, out_channel, kernel_size=41, stride=1, padding=20)
    
    def forward(self, x):
        out = self.deconv1(x)
        out = self.upsampling(out)
        out = self.deconv2(out)
        return out

class AutoEncoderNet(nn.Module):
    def __init__(self, img_shape, sig_shape, encoder2D_channel, encoder1D_channel) -> None:
        super(AutoEncoderNet, self).__init__()
        self.img_shape = img_shape
        self.sig_shape = sig_shape
        c_i, w, h  = self.img_shape
        c_s, l = self.sig_shape
        self.img_in_channel = c_i                       # 图像通道数
        self.img_out_channel = encoder2D_channel        # 2DConvAE输出向量大小
        self.sig_in_channel = c_s                       # 音频音频通道数
        self.sig_out_size = encoder1D_channel           # 1DConvAE输出向量大小

        self.encoder2D = Encoder2D(self.img_in_channel, self.img_out_channel)
        new_w, new_h, new_c_i = w // 4, h // 4, self.img_out_channel
        fletten2D_size = new_w * new_h * new_c_i
        self.fc_2D = nn.Linear(fletten2D_size, 2048)         # 2DConvAE的全连接层
        self.decoder2D = Decoder2D(self.img_out_channel, self.img_in_channel)

        self.encoder1D = Encoder1D(self.sig_in_channel, self.sig_out_size)
        new_l, new_c_s = l//20, self.sig_out_size
        fletten1D_size = new_l * new_c_s
        self.fc_1D_1 = nn.Linear(fletten1D_size, 640)         # 1DConvAE的第一个全连接层
        self.fc_1D_2 = nn.Linear(640, fletten1D_size)         # 1DConvAE的第一个全连接层
        self.decoder1D = Decoder1D(self.sig_out_size, self.sig_in_channel)

        self.rnn = nn.LSTM(3328, 512, 2)
        self.fc_final = nn.Linear(512, 2)
    
    def forward(self, img, sig):
        length = img.shape[0]
        batch_size = img.shape[1]

        img = img.reshape(length*batch_size, *self.img_shape)
        sig = sig.reshape(length*batch_size, *self.sig_shape)

        img_out = self.encoder2D(img)
        img_vector = img_out.reshape(length, batch_size, -1)
        img_vector = self.fc_2D(img_vector)
        img_pre = self.decoder2D(img_out).reshape(length, batch_size, *self.img_shape)

        sig_out = self.encoder1D(sig)
        sig_vector = sig_out.reshape(length, batch_size, -1)
        sig_vector = self.fc_1D_1(sig_vector)
        sig_vector = self.fc_1D_2(sig_vector)
        sig_pre = self.decoder1D(sig_out).reshape(length, batch_size, *self.sig_shape)

        vector = torch.concat((img_vector, sig_vector), dim=2)  # 向量合并
        _, (ht, _) = self.rnn(vector)
        output = ht[-1]                 # 最后一层隐藏层的一个时间步的值即为输出
        output = self.fc_final(output)  # output大小为[batchsize, 2]
        arousal_pre = output[:, 0]        # 长度为batchsize
        valence_pre = output[:, 1]        # 长度为batchsize

        return img_pre, sig_pre, arousal_pre, valence_pre


if __name__ == "__main__":
    block = AutoEncoderNet([3, 96, 96], [1, 640], 32, 40).to("cuda")
    x1 = torch.ones([5, 20, 3, 96, 96])     # 序列长度为50，batch大小为4
    x2 = torch.ones([5, 20, 1, 640])
    x1 = x1.to("cuda")
    x2 = x2.to("cuda")
    img_pre, sig_pre, arousal_pre, valence_pre = block(x1, x2)
    print(img_pre.shape, sig_pre.shape, arousal_pre.shape, valence_pre.shape)