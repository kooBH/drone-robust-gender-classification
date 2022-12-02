import argparse
import glob
import os
import pickle
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


###################################################################
ENHANCE_MODEL_PATH = "./weights/task2/U"
###################################################################


def calc_mean_invstddev(feature):
    if len(feature.size()) != 2:
        raise ValueError("We expect the input feature to be 2-D tensor")
    mean = feature.mean(0)
    var = feature.var(0)
    # avoid division by ~zero
    eps = 1e-8
    if (var < eps).any():
        return mean, 1.0 / (torch.sqrt(var) + eps)
    return mean, 1.0 / torch.sqrt(var)


def apply_mv_norm(features):
    for i in range(features.size(0)):
        if i == 0:
            mean, invstddev = calc_mean_invstddev(features[i])
            res = (features[i] - mean) * invstddev
            res = res.unsqueeze(0)
        else:
            mean, invstddev = calc_mean_invstddev(features[i])
            res1 = (features[i] - mean) * invstddev
            res1 = res1.unsqueeze(0)
            res = torch.cat([res, res1], dim=0)
    return res


class ResidualBlock_transcnn(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock_transcnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=5, stride=stride, padding=2, groups=in_channel),
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),

        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, kernel_size=5, stride=stride, padding=2, output_padding=1,
                               groups=out_channel),
            nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride

    def forward(self, x):
        residual = x

        if self.stride == 1:
            out = self.layer1(x)

            if residual.shape[2] < out.shape[2]:
                residual = self.upsample(residual)
            out += residual

        if self.stride == 2:
            out = self.layer2(x)
            if residual.shape[2] < out.shape[2]:  ## time compare
                residual = self.upsample(residual)

            out += residual

        return out


class Resnet_transcnn(nn.Module):
    def __init__(self, block, init_channel, out_channel, blockslayers, stride):
        super(Resnet_transcnn, self).__init__()
        self.layer = self.make_layer(block, init_channel, out_channel, blockslayers, stride)

    def make_layer(self, block, init_channel, out_channel, blockslayers, stride):
        layers = []
        layers.append(block(init_channel, out_channel[0], stride[0]))
        for i in range(blockslayers - 1):
            layers.append(block(out_channel[i], out_channel[i + 1], stride[i + 1]))
        # print(layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class ResidualBlock_cnn(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock_cnn, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=5, stride=stride, padding=2, groups=in_channel),
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
        )
        self.upchannel = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.layer(x)
        # print(out.shape)
        if residual.shape[1] < out.shape[1]:  # channel compare
            residual = self.upchannel(residual)
            # print(residual.shape)

        out += residual
        return out


class Resnet_cnn(nn.Module):
    def __init__(self, block, init_channel, out_channel, blockslayers, stride):
        super(Resnet_cnn, self).__init__()
        self.layer = self.make_layer(block, init_channel, out_channel, blockslayers, stride)

    def make_layer(self, block, init_channel, out_channel, blockslayers, stride):
        layers = []
        layers.append(block(init_channel, out_channel[0], stride[0]))
        for i in range(blockslayers - 1):
            layers.append(block(out_channel[i], out_channel[i + 1], stride[i + 1]))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps,
                                    track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps,
                                    track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False,
                 padding_mode="zeros"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding

        if complex:
            conv = ComplexConv2d
            bn = ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, padding=(0, 0), complex=False):
        super().__init__()
        if complex:
            tconv = ComplexConvTranspose2d
            bn = ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d

        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               output_padding=output_padding, padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels=2,
                 complex=False,
                 model_complexity=45,
                 model_depth=20,
                 padding_mode="zeros"):
        super().__init__()

        if complex:
            model_complexity = int(model_complexity // 1.414)

        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        self.encoders = []
        self.model_length = model_depth // 2

        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=complex,
                             padding_mode=padding_mode)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1],
                             kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i],
                             output_padding=self.dec_output_paddings[i], complex=complex)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        if complex:
            conv = ComplexConv2d
        else:
            conv = nn.Conv2d

        linear = conv(self.dec_channels[-1], 2, 1)

        self.add_module("linear", linear)
        self.complex = complex
        self.padding_mode = padding_mode
        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)

        self.sigmoid = nn.Sigmoid()

    def forward(self, real, imag):

        real_spec = real
        imag_spec = imag
        real_spec4 = torch.unsqueeze(real_spec, 1)  # B 1 321 4T]
        imag_spec4 = torch.unsqueeze(imag_spec, 1)  # B 1 321 4T]
        cmp_spec = torch.cat([real_spec4, imag_spec4], 1)  # [B 2 F T]
        # cmp_spec = torch.unsqueeze(cmp_spec,1) #[B 1 2 F T]
        # cmp_spec = cmp_spec.permute(0,1,3,4,2)
        if self.complex:
            x = cmp_spec
        else:
            x = cmp_spec
        # go down
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)

            x = encoder(x)
            # print("x{}".format(i), x.shape)
        # xs : x0=input x1 ... x9

        # print(x.shape)
        po = x
        for i, decoder in enumerate(self.decoders):
            po = decoder(po)
            if i == self.model_length - 1:
                break
            # print(f"p{i}, {po.shape} + x{self.model_length - 1 - i}, {xs[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            skipconnection = xs[self.model_length - 1 - i]
            po = torch.cat([po, skipconnection], dim=1)

        # print(p.shape)
        mask = self.linear(po)
        mask = self.sigmoid(mask)
        # mask = torch.squeeze(mask,1)

        # bd['M_hat'] = mask
        return real_spec * mask[:, 0, :, :], imag_spec * mask[:, 1, :, :]

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 10:
            pass
        elif model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (7, 5),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (1, 1),
                                (1, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 (3, 2),
                                 (3, 2),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1), ]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            self.dec_kernel_sizes = [(5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (7, 5),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (1, 1),
                                (1, 1),
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (3, 2),
                                 (3, 2),
                                 (0, 3),
                                 (3, 0)]
            self.dec_output_paddings = [(0, 0),  # 17
                                        (0, 0),  # 33
                                        (0, 0),  # 65
                                        (0, 0),  #
                                        (0, 0),  #
                                        (0, 0),
                                        (0, 0),
                                        (0, 0),
                                        (0, 0),
                                        (0, 0)]

        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))


def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1)  # Batch preserving sum for convenience.

    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean ** 2) / (bsum(clean ** 2) + bsum(noise ** 2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    # wSDR = mSDRLoss(clean, clean_est)
    return torch.mean(wSDR)


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return -torch.mean(snr)


class GrandChallengeDataset_Test(Dataset):

    def __init__(self, data_test):
        self.wav_list = data_test
        self.test_wav, _ = librosa.load(self.wav_list, sr=16000)  # mono로 읽음
        self.test_wav = torch.from_numpy(self.test_wav).unsqueeze(0)
        self.tgt_wav_len = self.test_wav.shape[1]

        self.win_len = int(1024)
        self.hop_len = int(256)
        self.window = torch.hann_window(window_length=self.win_len, periodic=True,
                                        dtype=None, layout=torch.strided, device=None,
                                        requires_grad=False,
                                        )

        self.test_spec = torch.stft(self.test_wav,
                                    window=self.window, n_fft=self.win_len,
                                    hop_length=self.hop_len, win_length=self.win_len
                                    )
        self.test_spec = self.test_spec[0]  # 1 F T 2 --> F T 2

        self.total_chunk = self.test_spec.shape[1] // 512

    def __getitem__(self, index):
        # pdb.set_trace()
        data_full_name = self.wav_list
        data_name = os.path.splitext(os.path.split(data_full_name)[1])[0]

        if index == self.total_chunk:
            out_data = self.test_spec[:, index * 512:512 * (index + 1), :]
            out_data_real = out_data[:, :, 0]
            out_data_imag = out_data[:, :, 1]

            tgt_wav_len = out_data.shape[1]

            empty_in_r = torch.zeros([513, 512])
            empty_in_r[:, :tgt_wav_len] = out_data_real
            out_data_real = empty_in_r

            empty_in_i = torch.zeros([513, 512])
            empty_in_i[:, :tgt_wav_len] = out_data_imag
            out_data_imag = empty_in_i

        out_data = self.test_spec[:, index * 512:512 * (index + 1), :]
        out_data_real = out_data[:, :, 0]
        out_data_imag = out_data[:, :, 1]

        return data_name, out_data_real, out_data_imag, self.tgt_wav_len, data_full_name

    def __len__(self):
        return self.total_chunk + 1


def search(d_name, li):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.wav':
                li.append(os.path.join(os.path.join(os.path.abspath(d_name), paths), filename))
    len_li = len(li)
    return li


def complex2audio(complex_ri, window, length):
    window = window
    length = length
    complex_ri = complex_ri
    audio = torch.istft(input=complex_ri, n_fft=int(1024), hop_length=int(256), win_length=int(1024), window=window,
                        center=True, normalized=False, onesided=True, length=length)
    return audio

class enhancer():
    def __init__(self,path_chkpt="Unet_V2.pth",device="cuda:0"):
        self.device = device
        self.model = UNet()
        self.model.load_state_dict(torch.load(path_chkpt, map_location=device))
        self.model = self.model.to(device)

        self.window = torch.hann_window(window_length=int(1024), periodic=True, dtype=None, layout=torch.strided, device=None,requires_grad=False).to(device)

    def process(self,path_audio):
        with torch.no_grad():
            test_dataset = GrandChallengeDataset_Test(path_audio)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
             num_workers=0)

            for j, (data_name, out_data_real, out_data_imag, tgt_wav_len, data_full_name) in enumerate(test_loader):
                if j == 0:
                    audio_real = out_data_real.to(self.device)
                    audio_imagine = out_data_imag.to(self.device)

                    enhance_r, enhance_i = self.model(audio_real, audio_imagine)
                    enhance_r = enhance_r.unsqueeze(3)
                    enhance_i = enhance_i.unsqueeze(3)
                    enhance_spec = torch.cat((enhance_r, enhance_i), 3)
                else:
                    audio_real = out_data_real.to(self.device)
                    audio_imagine = out_data_imag.to(self.device)

                    enhance_r, enhance_i = self.model(audio_real, audio_imagine)
                    enhance_r = enhance_r.unsqueeze(3)
                    enhance_i = enhance_i.unsqueeze(3)
                    enhance_spec2 = torch.cat((enhance_r, enhance_i), 3)
                    enhance_spec = torch.cat((enhance_spec, enhance_spec2), 2)

            audio_max_len = int(enhance_spec.shape[2] * 256 - 1)
            audio_me_pe = complex2audio(enhance_spec, self.window, audio_max_len)

            audio_me_pe = audio_me_pe.to('cpu')
            output_wav = audio_me_pe[:, :int(tgt_wav_len[0])]

            output_wav = output_wav.numpy()

        return output_wav
