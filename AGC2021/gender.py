import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append('./')
import torch.nn as nn
import numpy as np
import librosa
import torch
import torch.nn.functional as F

import os
import pickle
import glob

class Feature_Extractor():
    def __init__(self, n_fft=512, hopsize=128, window='hann'):
        self.nfft = n_fft
        self.hopsize = 128
        self.window = 'hann'
        self.melW = 128
        self.n_mfcc = 40
        # self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate = 16000, n_mels=128, n_fft = 512, hop_length = 128,mel_scale='slaney',norm ='slaney')  
        # self.to_DB = torchaudio.transforms.AmplitudeToDB()  

    def MFCC(self, sig):

        try:
            S = librosa.feature.mfcc(y=sig, sr=16000, n_mfcc=self.n_mfcc,
                                     n_fft=self.nfft, hop_length=self.hopsize)
        except:
            S = np.zeros((40, 50))
            # pdb.set_trace()
        return S

    def logmel(self,sig):
        #pdb.set_trace()
        S = librosa.feature.melspectrogram(y=sig,
                            n_fft=self.nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window,
                            pad_mode='reflect')
        S = librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=None)
        return S

    # def logmel(self,sig):
    #     print(sig.shape)
    #     sig = torch.from_numpy(sig)
    #     print(sig.shape)
    #     S = self.mel_transform(sig)
    #     print(S.shape)
    #     S= self.to_DB(S)
    #     print(S.shape)
    #     print('asdasd')

    #     return S

    def Chroma(self, sig):
        S = librosa.feature.chroma_stft(y=sig,
                                        sr=16000,
                                        n_fft=self.nfft,
                                        hop_length=self.hopsize)
        return S

    def tonnetz(self, sig):
        S = librosa.feature.tonnetz(y=sig,
                                    hop_length = self.hopsize,
                                    sr=16000)
        return S

'''Tester'''


class ModelTester:
    def __init__(self, model, ckpt_path, device):
        # Essential parts
        self.device = torch.device('cuda:{}'.format(device))
        # self.model = model.to(self.device)
        self.model = model.cuda()
        self.load_checkpoint(ckpt_path)
        self.model.eval()

    def load_checkpoint(self, ckpt):
        print('[task2] Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def test(self, batch):
        with torch.no_grad():
            inputs = batch
            inputs = inputs.cuda()
            inputs = torch.unsqueeze(inputs, 0)
            outputs = self.model(inputs)
            scores = outputs['clipwise_output']
            # child_preds = scores > 0.5
            # child_preds = child_preds[:,:,1]
            preds = scores > 0.3
            # preds[:,:,0] = child_preds
            #################################

            return preds


'''Network Fintuning'''


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class finetunePANNs(nn.Module):
    def __init__(self, PANNs_pretrain, class_num):
        super(finetunePANNs, self).__init__()
        self.PANNs = PANNs_pretrain

        self.add_fc1 = nn.Linear(527, class_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.add_fc1)

    def forward(self, input):
        x = self.PANNs(input)
        embed = x['embedding']
        clipwise_output = torch.sigmoid(self.add_fc1(embed))
        output_dict = {'clipwise_output': clipwise_output}

        return output_dict


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


'''Network'''


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


def _resnet_conv3x3(in_planes, out_planes):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    # 1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2),
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet38(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins,
                 classes_num):
        super(ResNet38, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(in_channels=2, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)
        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)
        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input):
        """
        Input: (batch_size, T, F)"""

        x = input.transpose(1, 3)
        # x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)

        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x.permute(0,2,1)))
        embedding = F.dropout(x, p=0.5, training=self.training)
        embedding = self.fc_audioset(embedding)

        output_dict = {'embedding': embedding}

        return output_dict

class SpectralMultiScale(nn.Module):
    def __init__(self, inp_dim, lstm_h):
        super().__init__()
        self.lstm_h = lstm_h
        self.attention = SoftAttention(lstm_h, lstm_h)
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, lstm_h, 3),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 3),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 3),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            TransposeAttn(lstm_h)
        )

        self.cnn5 = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, lstm_h, 5),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 5),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 5),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            TransposeAttn(lstm_h)
        )

        self.cnn7 = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, lstm_h, 7),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 7),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 7),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            TransposeAttn(lstm_h)
        )
        self.height_regressor = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1),
        )

        self.age_regressor = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1),
        )

        self.gender_regressor = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1),
            nn.Sigmoid()
        )
        self.gender_regressor_CEE = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(1)

        fm3 = self.cnn3(x).view(-1, self.lstm_h)
        fm5 = self.cnn5(x).view(-1, self.lstm_h)
        fm7 = self.cnn7(x).view(-1, self.lstm_h)

        fm = torch.cat([fm3, fm5, fm7], 1)
        gender = self.gender_regressor_CEE(fm)
        age = self.age_regressor(fm)

        return gender, age

class SpectralMultiScale3G2ACEE(nn.Module):
    def __init__(self, inp_dim, lstm_h):
        super().__init__()
#        inp_dim = 513
        self.lstm_h = lstm_h
        self.attention = SoftAttention(lstm_h, lstm_h)
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, lstm_h, 3),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 3),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 3),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            TransposeAttn(lstm_h)
        )

        self.cnn5 = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, lstm_h, 5),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 5),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 5),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            TransposeAttn(lstm_h)
        )

        self.cnn7 = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, lstm_h, 7),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 7),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 7),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            TransposeAttn(lstm_h)
        )

        self.age_regressor = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1),
        )

        self.gender_regressor = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1),
            nn.Sigmoid()
        )
        self.gender_regressor_CEE = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 3),
            nn.Sigmoid()
        )
        self.age_regressor_CEE = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(1)

        fm3 = self.cnn3(x).view(-1, self.lstm_h)
        fm5 = self.cnn5(x).view(-1, self.lstm_h)
        fm7 = self.cnn7(x).view(-1, self.lstm_h)

        fm = torch.cat([fm3, fm5, fm7], 1)
        gender = self.gender_regressor_CEE(fm)
        age = self.age_regressor_CEE(fm)

        return gender, age

class TransposeAttn(nn.Module):
    def __init__(self, lstm_h):
        super().__init__()
        self.attention = SoftAttention(lstm_h, lstm_h)

    def forward(self, x):
        attn_output = self.attention(x.transpose(1,2))
        return attn_output

class SoftAttention(nn.Module):
    def __init__(self, emb_dim, attn_dim):
        super().__init__()
        self.attn_dim = attn_dim
        self.emb_dim = emb_dim
        self.W = torch.nn.Linear(self.emb_dim, self.attn_dim)
        self.v = nn.Parameter(torch.Tensor(self.attn_dim), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.attn_dim)

        for weight in self.v:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, values):
        attention_weights = self._get_weights(values)
        values = values.transpose(1,2)
        weighted = torch.mul(values, attention_weights.unsqueeze(1).expand_as(values))
        representations = weighted.sum(2).squeeze()
        return representations

    def _get_weights(self, values):
        batch_size = values.size(0)
        weights = self.W(values)
        weights = torch.tanh(weights)
        e = weights @ self.v
        attention_weights = torch.softmax(e.squeeze(1), dim=-1)
        return attention_weights

def get_strong_label(entry, n_threshold_window=10, n_threshold_continue=10, sec_window=0.1):
    answer = {0: [], 1: [], 2: []}
    is_empty = True
    # detect change point
    pivot_s, pivot_e, cur_continue_start = 0, n_threshold_window, 0
    n_cur_continue, n_temp_continue = 0, 0
    cur_continue_class = -1
    while True:
        if pivot_e >= len(entry):
            last_list = entry[pivot_s:]
            last_class = max(last_list, key=last_list.count)
            if is_empty:
                if cur_continue_class == -1:
                    answer[int(max(last_list, key=last_list.count))].append([0, len(entry)-1])
                else:
                    answer[cur_continue_class].append([cur_continue_start, len(entry)-1])
            else:
                if n_cur_continue >= n_threshold_continue:
                    answer[cur_continue_class].append([cur_continue_start, len(entry)-1])
            break

        temp_list = entry[pivot_s:pivot_e]
        temp_class = max(temp_list, key=temp_list.count)
        if cur_continue_class == -1:
            cur_continue_class = temp_class
        else:
            if temp_class != cur_continue_class:
                if n_cur_continue >= n_threshold_continue:
                    answer[cur_continue_class].append([cur_continue_start, pivot_s-1])
                    is_empty = False
                pivot_s, pivot_e, cur_continue_start = pivot_s+1, pivot_e+1, pivot_s
                n_cur_continue, cur_continue_class = 0, temp_class
            else:
                n_cur_continue += 1
                pivot_s, pivot_e = pivot_s+1, pivot_e+1
    # calculate ts_ratio in wav_chunk duration
    modified_answer = {0: [], 1: [], 2: []}
    for c, d in answer.items():
        if d:
            for ts in d:
                mid = (ts[0]+ts[1])//2
                modified_answer[c].append(int(mid*0.1*16000))
    return modified_answer

def data_reformation(wav):
    wav = torch.from_numpy(wav)
    if wav.shape[0] <= 26500:
        wav = torch.nn.functional.pad(wav, (26500-wav.shape[0], 0))

    wav = wav / max(torch.abs(torch.max(wav)), torch.abs(torch.min(wav)))

    spectral_transform = torchaudio.transforms.Spectrogram(n_fft=1024)
    wav = spectral_transform(wav)

    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav

'''JSON'''


def sp2clock(idx, sr=16000):
    sec = int(idx / 16000)
    minu = int(sec // 60)
    result = '{:02d}:{:02d}'.format(minu, sec - 60 * minu)
    return result

class gender():
    def __init__(self,ckpt_path) : 
        '''Feature Extractor Init'''
        self.f_extractor = Feature_Extractor()

        '''Model Init'''
        PANNs_model = ResNet38(32000, 1024, 1024 // 4, 128, 3)
        self.gender_model = finetunePANNs(PANNs_model, 3)

        # pdb.set_trace()
        '''Tester Init'''
        #tester = ModelTester(gender_model, cfg.TASK2.GENDER_MODEL, 0)
        self.tester = ModelTester(self.gender_model, ckpt_path, 0)

        '''Class List'''
        self.hop_size = 128
        self.sr  = 16000

    def process(self,wav,enhance_wav,timestamps) : 
        class_list = ['C','W','M']

        enhance_wav = np.squeeze(enhance_wav,axis=0)
        enhance_max = np.max(enhance_wav)
        noisy_max = np.max(wav)
        #real_max = max(enhance_max,noisy_max)
        enhance_wav /= enhance_max
        wav /= noisy_max

        outputs = []

        # iter and load wav, pass gender classification
        for ts in timestamps:
            start_ts, end_ts = ts
            start_idx, end_idx = int(start_ts * self.sr), int(end_ts * self.sr)
            # duration = end_idx-start_idx
            #duration = end_ts - start_ts
            wav_chunk = wav[start_idx:end_idx]
            enhance_chunk = enhance_wav[start_idx:end_idx]
            ###########################################################################

            label = {"ts":[int(ts[0]),int(np.ceil(ts[1]))],"class":[]}

            '''Feature Extraction'''
            feature1 = self.f_extractor.MFCC(wav_chunk)
            feature2 = self.f_extractor.MFCC(enhance_chunk)

            feature1 = self.f_extractor.logmel(wav_chunk)
            feature2 = self.f_extractor.logmel(enhance_chunk)

            feature = np.stack((feature1,feature2),axis=0)
            
            m_inputs =  torch.FloatTensor(feature).transpose(1,2)
            input_T = m_inputs.shape[1]
            
            '''Model Forwarding'''
            m_outputs = self.tester.test(m_inputs)
            
            output = {'C': None, 'W': None, 'M': None}
            output_T = m_outputs.shape[1]
            alpha = (input_T / output_T)
            for i in range(3):
                class_name = class_list[i]
                tmp = m_outputs[0,:,i]
                exists_frame_idx = (tmp == True).nonzero(as_tuple=True)[0]
                
                if len(exists_frame_idx) == 0:
                    continue
                else:
                    label["class"].append(class_name)
            if len(label["class"]) == 0:
                continue
            outputs.append(label)
        return outputs


