import argparse
import torch

import os,sys
dir_src = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(dir_src)

from FSN_modules import *
from FullSubnetPlus import FullSubNet_Plus


class SE(device="cuda:0"):
    def __init__(self,path="",n_fft=512,n_hop=256):
        self.device = device
        self.n_fft = n_fft
        self.n_hop = n_hop


        self.model = FullSubNet_Plus()
        chkpt = torch.load(path,map_location="cpu")

        if "model" in chkpt :
            state_dict = chkpt["model"]
        else : 
            state_dict = chkpt
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.window = torch.hann_window(n_fft)

    def enhance(self,x):
        X = torch.stft(
            x,
            n_fft = self.n_fft,
            hop_length = self.n_hop,
            window = self.window,
            return_complex = True
        )

        X = X.unsqueeze(0)
        with torch.no_grad() : 
            # preprocess input
            m,r,i = self.model.input(X)
            # run model
            output = self.model(m,r,i)

            # process output
            Y = self.output(output,r,i)

            # iSTFT
            y = torch.istft(
                Y,
                n_fft = self.n_fft,
                hop_length = self.n_hop,
                window = self.window
            )

            y = y.detach().squeeze(0),cpu(),numpy()
            return y

if __name__ == "__main__" : 
    import librosa as rs
    import soundfile as sf

    x = rs.load("input.wav")[0]

    se = SE()
    se.enhance()