import numpy as np
import librosa  as rs
from enum import Enum

class state(Enum):
   off_speech=0
   rising=1
   on_speech =2
   falling = 3

class VAD():
    def __init__(self,
    thr_energy=0.04,
    thr_up = 4,
    thr_down = 40,
    freq_s = 10,
    freq_e = 200,
    unit_frame = 30,
    max_utterance = 400, # 4 sec (1/fs*n_hop*500 = 4),
    fs=16000,
    n_hop=256,
    n_fft= 1024,
    log=False
    ):
        # data parameters
        self.n_hop=n_hop
        self.n_fft= n_fft

        # algorithm parameters
        self.thr_energy = thr_energy
        self.thr_up = thr_up
        self.thr_down = thr_down
        self.unit_frame = unit_frame

        self.freq_s = freq_s
        self.freq_e = freq_e

        self.ret = False
        self.max_utterance = max_utterance
        
        """
            + state 
                0 : inactive
                1 : rising edge
                2 : on active
                3 : falling edge
        """
        self.state = state.off_speech
        self.activation = False
        self.is_utterance = False
        self.cnt = 0
        self.cnt_frame = 0
        self.cnt_speech = 0

        self.log = log
    
    """
        x : mag [ F,T ]
    """
    def feed(self,X):
        for idx in range(X.shape[1]):
            val = X[self.freq_s:self.freq_e,idx].mean()
            #print("VAD::val[{}] | {:.4f} | {} | {}| {}".format(self.total_idx,val,self.cnt,self.activation,self.state))

            if val > self.thr_energy :
                self.activation = True
            else :
                self.activation = False

            ## state machine
            if self.activation : 
                if self.state == state.off_speech :
                    self.state = state.rising
                    self.cnt = 0

                elif self.state == state.rising :
                    self.cnt+=1
                    if self.cnt >= self.thr_up : 
                        self.state = state.on_speech
                        self.cnt = 0
                        self.is_utterance = True
                        if self.log:
                            print("VAD::on_speech {}".format(val))

                elif self.state == state.falling :
                    self.state = state.on_speech
                    self.cnt = 0
                # on_speech
                else :  
                    self.cnt_speech+=1
                    if self.cnt_speech > self.max_utterance :
                        print("VAD::Utterance is too long. ")
                        self.state = state.rising
                        self.is_utterance = False
                        self.cnt_speech = 0

            else :
                if self.state == state.rising :
                    self.state = state.off_speech
                    self.cnt = 0
                elif self.state == state.on_speech :
                    self.state = state.falling
                    self.cnt = 0

                elif self.state == state.falling :

                    self.cnt+=1
                    if self.cnt > self.thr_down :
                        self.state = state.off_speech
                        self.is_utterance = False
                        if self.log :
                            print("VAD::off_speech {}".format(val))


                # off_speech
                else :
                    pass
        #print("VAD::{:.4f} | {} | {}| {}".format(val,self.cnt,self.activation,self.state))
        return self.is_utterance

    def clear(self):
        self.cnt=0
        self.activation=False
        self.state = state.off_speech
