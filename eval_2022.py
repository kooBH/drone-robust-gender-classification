from tqdm.auto import tqdm

import librosa as rs
import soundfile as sf
import numpy as np
import torch

import pickle,os,glob
dir_src = os.path.dirname(os.path.abspath(__file__))

import argparse
import shutil

from config.hparams import HParam

from BufferManager import BufferManager
from SpeechEnhancement.DCUNET import UNet
from VAD.VAD_energy import VAD
from DSP.DSP import DSP
from GenderClassification.Gender import Gender


def eval(path,hp,buffer,window,denoising,vad,dsp,gender,device="cuda:0"):
    raw = rs.load(path,sr=hp.de.sr,mono=False)[0]
    #raw = raw[:,1300000:2000000]
    len_data = int(raw.shape[1]/hp.bm.n_chunk)

    idx = 0
    cnt_utt = 0
    cnt_sil = 0

    # C F M N 
    list_cls=np.zeros(hp.gd.n_class,np.int32)

    label = []

    bool_speech_on = False

   # read chunk by chunk
    while idx < len_data : 
        #print("{} | feed : {}".format(idx,raw[:,idx*n_chunk:(idx+1)*n_chunk].shape))
        data,flag_ret = buffer.feed(raw[:,idx*hp.bm.n_chunk:(idx+1)*hp.bm.n_chunk])

        ## enough data to process
        if data is not None : 
            #print("process : {}".format(data.shape))  
            #data = data/np.max(np.abs(data)+1e-7)

            X = torch.stft(torch.from_numpy(data).to(device),
                hop_length=hp.de.n_hop, n_fft=hp.de.n_fft,window = window,center=False
                ).float()
            #X = torch.unsqueeze(X,0)
            ## Denosing
            denoised_real,denoised_imag = denoising(X[0:1,:,:,0],X[0:1,:,:,1])
            # denoised_mag [1,F,T]
            denoised_mag = torch.sqrt(torch.pow(denoised_real,2)+torch.pow(denoised_imag,2))
            #print(denoised_mag.shape)


            ## VAD
            is_utterance =  vad.feed(denoised_mag[0,:,:])

            if not is_utterance :
                cnt_sil += 1
            flag_utt_start = buffer.toggle_utt(is_utterance)

            if(flag_utt_start) : 
                output = {"ts":[],"class":[]}
                ts = int(idx*hp.bm.n_chunk/16000)
                output["ts"].append(ts)
            
        ## Utterance Generated
        if flag_ret : 
            ts = int(np.ceil(idx*hp.bm.n_chunk/16000))
            output["ts"].append(ts)

            utt = buffer.get_utterance()
            cnt_utt +=1

            utt = rs.resample(utt,sr_de,sr_dsp,res_type="fft")

            ## DSP
            n_frame_utt = int(utt.shape[1]/n_hop_dsp)
            dsp.Process(utt,n_frame_utt)

            ## GenderClassification
            pred = gender.infer(utt[0])

            if pred == 2: 
                cls_gender = "M"
            elif pred == 1 : 
                cls_gender = "W"
            elif pred == 0 : 
                cls_gender = "C"
            # pred == 3 or -1
            else :
                cls_gender = "N"

            output["class"].append(cls_gender)
            label.append(output)

            cnt_sil = 0
        idx += 1

    return label

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default="config/v24.yaml")
    parser.add_argument('--default',type=str,default="config/default.yaml")
    parser.add_argument('--device',"-d",type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    dir_in = "eval"
    dir_out = "output_2022"

    os.makedirs(dir_out,exist_ok=True)

    hp = HParam(args.config,args.default)
    print("Configuration {} based on {}".format(args.config,args.default))

    ### Parameters 
    n_channel = 7

    ## Buffer Manager
    n_chunk = hp.bm.n_chunk
    n_unit = hp.bm.n_unit
    n_hop = hp.bm.n_hop
    device = args.device
    print(device)
    max_unit = hp.bm.max_unit

    ## Denosinig
    n_fft_de = hp.de.n_fft
    n_hop_de = hp.de.n_hop
    sr_de = hp.de.sr

    ## VAD
    pre_unit = hp.vad.pre_unit

    ## Speech Enhancement
    n_fft_se = hp.se.n_fft
    n_hop_se = hp.se.n_hop
    sr_se = hp.se.sr

    ## DSP
    n_fft_dsp = hp.dsp.n_fft
    n_hop_dsp = hp.dsp.n_hop
    sr_dsp = hp.dsp.sr

    ## Gender Classification
    n_class = hp.gd.n_class

    ## Init 
    os.makedirs(dir_out,exist_ok=True)
    window = torch.hann_window(
        window_length=hp.de.n_fft, periodic=True, dtype=None, layout=torch.strided, device=None,
                               requires_grad=False).to(device)

    ## Module init
    buffer = BufferManager(
        n_channel=n_channel,
        n_chunk= hp.bm.n_chunk,
        n_unit = hp.bm.n_unit,
        max_unit= hp.bm.max_unit,
        pre_unit=hp.bm.pre_unit
    )

    # Denoising
    denoising = UNet()
    denoising.load_state_dict(torch.load(
        dir_src+"/SpeechEnhancement/Unet.pt",
        map_location=device))
    denoising.to(device)
    denoising.eval()

    # VAD
    vad = VAD(
        thr_energy =hp.vad.thr_energy,
        fs=hp.de.sr,
        n_fft=hp.de.n_fft,
        n_hop=hp.de.n_hop,
        thr_up=hp.vad.thr_up,
        thr_down = hp.vad.thr_down
    )

    # DSP
    dsp = DSP(path=dir_src+"/DSP/libDSP.so",epsiBF=hp.dsp.epsiBF,epsiSVE=hp.dsp.epsiSVE)

    # Gender Classification
    gender = Gender(
        nOut=hp.gd.nOut,
        device=device,
        max_frames=hp.gd.max_frames
        ,model=hp.gd.model,
        version=hp.gd.version,
        n_class=hp.gd.n_class,
        thr = hp.gd.thr,
        sr = hp.gd.sr
        )
    
    total_score = 0

    list_target = glob.glob(os.path.join("eval","*.wav"))

    for path in tqdm(list_target): 
        label = eval(path,hp,buffer,window,denoising,vad,dsp,gender,device)

        name_audio = path.split("/")[-1]
        id_audio = name_audio.split(".")[0]

        pkl_output = open("{}/{}.pkl".format(dir_out,id_audio), 'wb')
        pickle.dump(label, pkl_output)
        pkl_output.close()



 
