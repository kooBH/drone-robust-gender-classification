from AGC2021.vad import vad
from AGC2021.enhance import  enhancer
from AGC2021.gender import gender

import librosa as rs
import numpy as np
from tqdm.auto import tqdm
import pickle,os,glob

device = "cuda:0"
path_chkpt_unet = "AGC2021/Unet_v2.pth"

print("Init Speech Enhancement Model")
se = enhancer(path_chkpt = path_chkpt_unet,device = device)
print("Init Gender Classification Model")
gd = gender("./AGC2021/sub1_1_89.pt")

list_target = glob.glob(os.path.join("eval","*.wav"))

os.makedirs("output_2021",exist_ok=True)

for path_audio in tqdm(list_target) : 

    name_audio = path_audio.split("/")[-1]
    id_audio = name_audio.split(".")[0]

    audio = rs.load(path_audio, sr=16000)[0]
    audio_se = se.process(path_audio)
    label_utt = vad(audio_se)
    outputs = gd.process(audio,audio_se,label_utt)

    pkl_output = open("output_2021/{}.pkl".format(id_audio), 'wb')
    pickle.dump(outputs, pkl_output)
    pkl_output.close()




