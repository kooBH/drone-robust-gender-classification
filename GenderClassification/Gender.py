import sys, time, os, argparse
import yaml
import numpy as np
import torch
from glob import glob
import zipfile
import warnings
import datetime
import soundfile
from tqdm.auto import tqdm

dir_src = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(dir_src)
from model.aamsoftmax import LossFunction

warnings.simplefilter("ignore")

import librosa

class Gender():
    def __init__(self,nOut,device,max_frames,model,version,n_class,sr=16000,thr=8):
        n_gpus = torch.cuda.device_count()

        #print('Python Version:', sys.version)
        #print('PyTorch Version:', torch.__version__)
        #print('Number of GPUs:', torch.cuda.device_count())

        self.nOut = nOut
        self.device = device
        self.max_frames = max_frames
        self.thr = thr
        self.sr = sr
        self.n_class = n_class

        print("Gender | version {} | n_class {} | max_frams {}".format(version,n_class,max_frames))

        """  Conversion
        state = torch.load("smaller_300.model", map_location="cpu")
        state2 = {}
        for key,value in state.items() :
            if "__L__." in key :
                continue
            state2[key.replace("__S__.","")]=value
        torch.save(state2,"s300.pt")
        """
        margin = 0.1
        if model == "ResNet": 
            from model.ResNetSE34L import ResNetSE
            self.m = ResNetSE(nOut).to(device)
            self.m.load_state_dict(torch.load(dir_src+"chkpt/resnet_{}_S.pt".format(version),map_location=device))

            self.p = LossFunction(nOut,n_class,margin=margin).to(device)
            self.p.load_state_dict(torch.load(dir_src+"chkpt/resnet_{}_L.pt".format(version),map_location=device))
        elif model == "ResNetV2": 
            from model.ResNetSE34V2 import ResNetSEV2
            self.m = ResNetSEV2(nOut).to(device)
            self.m.load_state_dict(torch.load(dir_src+"chkpt/resnetv2_{}_S.pt".format(version),map_location=device))

            self.p = LossFunction(nOut,n_class,margin=margin).to(device)
            self.p.load_state_dict(torch.load(dir_src+"chkpt/resnetv2_{}_L.pt".format(version),map_location=device))
        elif model == "RawNet":
    
            import RawNet3.RawNet3 as RawNet3
            self.m  = (RawNet3.MainModel(
                nOut=nOut,
                encoder_type="ECA",
                sinc_stride=10,
                max_frame = max_frames,
                sr=sr
                )).to(device)
            self.m.load_state_dict(torch.load(dir_src+"chkpt/rawnet_{}_S.pt".format(version),map_location=device))

            self.p = LossFunction(nOut,n_class,margin=0.1).to(device)
            self.p.load_state_dict(torch.load(dir_src+"chkpt/rawnet_{}_L.pt".format(version),map_location=device))
        else : 
            raise Exception("몰?루?::Unknown model : {}".format(model))
        self.m.eval()
        self.p.eval()

        self.label = torch.zeros(10,dtype=torch.int64).to(device)

        self.n_per_speaker=1

    def ensegment(self, audio, max_frames, evalmode=True, num_eval=10):

        # Maximum audio length
        max_audio = max_frames * int(self.sr/100) + 240

        audio = audio/np.max(np.abs(audio))
        
        audiosize = audio.shape[0]
        if audiosize <= max_audio:
            shortage    = max_audio - audiosize + 1 
            audio       = np.pad(audio, (0, shortage), 'wrap')
            audiosize   = audio.shape[0]

        if evalmode:
            startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
        else:
            startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
        
        feats = []
        if evalmode and max_frames == 0:
            feats.append(audio)
        else:
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
        feat = np.stack(feats,axis=0).astype(np.float)
        
        return feat
        
    def infer_file(self,path_wav) : 
        x,sr = librosa.load(path_wav,sr=8000,mono=True)

        return self.infer(x)

    def infer(self,audio):
        with torch.no_grad():
            feat = self.ensegment(audio, self.max_frames)

            feat= torch.from_numpy(feat).float()
            feat = feat.to(self.device)

            feat= feat.reshape(-1, feat.size()[-1])
            embed = self.m(feat)
            embed = embed.reshape(self.n_per_speaker, -1, embed.size()[-1]).transpose(1, 0).squeeze(1)

            loss,prec1,output,pred = self.p(embed,self.label)

            # [10, 512]
            #print(pred.shape)


            pred_score = [0,0,0,0]
            for i in range(len(pred)):
                pred_score[int(pred[i])] += 1

            #print(pred_score)

            if max(pred_score) < self.thr:
                prediction = -1
            else :
                prediction = pred_score.index(max(pred_score))

            return prediction

"""
    1. eval set : C F 
    2. rawnet : 
    3. 
    4. DCUNET->VAD :
        - big pic
    5. [delay]2021 -> work
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "SpeakerNet")
    parser.add_argument('--max_frames',     type=int,   default=400,    help='Input length to the network for training')
    parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
    parser.add_argument('--dir_in', "-i",          type=str, required=True)
    parser.add_argument('--model', "-m",          type=str, required=True)
    parser.add_argument('--version', "-v",          type=str, required=True)
    parser.add_argument('--device',          type=str, required=False,default="cuda:1")
    parser.add_argument('--n_class', "-n",          type=int, required=False,default=3)
    args = parser.parse_args()

    device =args.device
    n_class = args.n_class


    m = Gender(nOut=args.nOut,device=device,max_frames=args.max_frames,model=args.model,version=args.version,n_class=n_class)

    list_target = glob(os.path.join(args.dir_in,"**","*.wav"))

    # [n_class][T, n_total, n_false]
    score = np.zeros((n_class,2),np.int32)
    metric = 0

    for path in tqdm(list_target) : 
        category = path.split("/")[-2]

        if n_class !=4 and category == "N" : 
            continue
        pred = m.infer_file(path)
        if category == "N" :
            if pred == 3 :
                metric += 0 
                score[3,0]+=1
            else : 
                metric -=0.25
            score[3,1]+=1
        elif category == "M" : 
            if pred == 2 :
                score[0,0]+=1
                metric +=0.25
            else : 
                metric -=0.25
            score[0,1]+=1
        elif category == "F" : 
            if pred == 1 :
                score[1,0]+=1
                metric +=0.25
            else : 
                metric -=0.25
            score[1,1]+=1
        elif category == "C" : 
            if pred == 0 :
                metric +=0.5
                score[2,0]+=1
            else : 
                metric -=0.5
            score[2,1]+=1


    precision = np.zeros(n_class)
    recall = np.zeros(n_class)

    print_csv = True

    if print_csv : 
        acc = score[:,0]/score[:,1]
        if n_class == 4 :
            print("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(args.dir_in,args.model+args.version,np.sum(score[:,0])/np.sum(score[:,1]),metric,acc[0],acc[1],acc[2],acc[3]  ))
        else :
            print("{},{},{},{},{},{},{}".format(args.dir_in,args.model+args.version,np.sum(score[:,0])/np.sum(score[:,1]),metric,acc[0],acc[1],acc[2]  ))
    else : 
        print("==== {} |{} ====".format(args.model+args.version,args.dir_in))
        acc = score[:,0]/score[:,1]
        print("acc M : {}".format(acc[0]))
        print("acc F : {}".format(acc[1]))
        print("acc C : {}".format(acc[2]))
        if n_class == 4 :
            print("acc N : {}".format(acc[3]))
        print("-----------")
        print("acc   : {}".format(np.sum(score[:,0])/np.sum(score[:,1])))
        print("score : {}".format(metric))
