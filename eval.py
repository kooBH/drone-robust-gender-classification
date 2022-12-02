import torch
import torch.nn

import numpy
import librosa as rs
import json

import pickle,os,glob
from tqdm.auto import tqdm


"""
s(Substitutions) : 제출한 시간대에 정답 클래스가 아닌 다른 클래스 발화가 있을 경우
D(Deletions) : 정답 발화 시간대에 제출된 답안이 없을 경우
I(Insertions) : 제출한 클래스가 정답 구간에 해당 되지 않고 다른 클래스의 발화가 없을 경우

* 한 정답 구간에서 여러 개의 예측이 있을 경우 가장 먼저 예측한 것을 정답으로 하고 그 다음번 예측은 S, I로 분류함
"""


def ER(cls,GT,outputs):
    S = 0
    D = 0
    I = 0
    N = 0

    set_GT = []
    set_out = []

    ## Init 
    for idx in range(len(GT)):
        c_GT = {"class":cls}
        c_GT["ts"] = GT[idx]
        c_GT["D"]=1
        set_GT.append(c_GT)

    for idx2 in range(len(outputs)) : 
        outputs[idx2]["SI"] = 0
        c_out = outputs[idx2]
        set_out.append(c_out)

    # Cal
    for i in range(len(set_GT)) : 
        c_GT = set_GT[i]
        for j in range(len(set_out)) :
            c_out = set_out[j]
            # timestamp of label
            t = (c_out["ts"][0]+c_out["ts"][1])/2

            #print("{} | {} {} | {} {}".format(t,c_GT["ts"][0],c_GT["ts"][1],c_GT["class"],c_out["class"]))

            if t >= c_GT["ts"][0] and t <= c_GT["ts"][1] : 
                #print("-> {} | {} | {}".format(t,c_out["class"],c_out["SI"]))
                for c_cls in c_out["class"] : 
                    #print("{} {}".format(c_GT["class"],c_cls))
                    if c_GT["class"] == c_cls : 
                        # Insertion
                        if c_GT["D"] == 0 : 
                            c_out["SI"]+=1
                        # Good
                        else : 
                            c_GT["D"]=0
                    # substitution
                    else :
                        c_out["SI"]+=1
                #print("{} | {}".format(c_GT["D"],c_out["SI"]))

    SI = 0 
    for i in range(len(set_out)):
        SI += set_out[i]["SI"]
    D = 0
    for i in range(len(set_GT)):
        D += set_GT[i]["D"]

    N = len(set_GT)

    # ER
    #print(set_GT)
    #print(set_out)

    #print("S+I : {}".format(SI))
    #print("D   : {}".format(D))
    #print("N   : {}".format(N))

    ER = (S+D+I)/N
    return ER


if __name__ == "__main__" : 
    list_target = glob.glob(os.path.join("eval","*.wav"))

    f_GT = open("eval/GT.json","r")
    GT = json.load(f_GT)
    f_GT.close()

    ER_2021 = 0
    ER_2022 = 0

    for path_audio in tqdm(list_target) : 

        name_audio = path_audio.split("/")[-1]
        id_audio = name_audio.split(".")[0]

        c_GT = GT[id_audio]

        f_2021 = open("output_2021/{}.pkl".format(id_audio), 'rb')
        f_2022 = open("output_2022/{}.pkl".format(id_audio), 'rb')

        ans_2021 = pickle.load(f_2021)
        ans_2022 = pickle.load(f_2022)

        f_2021.close()
        f_2022.close()
        
        #print("===   {}   ===".format(id_audio))
        #print("2021 : {:.4f}".format(ER(id_audio,c_GT,ans_2021)))
        #print("2022 : {:.4f}".format(ER(id_audio,c_GT,ans_2022)))

        ER_2021 += ER(id_audio,c_GT,ans_2021)
        ER_2022 += ER(id_audio,c_GT,ans_2022)
    ER_2021/= len(list_target)
    ER_2022/= len(list_target)

    print("ER 2021 : {:.4f}".format(ER_2021))
    print("ER 2022 : {:.4f}".format(ER_2022))

