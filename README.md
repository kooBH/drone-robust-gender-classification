# drone-robust-gender-classification

![image](https://user-images.githubusercontent.com/39723411/205220727-e9cca01c-b1f4-4afe-875c-a53c7790ca0e.png)


 For autonomous rescue drones, it is important to recognize rescue requesters. Since there are lots of noise sources in a disaster area including the drone itself, such a system also requires denoising and voice activity detection. This code provides a male-woman-child recognition system for rescue drones which is written in Python, based on Pytorch.


---

Man-Woman-Child recognition for mobile drone with [UMA-8](https://www.minidsp.com/products/usb-audio-interface/uma-8-microphone-array?gclid=CjwKCAiA-8SdBhBGEiwAWdgtcKlf8YIUjy-Bmm8vfHFDcEtS490jpcv3MwUBBPvpt2K5mIKh9NLl8BoCSooQAvD_BwE). 

## set up

### Too Large file to upload in github
[Place it into AGC2021/](https://drive.google.com/file/d/1N2NAxBDdmVgf5J8tL_hMXRNuW1TKJUk4/view?usp=sharing)

### libDSP.so  
+ DSP/ARM/libDSP.so : compiled on NVIDIA Xavier NX    
+ DSP/x86/libDSP.so : compiled on Intel CPU    

Place selected ```libDSP.so``` into DSP/libDSP.so  

## Usage

```
python run.py -d <cuda:0 or cuda:1 or cpu> -i <directory_of_input_files>  -o <directory_of_output_files>
```

see ```run.py``` for detail usage. 


# Acknowledgement
이 연구는 2022년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구의 결과물임 (No.1711152445, 인명 구조용 드론을 위한 영상/음성 인지 기술 고도화)

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.1711152445, Advanced Audio-Visual Perception for Autonomous Rescue Drones)
  
  
평가 데이터는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "위급상황 음성/음향"을 활용하여 제작된 데이터입니다. 평가 데이터 생성에 활용된 데이터는 AI 허브(aihub.or.kr)에서 다운로드 받으실 수 있습니다.
