




## Datasets / Benchmarks

|  Date  | Keywords | Website | Paper | Scale |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2024-06 |     Facial,DM     | [code](https://github.com/Purdue-M2/AI-Face-FairnessBench) | [AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark](https://arxiv.org/pdf/2406.00783)  |   |
| 2024-01 |     Facial,DM     | [code](https://github.com/xaCheng1996/DiFF) | [Diffusion Facial Forgery Detection](https://arxiv.org/pdf/2401.15859)  |   500K (13) |
| 2023-06 |     Gen,DM,GAN     | [GenImage](https://github.com/GenImage-Dataset/GenImage) | [GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image ](https://arxiv.org/pdf/2306.08571)  |   |
| 2019-12 |     Gen,GAN     | [ForenSynths](https://github.com/PeterWang512/CNNDetection) | [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035)  |   |


| 2019-12 |     Gen,GAN     | [DIF]() | [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035)  |   |
| 2019-12 |     Gen,GAN     | [DiffusionForensics]() | [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035)  |   |
| 2019-12 |     Gen,GAN     | [GANGen-Detection]() | [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035)  |   |
| 2019-12 |     Gen,DM     | [Ojha]() | [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035)  |   |
| 2019-12 |     Gen,DM     | [AntifakePrompt]() | [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035)  |   |









https://www.dfad.unimore.it/challenge/  [submission/results](https://benchmarks.elsa-ai.eu/?ch=3)
https://huggingface.co/datasets/elsaEU/ELSA_D3

https://www.kaggle.com/competitions/multi-ffdi/data


## Papers

### Image/Video
[ECCV2024] Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection https://github.com/mever-team/rine


FakeBench: Uncover the Achilles’ Heels of Fake Images with
Large Multimodal Models

[CVPR2024W] Raising the Bar of AI-generated Image Detection with CLIP [paper](https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Cozzolino_Raising_the_Bar_of_AI-generated_Image_Detection_with_CLIP_CVPRW_2024_paper.pdf) [code](https://github.com/grip-unina/ClipBased-SyntheticImageDetection/)   
[*CVPRW2024*] Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks [paper](https://arxiv.org/abs/2406.04932) [code](https://github.com/fedeloper/binary_deepfake_detection) 


Diffusion Noise Feature: Accurate and Fast
Generated Image Detection https://arxiv.org/pdf/2312.02625 https://github.com/YichiCS/DNF


Detecting AI-Generated Text: Factors Influencing
Detectability with Current Methods https://arxiv.org/pdf/2406.15583




### Audio

Enhancing Generalization in Audio Deepfake Detection: A Neural Collapse based Sampling and Training Approach https://arxiv.org/pdf/2404.13008 https://arxiv.org/abs/2404.13008v1
## Challenging
https://www.dfad.unimore.it/challenge/

https://www.atecup.cn/deepfake


import os
import shutil

txt_file_path = 'real_coco.txt'
source_directory = '/home/jwang/ybwork/data/DFBenchmark/our/train/0_real'
destination_directory = './real'

os.makedirs(destination_directory, exist_ok=True)

with open(txt_file_path, 'r') as file:
    filenames = file.readlines()

filenames = [filename.strip() for filename in filenames]

for filename in filenames:
    filename_dir = filename.split('_')[0]
    filename_name = filename.split('_')[1]
    source_file = os.path.join(source_directory, filename_dir, filename_name)
    destination_file = os.path.join(destination_directory, filename)
    if os.path.isfile(source_file):
        shutil.copy2(source_file, destination_file)
        print(f'Copied: {source_file} to {destination_file}')
    else:
        print(f'File not found: {source_file}')

print('All files have been processed.')