# SGTAPose
This is an official code for CVPR 2023 paper "Robot Structure Prior Guided Temporal Attention for Camera-to-Robot Pose
Estimation from Image Sequence".

[\[PDF\]](https://arxiv.org/pdf/2307.12106.pdf) [\[Video\]](https://www.youtube.com/watch?v=5fQp-yBubZs&t=12s)
# Code Release Schedule
- [x] Installation
- [x] Dataset and Pretrained model Release
- [x] Training Code Release
- [x] Inference Code Release

# Installation
* **Install Pytorch**
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 --extra-index-url https://download.pytorch.org/whl/cu111 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
* **Install from requirements.txt**
```
pip install -r requirements.txt
```
* **Compile Deformable Convolution Networks**
```
cd sgtapose/lib/model/networks
git clone https://github.com/lbin/DCNv2.git
cd DCNv2
git checkout pytorch_xxx ###(xxx refers to different versions of torch, e.g. git checkout pytorch_1.9. You need to ensure the available DCNv2's version agrees with your torch's version)
./make.sh
```
* **Install sgtapose package**
```
python setup.py install
```
* **Edit LM Solver Path**
```
cd sgtapose/rf_tools
# In the LM.py, you have to change the path in line 10 to your own absolute path
```

# Dataset and Pretrained model
We have a big training dataset which involves two parts: franka_data_1020 and near_franka_data_1024. We train our model over the two datasets.
Further, we provide a validation dataset named as syn_test.
We can use the following command to download the dataset from the [website](https://mirrors.pku.edu.cn/dl-release/SGTAPose_CVPR2023) 
```
mkdir data
cd data
### download franka_data_1020.tar.gz from https://mirrors.pku.edu.cn/dl-release/SGTAPose_CVPR2023
### download near_franka_data_1024.tar.gz from https://mirrors.pku.edu.cn/dl-release/SGTAPose_CVPR2023
### download syn_test.zip from https://mirrors.pku.edu.cn/dl-release/SGTAPose_CVPR2023
tar -Jxvf franka_data_1020.tar.gz
tar -Jxvf near_franka_data_1024.tar.gz
unzip syn_test.zip
cd ..
```
We also need to download the real-world test dataset from the [website](https://drive.google.com/drive/folders/1LxoLOUE43jAPXSxfuTSyr7LBebkEn54v?usp=drive_link)
Specifically, we need to download these:
```
cd data
### download panda-3cam_azure.zip from https://drive.google.com/drive/folders/1LxoLOUE43jAPXSxfuTSyr7LBebkEn54v?usp=drive_link
### download panda-3cam_realsense.zip from https://drive.google.com/drive/folders/1LxoLOUE43jAPXSxfuTSyr7LBebkEn54v?usp=drive_link
### download panda-orb.zip from https://drive.google.com/drive/folders/1LxoLOUE43jAPXSxfuTSyr7LBebkEn54v?usp=drive_link
unzip panda-3cam_azure.zip
unzip panda-3can_realsense.zip
unzip panda-orb.zip
```

Besides, the pretrained model and inference results are also available in the [website](https://mirrors.pku.edu.cn/dl-release/SGTAPose_CVPR2023). We can 
use the following command to download them.
```
### download pretrained_model.zip from https://mirrors.pku.edu.cn/dl-release/SGTAPose_CVPR2023
unzip pretrained_model.zip
```
Since SGTAPose relies on the image sequences as inputs, so we manually split the real-world dataset into several videos. We can download the split information as follows:
```
### download dream_real_info.zip from https://mirrors.pku.edu.cn/dl-release/SGTAPose_CVPR2023
unzip dream_real_info.zip
```
After downloading the data, pretrained model, and split information, we can see the code structure as follows:
```
SGTAPose
--sgtapose
--data
  --franka_data_1020
  --near_franka_data_1024
  --panda-orb
  --panda-3cam_azure
  --panda-3cam_realsense
--pretrained_model
--dream_real_info
```


# Training Code
Here we offer the training code in parallel settings. Before you start training, you need to change some paths regarding dataset in the sgtapose/scripts/train_scripts.

Run the following script to train the model.
```
cd sgtapose
bash scripts/train_scripts.sh
```
