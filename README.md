# SGTAPose
This is an official code for CVPR 2023 paper "Robot Structure Prior Guided Temporal Attention for Camera-to-Robot Pose
Estimation from Image Sequence".

[\[PDF\]](https://arxiv.org/pdf/2307.12106.pdf) [\[Video\]](https://www.youtube.com/watch?v=5fQp-yBubZs&t=12s)
# Code Release Schedule
- [x] Installation
- [ ] Training Dataset Release
- [x] Training Code Release
- [ ] Inference Code Release

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
cd SGTAPose/lib/model/networks
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
cd SGTAPose/rf_tools
# In the LM.py, you have to change the path in line 10 to your own absolute path
```

# Training Dataset

# Training Code
* **Training in parallel settings**:
Here we offer the training code in parallel settings. Run the following script to train the model.
```
cd sgtapose
bash scripts/train_scripts.sh
```
