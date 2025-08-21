First thing first, we need anaconda.

🔗🔗🔗：https://www.anaconda.com/download 

> Why Anaconda?  
> One sentence: it saves time and headaches for beginners.

Use the link to download, download the Individual Edition for FREE

🚨🚨🚨WARNINGS:
1. Select “Install for me only” 
2. Avoid installing on your system drive (usually C:)  
3. DON'T ADD PATH AUTOMATICALLY

🛣️🛣️🛣️Path in System:
1. Press Win+S on your keyboard  
2. Search for "Environment Variables"
3. Edit Path(double click path)
4. Add 3 entries


Add the following three entries to your Path variable
D:\Anaconda3
D:\Anaconda3\Scripts
D:\Anaconda3\Library\bin

Save and open Anaconda prompt
type:
conda --version
🎉🎉🎉if you see conda 24.x.x, congrats!!! Let's move on！！！

Now we need a virtual python environment, py39
We choose Python 3.9 because it’s currently the stable baseline for most deep-learning libraries.

1. "conda create -n py39 python=3.9", create py39 environment
2. "conda activate py39", active py39 environment in anaconda prompt
3. "conda install pytorch torchvision torchaudio cpuonly -c pytorch", install libraries

Next, let's verify, is everything done?
1. "conda activate py39", active py39 environment in anaconda prompt
2. copy this part of code,  conda will display the versions of Python, PyTorch, and TorchVision
``` python
python
import sys, torch, torchvision
print("Python version :", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("TorchVision   :", torchvision.__version__)
```

like this [version](library/version.jpg)

🎉🎉🎉 congrats！！！You have done all preperation steps！let's move on！！！Try to write our first train.py