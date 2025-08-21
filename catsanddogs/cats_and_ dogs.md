🐱🐶 Build Your First Cat vs. Dog Classifier — Zero to Hero in 15 min!

---

1️⃣ Grab the Dataset  
🔗 [Kaggle Cats vs. Dogs Small](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

Unzip it → you’ll get `test/` and `train/` folders.

---

2️⃣ GPU Check-Up 🖥️➡️🚀  
Open Anaconda Prompt and type:

```bash
nvidia-smi
```

You’ll see something like:

```
Driver Version: 551.xx   CUDA Version: 12.x
```

- If Driver < 450.xx → update to the latest driver first.  
- If Driver ≥ 450.xx → you’re good to install CUDA.

---

3️⃣ Install CUDA Toolkit  
🔗 [CUDA Toolkit 12.4.1](https://developer.nvidia.com/cuda-toolkit-archive)  
- OS: Windows → x86_64 → 11/10 → exe(local)  
- Install options: Custom → tick only CUDA / Development → finish → reboot.

Make sure these paths are in System Environment Variables:

```
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp
```

---

4️⃣ Install GPU PyTorch 🐍⚡  
In the same Anaconda Prompt:

```bash
conda activate py39
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

> CUDA 12.1 vs 12.4 are compatible — no worries!

---

5️⃣ Verify GPU Setup ✅  

```python
import torch
print(torch.__version__)          # 2.x.x+cu121
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # e.g., GeForce RTX 3060
```

If you see `True` plus your GPU name, party time! 🎉

---

6️⃣ Folder Structure 📁  
Create:

```
cat_and_dog_Classification/
├─ data/
│   ├─ train/
│   │   ├─ cat/
│   │   └─ dog/
│   └─ test/
│       ├─ cat/
│       └─ dog/
├─ models/
└─ train.py
```

Copy the downloaded images into the correct sub-folders.

---

7️⃣ Train 🏃‍♂️💨  
Open Anaconda Prompt, activate `py39`, and navigate to the project:

```bash
cd D:\Programing\cat_and_dog_Classification
```

Minimal `train.py` (GPU-ready):

```python
import torch, torchvision, os
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch import nn, optim
#import libraries we need to use

data_dir = 'data' #this is your path of images,it shuold have two folders, test and train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) #adjust size of every picture, so that computer can identify them

train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
test_ds  = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size=32, shuffle=False)
# ds is dataset, it has pics and label
# dl is dataloader used to package the dataset into batches and spitting out a batch at a time during iteration
# shuffle=True, used during training, the data order is randomly shuffled in each epoch to prevent the model from learning order bias

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Load a ResNet-18 network pre-trained on ImageNet.
# Replace its final fully-connected layer with a new one having 2 output units (for cat vs dog).
# Move the entire model to the GPU (or CPU) specified by device.
# Use cross-entropy loss for multi-class classification.
# Optimize all parameters with the Adam optimizer and a learning rate of 1e-4.

for epoch in range(5):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} done!')

torch.save(model.state_dict(), 'models/cat_dog_gpu.pth')
print('🎉 Training finished & model saved!')
# For 5 epochs, loop through the training set in mini-batches:
# move each batch of images (x) and labels (y) to the GPU,
# zero the gradients,
# forward-pass to get predictions,
# compute cross-entropy loss,
# back-propagate gradients,
# update weights with the optimizer.
# After every epoch, print the last batch’s loss.
# Finally, save the learned weights to  models/cat_dog_gpu.pth 
```

---

8️⃣ Next Steps 🚀  
- Increase epochs, data augmentation, learning-rate schedules…  
- Try EfficientNet, ResNet50, or ResNet80 for even better accuracy!

Happy coding!