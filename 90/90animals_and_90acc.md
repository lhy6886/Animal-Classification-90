1. 
💡 Additional Libraries
 • ⏱️  time  – not actively used, but typically imported for timing utilities.
 • 📊  torch.utils.tensorboard.SummaryWriter  – for logging metrics to TensorBoard.
 • 🛑 Custom module  early_stop.EarlyStopper  – implements patience-based early stopping.
 • 🔄  torch.optim.lr_scheduler.ReduceLROnPlateau  – automatic learning-rate reduction when validation accuracy plateaus.
2. 
✨ New Functionality Compared to the Basic Version
 • 🥇 Two data splits: separate training and validation sets ( train_ds ,  test_ds ) with different transforms.
 • 🎨 Augmentation: random horizontal flip, rotation, and color jitter for training images; only resize + ToTensor for validation.
 • 🧠 Transfer learning with selective fine-tuning:
 – Loads a pretrained ResNet-152 instead of ResNet-18.
 – Freezes the backbone, unfreezes  layer3 ,  layer4 , and the new classifier ( fc ) for fine-tuning.
 • 📈 Validation loop: computes accuracy on the validation set after each epoch.
 • 🛑 Early stopping: stops training if validation accuracy does not improve for 2 consecutive epochs.
 • 🔄 Learning-rate scheduling: halves the LR when validation accuracy plateaus for 1 epoch.
 • 📊 TensorBoard logging: records training loss, training accuracy, and validation accuracy after every epoch.
 • 🔢 Dynamic class count: infers  num_classes  from the training folder ( len(os.listdir(...)) ), so the code does not hard-code 2 classes.
 • 📆 Longer training: up to 10 epochs (vs. 5 in the basic script) unless early stopping triggers.
 • ⚖️ Weight decay (L2 regularization) of 1e-4 on the optimizer.
3. 
⚠️ Key Points / Caveats
 • 📁 Make sure the  early_stop  module and its  EarlyStopper  class are in the Python path.
 • 📂 Ensure the directory structure under  data/animals90  contains  train/  and  test/  sub-folders, each with class-named sub-directories.
 • 🚀 TensorBoard must be launched manually ( tensorboard --logdir runs/exp1 ) to visualize the metrics at  http://localhost:6006 .
 • 🖥️ Because only the last layers are fine-tuned, GPU memory usage is moderate, but ResNet-152 is still heavy—verify GPU availability.




```python
import torch, torchvision, os, time
import torchvision, torch.nn as nn
from torchvision import transforms, datasets
from early_stop import EarlyStopper
from torch.utils.tensorboard import SummaryWriter

data_dir = 'data/animals90'
batch_size = 32
epochs = 10
lr = 3e-4

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


train_ds = datasets.ImageFolder('C:/baidu/Programing/cat-dog-project/data/animals90/train', transform=train_transform)
test_ds  = datasets.ImageFolder('C:/baidu/Programing/cat-dog-project/data/animals90/test',  transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet152(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
num_classes = len(os.listdir(os.path.join('data', 'animals90', 'train')))
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-4)
for name, p in model.named_parameters():
    if name.startswith('layer3') or name.startswith('layer4') or name.startswith('fc'):
        p.requires_grad = True
early_stopper = EarlyStopper(patience=2, mode='max')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=1, verbose=True)
criterion = torch.nn.CrossEntropyLoss()

writer = SummaryWriter('runs/exp1')


for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0
    correct = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    acc = correct / len(train_ds)
    print(f'Epoch {epoch}/{epochs} | loss={running_loss/len(train_loader):.4f} | acc={acc:.3f}')


    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    val_acc = correct / total
    scheduler.step(val_acc)
    print(f'Val acc: {val_acc:.3f}')
    early_stopper(val_acc)
    if early_stopper.early_stop:
        print("No improvement on the validation set, early stopping！")
        break
    global_step = epoch
    writer.add_scalar('Loss/train', running_loss / len(train_loader), global_step)
    writer.add_scalar('Accuracy/train', acc, global_step)
    writer.add_scalar('Accuracy/test', val_acc, global_step)

print('Training completed！')
print(f'Val acc: {correct/total:.3f}')
torch.save(model.state_dict(), 'best.pt')
print('The final model weights have been saved to best.pt')
writer.close()
```

early_stopper:

```python
import torch

class EarlyStopper:
    def __init__(self, patience=2, mode='max', delta=0.0):
        self.patience = patience  # tolerance count
        self.counter  = 0  # how many epochs without improvement, reset to 0 on improvement
        self.best     = None  # cache of the best metric so far; None on first run
        self.early_stop = False  # early-stop flag; set to True to break loops
        self.delta    = delta  # must exceed this threshold to be considered a real improvement, preventing jitter misjudgment
        self.mode     = mode      # 'max' means higher is better (accuracy), 'min' means lower is better (loss)

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
        elif (self.mode == 'max' and metric < self.best - self.delta) or \
             (self.mode == 'min' and metric > self.best + self.delta):
        # when mode is max (higher is better) and current metric < best - delta, no improvement
        # when mode is min (lower is better) and current metric > best + delta, no improvement
        # +- delta removes tiny fluctuations
        # mode only takes two values, max or min, chosen according to use case
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best = metric
            self.counter = 0
```

And what's next？
Is everything done？
NO！Let's make a web demo！

Here we go:

```python
import gradio as gr, torch, torchvision.transforms as T, torchvision, os, torch.nn as nn
from PIL import Image          
# •  gradio  – hosts the web UI and handles drag-and-drop / upload.
# •  torch ,  torchvision ,  nn  – load the PyTorch model and run inference.
# •  transforms as T  – image preprocessing pipeline.
# •  os  – list directories to get class names.
# •  PIL.Image  – converts the Gradio-supplied NumPy array into a PIL image for preprocessing.

num_classes = len(os.listdir(os.path.join('data', 'animals90', 'train')))
model = torchvision.models.resnet152(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('best.pt', map_location='cpu'))
model.eval()
# • Determines how many output neurons the network must have.
# • Creates a fresh ResNet-152 with a classification head matching the dataset.
# • Loads the previously saved weights ( best.pt ) onto the CPU (or GPU if changed).
# • Switches to evaluation mode to disable dropout / batch-norm updates.

tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
# • Resizes any incoming image to 224×224 pixels.
# • Converts the PIL image to a CHW tensor in the range [0, 1].##
labels = os.listdir(os.path.join('data', 'animals90', 'train'))
# • Reads the folder names under  train/  to build the ordered list of class names that match the model’s output indices.
def predict(img):

    img = Image.fromarray(img.astype('uint8'), 'RGB')
    x = tf(img).unsqueeze(0)
    probs = torch.softmax(model(x)[0], 0)
    return {labels[i]: float(p) for i, p in enumerate(probs)}
# • Receives a NumPy image from Gradio.
# • Converts it to PIL and preprocesses.
# • Runs a forward pass, applies softmax to obtain probabilities.
# • Returns a dictionary mapping each class name to its confidence value.
gr.Interface(predict, gr.Image(), gr.Label(num_top_classes=5)).launch(share=True)
# •  gr.Image()  – input widget that accepts drag-and-drop or file uploads.
# •  gr.Label(num_top_classes=5)  – output widget showing the five highest-confidence labels.
# •  .launch(share=True)  – starts a local server at  http://127.0.0.1:7860  and generates a temporary public URL ( *.gradio.live ) for external access.
```