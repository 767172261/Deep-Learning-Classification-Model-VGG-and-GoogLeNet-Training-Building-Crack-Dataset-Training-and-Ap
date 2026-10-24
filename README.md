深度学习分类模型 VGG 和 GoogLeNet训练建筑裂缝数据集 建筑裂缝数据集的训练及应用，4万张，如何用于分类任务
@[toc]

**数据集描述：**
建筑裂缝数据集，4万张，用于分类任务dataset

数据集说明：
可用于深度学习分类模型（vgg goolenet系列，resnet系列）
数据集包含：
2w张混凝土裂缝图像
2w张非裂缝图像（即背景） 图像和背景共4w张

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/381595489eee4f28bac7df2188b86a38.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4f2f26e1464943d484605a574ab32032.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f9d22c584e94429a99121c0601d7bb0e.png)
建筑裂缝数据集训练深度学习分类模型（如 VGG 和 GoogLeNet），从环境设置、数据准备、模型选择和配置、训练过程、评估以及预测。代码示例，仅供参考学习。

### 一、环境设置

首先，确保环境中安装了必要的库。使用以下命令来安装这些依赖：

```bash
pip install torch torchvision matplotlib scikit-learn opencv-python tqdm
```

### 二、数据准备

假设你的数据集组织如下：

```
crack_dataset/
├── train/
│   ├── crack/
│   └── no_crack/
├── val/
│   ├── crack/
│   └── no_crack/
└── test/
    ├── crack/
    └── no_crack/
```

每个文件夹下分别存放2万张混凝土裂缝图像和2万张非裂缝图像。利用`torchvision.datasets.ImageFolder`来加载这些数据。

### 三、数据加载器

接下来，将定义数据加载器，用于加载并预处理图像数据：

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_data_loader(data_dir, batch_size=32, shuffle=True):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = get_data_loader('crack_dataset/train')
val_loader = get_data_loader('crack_dataset/val', shuffle=False)
test_loader = get_data_loader('crack_dataset/test', shuffle=False)
```

### 四、模型定义

从`torchvision.models`中导入预训练的VGG和GoogLeNet模型，并修改最后一层以适应我们的二分类任务。

#### VGG模型

```python
import torch.nn as nn
from torchvision import models

def vgg_model():
    model = models.vgg16(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model
```

#### GoogLeNet模型

```python
def googlenet_model():
    model = models.googlenet(pretrained=True)
    for param in model.inception3a.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
```

### 五、训练过程

定义训练和验证函数：

```python
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=25):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')
        
        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f'Validation Accuracy: {(100 * correct / total):.2f}%')

# 使用VGG模型进行训练
model_vgg = vgg_model()
train_model(model_vgg, train_loader, val_loader)

# 使用GoogLeNet模型进行训练
model_googlenet = googlenet_model()
train_model(model_googlenet, train_loader, val_loader)
```

加入**学习率衰减**和**早停法（Early Stopping）**，我们可以使用 PyTorch 提供的一些工具和自定义回调机制来实现这些功能。如何在之前的 VGG 或 GoogLeNet 模型训练过程中添加这些改进。

### 学习率衰减

学习率衰减可以帮助模型在训练后期更加稳定地收敛。PyTorch 提供了 `torch.optim.lr_scheduler` 模块来实现这一点。

#### 实现示例

```python
from torch.optim.lr_scheduler import StepLR

def train_model_with_lr_decay(model, train_loader, val_loader, num_epochs=25):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 使用 StepLR 调度器，每 7 个 epoch 将学习率乘以 0.1
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    trigger_times = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader.dataset)
            accuracy = 100 * correct / total
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    return
            
        # Adjust learning rate
        scheduler.step()

# 使用VGG模型进行训练
model_vgg = vgg_model()
train_model_with_lr_decay(model_vgg, train_loader, val_loader)

# 使用GoogLeNet模型进行训练
model_googlenet = googlenet_model()
train_model_with_lr_decay(model_googlenet, train_loader, val_loader)
```

### 早停法（Early Stopping）

早停法是一种防止过拟合的技术，它监控验证集上的损失或准确率，并在性能不再提升时提前终止训练。在上面的代码中已经包含了早停法的逻辑：

- 我们维护了一个 `best_val_loss` 变量来跟踪到目前为止最好的验证集损失。
- `trigger_times` 记录连续没有改进的次数。
- 如果连续 `patience` 次数都没有看到更好的结果，则停止训练。

### 其他学习率调度策略

除了 `StepLR`，PyTorch 还支持多种其他的学习率调度策略，例如：

- **ReduceLROnPlateau**: 当某个指标停止改进时降低学习率。
  
  ```python
  from torch.optim.lr_scheduler import ReduceLROnPlateau

  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
  
  # 在每个 epoch 结束后调用 scheduler.step(validation_loss)
  scheduler.step(val_loss)
  ```

- **CosineAnnealingLR**: 使用余弦退火策略调整学习率。
  
  ```python
  from torch.optim.lr_scheduler import CosineAnnealingLR

  scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
  ```


