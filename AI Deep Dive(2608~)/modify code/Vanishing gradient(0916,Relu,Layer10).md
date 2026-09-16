```python
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks')
from multiclass_functions1 import *
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    cpu
    


```python
# for random seed
# 랜덤한 값들이 규칙성을 가지고 나오게끔 조정해주는 코드(같은 패턴으로 랜던한값)
# ex) 1-3-5-2-5(seed 적용) 이후 다시 랜덤값 적용 시 같은 1-3-5-2-5(seed 적용)
# ex) 2-5-6-5-1-1(seed 미적용) 이후 다시 랜덤값 적용 시 다른 1-6-2-5-2(seed 미적용)
# 일정한 비교를 위한 랜덤 적용
import numpy as np
import random
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
```


```python
BATCH_SIZE = 256                        # 256개씩 꺼내서
LR = 1e-3                               # LR는 0.001(1 * 10 에 -3제곱 = 0.001)로 설정
EPOCH = 100
NoB = 10                                 # 레이어의 갯수
NoC = 1                                 # 채널의 갯수(노드의 갯수)
criterion = nn.CrossEntropyLoss()       # 분류문제이므로 CE 사용
activation = "ReLU"
new_model_train = False
video_save = False
model_type = f"{activation}{NoB}C{NoC}"
save_model_path = f"/content/drive/MyDrive/Colab Notebooks/results/VG/{model_type}_VG_MNIST.pt"
save_video_path = f"/content/drive/MyDrive/Colab Notebooks/results/VG/{model_type}.mp4"
```


```python
transform = transforms.ToTensor()
train_DS = datasets.MNIST(root = '/content/drive/MyDrive/Colab Notebooks/data', train=True, download=True, transform=transform)
test_DS = datasets.MNIST(root = '/content/drive/MyDrive/Colab Notebooks/data', train=False, download=True, transform=transform)
train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True)
```


```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        if activation == "Sigmoid":
            self.conv_block = nn.Sequential(
                nn.Conv2d(1,NoC,3, bias=False, padding=1),                                                            # CNN은 nn.Linear=(Fully connected Layer) 대신 nn.Conv2d(Convolution을 사용하는 Layer)를 사용
                nn.Sigmoid(),                                                                                         # Sigmoid 시험
                *[i for j in range(NoB-1) for i in [nn.Conv2d(NoC,NoC,3, bias=False, padding=1), nn.Sigmoid()]]       # Conv - sigmoid - Conv - sigmoid 계속 반복하여 층을 깊게 쌓음(if NoB가 작으면 층을 몇개 안쌓고, 크면 여러 겹 = conv_block)
                )
        elif activation == "BNSigmoid":
            self.conv_block = nn.Sequential(
                nn.Conv2d(1,NoC,3, bias=False, padding=1),
                nn.BatchNorm2d(NoC),
                nn.Sigmoid(),
                *[i for j in range(NoB-1) for i in [nn.Conv2d(NoC,NoC,3, bias=False, padding=1), nn.BatchNorm2d(NoC), nn.Sigmoid()]]
                )
        elif activation == "ReLU":
            self.conv_block = nn.Sequential(
                nn.Conv2d(1,NoC,3, bias=False, padding=1),
                nn.ReLU(),
                *[i for j in range(NoB-1) for i in [nn.Conv2d(NoC,NoC,3, bias=False, padding=1), nn.ReLU()]]
                )
        elif activation == "BNReLU":
            self.conv_block = nn.Sequential(
                nn.Conv2d(1,NoC,3, bias=False, padding=1),
                nn.BatchNorm2d(NoC),
                nn.ReLU(),
                *[i for j in range(NoB-1) for i in [nn.Conv2d(NoC,NoC,3, bias=False, padding=1), nn.BatchNorm2d(NoC), nn.ReLU()]]
                )
        # self.fc = nn.Linear(NoC*(28-2*NoB)*(28-2*NoB),10, bias=False)
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(NoC*7*7,10, bias=False)                                                                   # 7*7,10 (10 -> 10개의 노드로 끝냄. ex. 다중분류에서 개, 고양이, 소 면 3 / 현재는 mnnist 숫자 데이터셋이므로 10개)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.maxpool1(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
```


```python
model=CNN().to(DEVICE)
print(model)
x_batch, _ = next(iter(train_DL))
print(model(x_batch.to(DEVICE)).shape)
```

    CNN(
      (conv_block): Sequential(
        (0): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU()
        (2): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (3): ReLU()
        (4): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): ReLU()
        (6): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (7): ReLU()
        (8): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (9): ReLU()
        (10): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): ReLU()
        (12): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (13): ReLU()
        (14): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (15): ReLU()
        (16): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): ReLU()
        (18): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (19): ReLU()
      )
      (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc): Linear(in_features=49, out_features=10, bias=False)
    )
    torch.Size([256, 10])
    


```python
# Train함수 정의, 모델의 학습
def Train(model, train_DL, criterion):
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_history=[]
    acc_history=[]
    mean_grad_history=[]
    mean_weight_history=[]

    NoT=len(train_DL.dataset) # The number of training data

    model.train() # train mode로 전환
    for ep in range(EPOCH):
        rloss = 0 # running loss
        rcorrect = 0 # running correct
        rgrad = torch.zeros(NoB+1) # +1 : 마지막 fc 까지
        rweight = torch.zeros(NoB+1)
        for x_batch, y_batch in train_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # loss
            loss = criterion(y_hat, y_batch)
            # update
            optimizer.zero_grad() # gradient 누적을 막기 위한 초기화
            loss.backward() # backpropagation
            optimizer.step() # weight update
            # loss accumulation
            loss_b = loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE를 곱하면 마지막 18개도 32개를 곱하니까..
            rloss += loss_b
            # accuracy accumulation
            pred = y_hat.argmax(dim=1)
            corrects_b = torch.sum(pred == y_batch).item()
            rcorrect += corrects_b
            # grad and weight (batch)
            with torch.no_grad():
                wlist = [m.weight for m in model.modules() if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear)]
                rgrad += torch.tensor([w.grad.abs().sum() * x_batch.shape[0] / w.numel() for w in wlist])  # w.grad.abs() -> abs값에 gradient를 취하고 .sum() -> 그 값들을 더함
                rweight += torch.tensor([w.abs().sum() / w.numel() for w in wlist]) # w.abs() -> weight값에 절대값을 취하고 .sum() -> 그 값들을 더함
                # print(rweight)
        #         break
        # break
        # grad and weight (epoch)
        mean_grad_history += [rgrad/NoT] # 위 값들의 평균 (NoT=train 데이터 갯수)
        mean_weight_history += [rweight/len(train_DL)] # len(train_DL) = Batch 가 몇개인지로 나눠
        # print loss
        loss_e = rloss/NoT # epoch loss
        accuracy_e = rcorrect/NoT * 100
        loss_history += [loss_e]
        acc_history += [accuracy_e]
        print(f"Epoch: {ep+1}, train loss: {round(loss_e,3)}, train accuracy: {round(accuracy_e,1)} %")
        print("-"*20)

    return loss_history, acc_history, mean_grad_history, mean_weight_history
```


```python
if new_model_train:
    loss_history, acc_history, mean_grad_history, mean_weight_history  = Train(model, train_DL, criterion)

    torch.save({"model":model,
                "BATCH_SIZE":BATCH_SIZE,
                "LR":LR,
                "EPOCH":EPOCH,
                "NoB":NoB,
                "NoC":NoC,
                "loss_history":loss_history,
                "acc_history":acc_history,
                "activation":activation,
                "mean_grad_history":mean_grad_history,
                "mean_weight_history":mean_weight_history}, save_model_path)
```


```python
loaded = torch.load(save_model_path, map_location=DEVICE, weights_only=False)
load_model = loaded["model"]
loss_history = loaded["loss_history"]
acc_history = loaded["acc_history"]
mean_grad_history = loaded["mean_grad_history"]
mean_weight_history = loaded["mean_weight_history"]
BATCH_SIZE = loaded["BATCH_SIZE"]
LR = loaded["LR"]
EPOCH = loaded["EPOCH"]
NoB = loaded["NoB"]
NoC = loaded["NoC"]
activation = loaded["activation"]

test_acc = Test(load_model, test_DL)
print(count_params(load_model))
Test_plot(load_model, test_DL)
```

    Test accuracy: 9534/10000 (95.3 %)
    580
    


    
![png](Vanishing%20gradient%280916%2CRelu%2CLayer10%29_files/Vanishing%20gradient%280916%2CRelu%2CLayer10%29_8_1.png)
    



```python
# 각 Layer 별 평균 절대값 시각화
# Vanishing Gradient가 일어난다면, conv1 <- conv2layer <- fc 의 막대가 점점 작아질 것이다.(fc막대가 가장 높고 왼쪽으로 가면서 점점 낮아지는 막대 형성)

fig = plt.figure(figsize=[15, 9])
ymax_grad = max([i.max().item() for i in mean_grad_history])*1.1
ymax_weight = max([i.max().item() for i in mean_weight_history])*1.1
i = 1

plt.suptitle(f"Epoch: {range(1,EPOCH+1)[i]}, Activation = {activation}, # of channels for conv layer = {NoC}")

plt.subplot(2,2,1)
x_axis = [f"conv{i}" for i in range(1,NoB+1)]+["fc"]
plt.bar(x_axis,mean_grad_history[i].cpu(), width=0.8)
plt.ylim([0,ymax_grad])
plt.xlabel("layer")
plt.ylabel("mean absolute value")
plt.title("gradient")

plt.subplot(2,2,2)
x_axis = [f"conv{i}" for i in range(1,NoB+1)]+["fc"]
plt.bar(x_axis,mean_weight_history[i].cpu(), width=0.8)
plt.ylim([0,ymax_weight])
plt.xlabel("layer")
plt.ylabel("mean absolute value")
plt.title("weight")

ax1 = plt.subplot(2,1,2)
ax2 = ax1.twinx()
p1=ax1.plot(range(1,EPOCH+1),loss_history,'r')
p2=ax2.plot(range(1,EPOCH+1),acc_history)
ax1.set_xlim([-5,EPOCH+5])
ax1.set_ylim([min(loss_history)*0.7, max(loss_history)*1.1])
ax2.set_ylim([0, 100])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("accuracy")
ax1.set_title(f"Train Loss (Final Test accuracy = {test_acc} % with BS = {BATCH_SIZE} & LR = {LR})")
ax1.grid()
plt.legend(p1+p2,["loss", "accuracy"], loc="right")

plt.tight_layout()
```


    
![png](Vanishing%20gradient%280916%2CRelu%2CLayer10%29_files/Vanishing%20gradient%280916%2CRelu%2CLayer10%29_9_0.png)
    



```python
# if video_save:
#     from matplotlib.animation import FuncAnimation
#     fig = plt.figure(figsize=[15, 9])
#     ymax_grad = max([i.max().item() for i in mean_grad_history])*1.1
#     ymax_weight = max([i.max().item() for i in mean_weight_history])*1.1
#     def animate(i): # for 문의 i 라고 생각하면 됨
#         plt.clf()

#         plt.suptitle(f"Epoch: {range(1,EPOCH+1)[i]}, Activation = {activation}, # of channels for a conv layer = {NoC}")

#         plt.subplot(2,2,1)
#         x_axis = [f"conv{i}" for i in range(1,NoB+1)]+["fc"]
#         plt.bar(x_axis, mean_grad_history[i].cpu(), width=0.8)
#         plt.ylim([0,ymax_grad])
#         plt.xlabel("layer")
#         plt.ylabel("mean absolute value")
#         plt.title("gradient")

#         plt.subplot(2,2,2)
#         x_axis = [f"conv{i}" for i in range(1,NoB+1)]+["fc"]
#         plt.bar(x_axis, mean_weight_history[i].cpu(), width=0.8)
#         plt.ylim([0,ymax_weight])
#         plt.xlabel("layer")
#         plt.ylabel("mean absolute value")
#         plt.title("weight")

#         ax1 = plt.subplot(2,1,2)
#         ax2 = ax1.twinx()
#         p1=ax1.plot(range(1,EPOCH+1)[:i+1],loss_history[:i+1],'r')
#         p2=ax2.plot(range(1,EPOCH+1)[:i+1],acc_history[:i+1])
#         ax1.set_xlim([-5,EPOCH+5])
#         ax1.set_ylim([min(loss_history)*0.7, max(loss_history)*1.1])
#         ax2.set_ylim([0, 100])
#         ax1.set_xlabel("Epochs")
#         ax1.set_ylabel("Loss")
#         ax2.set_ylabel("accuracy")
#         ax1.set_title(f"Train Loss (Final Test accuracy = {test_acc} % with BS = {BATCH_SIZE} & LR = {LR})")
#         ax1.grid()
#         plt.legend(p1+p2,["loss","accuracy"], loc="right")

#         plt.tight_layout()

#     ani = FuncAnimation(fig, animate, frames=EPOCH, interval=50)
#     # frames 에 10 넣으면 for i in range(10) 이라고 보면 됨. 혹은 list 넣어주면 in list
#     ani.save(save_video_path, writer='imagemagick')
```
