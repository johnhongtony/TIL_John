```python
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks')
from multiclass_functions1 import *
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
```

    Mounted at /content/drive
    cpu
    


```python
# for random seed
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
# BATCH_SIZE = 256
# LR = 1e-3
BATCH_SIZE = 32
LR = 1e-3
EPOCH = 100
criterion = nn.CrossEntropyLoss()
NoB = 10
NoC = 1
NoBat = 5
activation = "NoSigmoid"    # 'NO' => batch Nomalization을 안했다. 'BN'=>batch Nomalization하겠다.
new_model_train = False
model_type = f"{activation}{NoB}C{NoC}"
save_model_path = f"/content/drive/MyDrive/Colab Notebooks/results/BN/{model_type}_MNIST.pt"
save_video_path = f"/content/drive/MyDrive/Colab Notebooks/results/BN/comparison_{activation[2:]}_{NoB}C{NoC}.mp4"
save_video_path2 = f"/content/drive/MyDrive/Colab Notebooks/results/BN/No{model_type[2:]}_only befAct.mp4"
save_video_path3 = f"/content/drive/MyDrive/Colab Notebooks/results/BN/BN{model_type[2:]}_only befAct.mp4"
```


```python
transform = transforms.ToTensor()
train_DS = datasets.MNIST(root = '/content/drive/MyDrive/Colab Notebooks/data', train=True, download=True, transform=transform)
test_DS = datasets.MNIST(root = '/content/drive/MyDrive/Colab Notebooks/data', train=False, download=True, transform=transform)
train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True)
```


```python
# bias = True 면 ReLU 도 잘 안된다. 일단 bias=False로 변경
in_channel = 1
# in_channel = 3
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        if activation == "NoSigmoid":    # 'NO' => batch Nomalization을 안했다. 'BN'=>batch Nomalization하겠다.
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channel,NoC,3, bias=False, padding=1),
                nn.Sigmoid(),
                *[i for j in range(NoB-1) for i in [nn.Conv2d(NoC,NoC,3, bias=False, padding=1), nn.Sigmoid()]]
                )
        elif activation == "BNSigmoid":    # 'NO' => batch Nomalization을 안했다. 'BN'=>batch Nomalization하겠다.
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channel,NoC,3, bias=False, padding=1),
                nn.BatchNorm2d(NoC),  # activation인 sigmoid 전에 BN을 추가해줌(사실 activation 후에 위치해도 크게 차이는 없음)
                nn.Sigmoid(),
                *[i for j in range(NoB-1) for i in [nn.Conv2d(NoC,NoC,3, bias=False, padding=1), nn.BatchNorm2d(NoC), nn.Sigmoid()]]
                )
        elif activation == "NoReLU":    # 'NO' => batch Nomalization을 안했다. 'BN'=>batch Nomalization하겠다.
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channel,NoC,3, bias=False, padding=1),
                nn.ReLU(),
                *[i for j in range(NoB-1) for i in [nn.Conv2d(NoC,NoC,3, bias=False, padding=1), nn.ReLU()]]
                )
        elif activation == "BNReLU":    # 'NO' => batch Nomalization을 안했다. 'BN'=>batch Nomalization하겠다.
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channel,NoC,3, bias=False, padding=1),
                nn.BatchNorm2d(NoC),  # activation인 sigmoid 전에 BN을 추가해줌(사실 activation 후에 위치해도 크게 차이는 없음)
                nn.ReLU(),
                *[i for j in range(NoB-1) for i in [nn.Conv2d(NoC,NoC,3, bias=False, padding=1), nn.BatchNorm2d(NoC), nn.ReLU()]]
                )
        # self.fc = nn.Linear(NoC*(28-2*NoB)*(28-2*NoB),10, bias=False)
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(NoC*7*7,10, bias=False)
        # self.fc = nn.Linear(NoC*8*8,10, bias=False)

    def forward(self, x):
        self.befAct=[]
        for layer in self.conv_block:
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid):
                with torch.no_grad():
                    self.befAct += [torch.flatten(x).detach().cpu()]
                x = layer(x)
            else:
                x = layer(x)

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
        (1): Sigmoid()
        (2): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (3): Sigmoid()
        (4): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): Sigmoid()
        (6): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (7): Sigmoid()
        (8): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (9): Sigmoid()
        (10): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): Sigmoid()
        (12): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (13): Sigmoid()
        (14): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (15): Sigmoid()
        (16): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): Sigmoid()
        (18): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (19): Sigmoid()
      )
      (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc): Linear(in_features=49, out_features=10, bias=False)
    )
    torch.Size([32, 10])
    


```python
def fmap_gen(model, train_DL, idx=5):
    x_batch, _ = next(iter(train_DL))
    im_data = torch.unsqueeze(x_batch[idx], dim=0).to(DEVICE)
    feature_map=[]
    x = im_data
    model.eval()
    with torch.no_grad():
        for layer in model.conv_block:
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid):
                feature_map += [x.detach().squeeze().cpu()]
                x = layer(x)
            else:
                x = layer(x)
    return feature_map

def Train(model, train_DL, criterion):
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_history=[]
    acc_history=[]
    befAct_history=[]
    fmap_history=[]

    NoT=len(train_DL.dataset) # The number of training data

    for ep in range(EPOCH):
        rloss = 0 # running loss
        rcorrect = 0 # running correct
        befAct_e=[]
        # fmap
        fmap_history += [fmap_gen(model, train_DL)]
        model.train() # train mode로 전환
        for i, (x_batch, y_batch) in enumerate(train_DL):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # before activation
            if i < NoBat:
                befAct_e += [model.befAct]
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
        # print loss
        loss_e = rloss/NoT # epoch loss
        accuracy_e = rcorrect/NoT * 100
        loss_history += [loss_e]
        acc_history += [accuracy_e]
        befAct_history += [befAct_e]
        print(f"Epoch: {ep+1}, train loss: {round(loss_e,3)}, train accuracy: {round(accuracy_e,1)} %")
        print("-"*20)

    return loss_history, acc_history, befAct_history, fmap_history
```


```python
if new_model_train:
    loss_history, acc_history, befAct_history, fmap_history = Train(model, train_DL, criterion)

    torch.save({"model":model,
                "BATCH_SIZE":BATCH_SIZE,
                "LR":LR,
                "EPOCH":EPOCH,
                "NoB":NoB,
                "NoC":NoC,
                "loss_history":loss_history,
                "acc_history":acc_history,
                "befAct_history":befAct_history,
                "fmap_history":fmap_history,
                "activation":activation}, save_model_path)
```


```python
if activation[2:] == "ReLU":
    act = "NoReLU"
elif activation[2:] == "Sigmoid":
    act = "NoSigmoid"
model_type = f"{act}{NoB}C{NoC}"
save_model_path = f"/content/drive/MyDrive/Colab Notebooks/results/BN/{model_type}_MNIST.pt"
loaded=torch.load(save_model_path, map_location="cpu", weights_only=False)
load_model=loaded["model"].to(DEVICE)
loss_history_No=loaded["loss_history"]
acc_history_No=loaded["acc_history"]
befAct_history_No=loaded["befAct_history"]
fmap_history_No=loaded["fmap_history"]
test_acc_No = Test(load_model, test_DL)

if activation[2:] == "ReLU":
    act = "BNReLU"
elif activation[2:] == "Sigmoid":
    act = "BNSigmoid"
model_type = f"{act}{NoB}C{NoC}"
save_model_path = f"/content/drive/MyDrive/Colab Notebooks/results/BN/{model_type}_MNIST.pt"
loaded=torch.load(save_model_path, map_location="cpu", weights_only=False)
load_model=loaded["model"].to(DEVICE)
loss_history_BN=loaded["loss_history"]
acc_history_BN=loaded["acc_history"]
befAct_history_BN=loaded["befAct_history"]
fmap_history_BN=loaded["fmap_history"]
test_acc_BN = Test(load_model, test_DL)
```

    Test accuracy: 1135/10000 (11.3 %)
    Test accuracy: 9437/10000 (94.4 %)
    


```python
for layer in load_model.conv_block:
    if isinstance(layer, nn.BatchNorm2d):
        print(layer.weight.detach().cpu())    # Batch Nomalization Layer의 weight는 분산을 담당(weight는 곱해지는 것, 얼만큼 세게 뿌릴지)
        print(layer.bias.detach().cpu())      # Batch Nomalization Layer의 bias는 평균을 담당(weight는 더해지는 것, 어디에 뿌릴지)
        print("-"*20)
```

    tensor([0.7034])
    tensor([-0.8660])
    --------------------
    tensor([0.7678])
    tensor([-0.1713])
    --------------------
    tensor([1.0240])
    tensor([-0.2730])
    --------------------
    tensor([1.8532])
    tensor([-0.7914])
    --------------------
    tensor([4.0370])
    tensor([-2.0524])
    --------------------
    tensor([2.4948])
    tensor([0.7477])
    --------------------
    tensor([2.3084])
    tensor([1.2031])
    --------------------
    tensor([1.9797])
    tensor([-2.0637])
    --------------------
    tensor([1.5177])
    tensor([-1.8384])
    --------------------
    tensor([1.5358])
    tensor([-1.8825])
    --------------------
    


```python
# hist_range=7
# def plot_hist(ax, samples, layer, hist_max):
#     for i in range(NoBat):
#         hist, bins = np.histogram(samples[i][layer], range=(-hist_range,hist_range), bins=100)
#         # [batch][layer]
#         bins = (bins[:-1] + bins[1:])/2
#         bins_m=bins[bins<0]
#         bins_p=bins[bins>0]
#         hist_m=hist[bins<0]
#         hist_p=hist[bins>0]
#         ax.bar(bins_m, hist_m, zs=i, zdir="y", color='r', alpha=0.5, width=0.1)
#         ax.bar(bins_p, hist_p, zs=i, zdir="y", color='g', alpha=0.5, width=0.1)
#     plt.xlabel("values")
#     plt.ylabel("batch #")
#     ax.set_xlim([-hist_range,hist_range])
#     ax.set_zlim([0, hist_max])
#     ax.view_init(elev=20,azim=-70)
# # %%
# from matplotlib.animation import FuncAnimation

# hist_mean_max_No = torch.zeros(NoB,EPOCH)
# for layer in range(NoB):
#     for ep in range(EPOCH):
#         for i in range(NoBat):
#             hist, _ = np.histogram(befAct_history_No[ep][i][layer], range=(-hist_range,hist_range), bins=100)
#             max_val = max(hist)
#         if max_val > hist_mean_max_No[layer][ep]:
#             hist_mean_max_No[layer][ep] = max_val
# hist_mean_max_No = hist_mean_max_No.mean(dim=1)

# hist_mean_max_BN = torch.zeros(NoB,EPOCH)
# for layer in range(NoB):
#     for ep in range(EPOCH):
#         for i in range(NoBat):
#             hist, _ = np.histogram(befAct_history_BN[ep][i][layer], range=(-hist_range,hist_range), bins=100)
#             max_val = max(hist)
#         if max_val > hist_mean_max_BN[layer][ep]:
#             hist_mean_max_BN[layer][ep] = max_val
# hist_mean_max_BN = hist_mean_max_BN.mean(dim=1)
# # %%
# fig = plt.figure(figsize=[20,9])
# def animate(ep): # for 문의 i 라고 생각하면 됨
#     plt.clf()

#     plt.suptitle(f"Epoch: {range(1,EPOCH+1)[ep]}, # of conv block = {NoB}, # of channels = {NoC}")

#     ax=plt.subplot(2,6,1, projection="3d")
#     plot_hist(ax, befAct_history_No[ep], layer=0, hist_max=hist_mean_max_No[0])
#     plt.title("first layer")

#     ax=plt.subplot(2,6,2, projection="3d")
#     plot_hist(ax, befAct_history_No[ep], layer=-1, hist_max=hist_mean_max_No[-1])
#     plt.title("last layer")

#     ax1=plt.subplot(2,6,3)
#     ax2 = ax1.twinx()
#     p1=ax1.plot(range(1,EPOCH+1)[:ep+1],loss_history_No[:ep+1],'r')
#     p2=ax2.plot(range(1,EPOCH+1)[:ep+1],acc_history_No[:ep+1])
#     ax1.set_xlim([-5,EPOCH+5])
#     ax1.set_ylim([0.1, 2.5])
#     ax2.set_ylim([0, 100])
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Train Loss")
#     ax2.set_ylabel("accuracy")
#     ax1.set_title(f"Test acc = {test_acc_No} %, BS = {BATCH_SIZE} & LR = {LR}", fontsize=7)
#     ax1.grid()
#     plt.legend(p1+p2,["loss","accuracy"], loc="right")

#     for r in range(2):
#         for i in range(5):
#             plt.subplot(4,12,7+i+12*r, xticks=[], yticks=[])
#             plt.imshow(fmap_history_No[ep][i+r*5], cmap="gray")
#             plt.title(f"layer {i+r*5}")

#     ax=plt.subplot(2,6,7, projection="3d")
#     plot_hist(ax, befAct_history_BN[ep], layer=0, hist_max=hist_mean_max_BN[0])
#     plt.title("first layer")

#     ax=plt.subplot(2,6,8, projection="3d")
#     plot_hist(ax, befAct_history_BN[ep], layer=-1, hist_max=hist_mean_max_BN[-1])
#     plt.title("last layer")

#     ax1=plt.subplot(2,6,9)
#     ax2 = ax1.twinx()
#     p1=ax1.plot(range(1,EPOCH+1)[:ep+1],loss_history_BN[:ep+1],'r')
#     p2=ax2.plot(range(1,EPOCH+1)[:ep+1],acc_history_BN[:ep+1])
#     ax1.set_xlim([-5,EPOCH+5])
#     ax1.set_ylim([0.1, 2.5])
#     ax2.set_ylim([0, 100])
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Train Loss")
#     ax2.set_ylabel("accuracy")
#     ax1.set_title(f"Test acc = {test_acc_BN} %, BS = {BATCH_SIZE} & LR = {LR}", fontsize=7)
#     ax1.grid()
#     plt.legend(p1+p2,["loss","accuracy"], loc="right")

#     for r in range(2):
#         for i in range(5):
#             plt.subplot(4,12,24+7+i+12*r, xticks=[], yticks=[])
#             plt.imshow(fmap_history_BN[ep][i+r*5], cmap="gray")
#             plt.title(f"layer {i+r*5}")

#     plt.tight_layout(pad=0,h_pad=0,w_pad=0,rect=(0,0,0,0))

# ani = FuncAnimation(fig, animate, frames=range(0,100), interval=50)
# # frames 에 10 넣으면 for i in range(10) 이라고 보면 됨. 혹은 list 넣어주면 in list
# ani.save(save_video_path, writer='imagemagick')
# # %%
# fig = plt.figure(figsize=[25,10])
# def animate(ep):
#     plt.clf()
#     plt.suptitle(f"Epoch: {range(1,EPOCH+1)[ep]}, # of conv block = {NoB}, # of channels = {NoC}")
#     for r in range(2):
#         for i in range(5):
#             ax = plt.subplot(2,5,1+i+5*r, projection="3d")
#             plot_hist(ax, befAct_history_No[ep], layer=i+5*r, hist_max=hist_mean_max_No[i+5*r])
#             plt.title(f"layer {i+5*r}")

#     # plt.tight_layout()

# ani = FuncAnimation(fig, animate, frames=range(0,100), interval=50)
# ani.save(save_video_path2, writer='imagemagick')
# # %%
# fig = plt.figure(figsize=[25,10])
# def animate(ep):
#     plt.clf()
#     plt.suptitle(f"Epoch: {range(1,EPOCH+1)[ep]}, # of conv block = {NoB}, # of channels = {NoC}")
#     for r in range(2):
#         for i in range(5):
#             ax = plt.subplot(2,5,1+i+5*r, projection="3d")
#             plot_hist(ax, befAct_history_BN[ep], layer=i+5*r, hist_max=hist_mean_max_BN[i+5*r])
#             plt.title(f"layer {i+5*r}")

#     # plt.tight_layout()

# ani = FuncAnimation(fig, animate, frames=range(0,100), interval=50)
# ani.save(save_video_path3, writer='imagemagick')
```
