import os.path
import re
import matplotlib.pyplot as plt
import numpy as np
from utils.log import make_sure_path_exists

log_file_path = r"log_channel_max_nas/log/Res2Unet_baseV2_PPIG/2_0,0001_0,99_0,0_0,01_nas_dataset/2023-12-14-13-58.log"
save_draw_dir = log_file_path.rsplit('.', 1)[0]
make_sure_path_exists(save_draw_dir)

show = True
save = True
with open(log_file_path, 'r',errors='ignore') as file:
    log_text = file.read()

train_pattern = r"Epoch: (\d+), LearningRate: ([\d.]+), TrainLoss: ([\d.]+), ValidPSNR: ([\d.]+), ValidSAM: ([\d.]+)"

train_epochs = []
train_lr_values = []
train_loss_values = []
valid_psnr_values = []
valid_sam_values = []

train_matches = re.findall(train_pattern, log_text)

for match in train_matches:
    epoch, lr, train_loss, train_psnr, train_sam = match
    train_epochs.append(int(epoch))
    train_lr_values.append(float(lr))
    train_loss_values.append(float(train_loss))
    valid_psnr_values.append(float(train_psnr))
    valid_sam_values.append(float(train_sam))


x_ticks = np.linspace(min(train_epochs), max(train_epochs), 10)
fontsize = 16

plt.figure(figsize=(12, 6))
plt.plot(train_epochs, train_lr_values)
plt.xlabel('epoch', fontsize=fontsize)
plt.ylabel('lr', fontsize=fontsize)
plt.title('NAS-lr', fontsize=fontsize+4)
plt.grid(True)
plt.xticks(x_ticks.astype(int))
if save:
    plt.savefig(os.path.join(save_draw_dir, "NAS-lr.png"))
if show:
    plt.show()


plt.figure(figsize=(12, 6))
plt.plot(train_epochs, train_loss_values)
plt.xlabel('epoch', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.title('NAS-TrainLoss', fontsize=fontsize+4)
plt.grid(True)
plt.xticks(x_ticks.astype(int))
if save:
    plt.savefig(os.path.join(save_draw_dir, "NAS-train_loss.png"))
if show:
    plt.show()


plt.figure(figsize=(12, 6))
plt.plot(train_epochs, valid_psnr_values)
plt.xlabel('epoch', fontsize=fontsize)
plt.ylabel('psnr', fontsize=fontsize)
plt.title('NAS-Valid-PSNR', fontsize=fontsize+4)
plt.grid(True)
plt.xticks(x_ticks.astype(int))
if save:
    plt.savefig(os.path.join(save_draw_dir, "NAS-train-valid-psnr.png"))
if show:
    plt.show()


plt.figure(figsize=(12, 6))
plt.plot(train_epochs, valid_sam_values)
plt.xlabel('epoch', fontsize=fontsize)
plt.ylabel('sam', fontsize=fontsize)
plt.title('NAS-Train-Valid-SAM', fontsize=fontsize+4)
plt.grid(True)
plt.xticks(x_ticks.astype(int))
if save:
    plt.savefig(os.path.join(save_draw_dir, "NAS-valid-sam.png"))
if show:
    plt.show()
