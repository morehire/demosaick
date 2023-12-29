import torch
import yaml
import numpy as np
import time
from tqdm import tqdm

from model.Res2Unet_base import MSFAUnet_base
from utils.dataset import MosaicDataset
from utils.dataset import MosaicDataset_single
from utils.dataset import MosaicDataset_towx
from utils.dataset import MosaicDataset_enhance
from torch.utils.data import DataLoader
from utils.metrics import getPSNR, getSAM
from utils.utils import random_seed, get_logger_ckpdir

from model.Res2Unet_PPI import Res2Unet_PPIG
from model.loss import MixLoss
from model.loss import MixLoss_ppi
import torch.optim as optim
from utils.normalize import deNormalizeMinMax, deNormalizeChannelMinMax
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model.EHANnet import EHANnet
import os


if __name__ == "__main__":
    masks = np.load("./masks.npy")
    masks_tensor = torch.from_numpy(masks)  #.unsqueeze(0)
    config_path = 'mosaick.yaml'
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    random_seed(config['random_seed'])
    checkpoint_save_dir, logger = get_logger_ckpdir(config, config['train']['model'])
    # 判断文件夹是否存在
    if not os.path.exists(checkpoint_save_dir):
        os.makedirs(checkpoint_save_dir)
        print(f"文件夹 '{checkpoint_save_dir}' 创建成功。")


    logger.info(config)

    DataSet = MosaicDataset_enhance
    if config['dataset']=='MosaicDataset_single':
        DataSet = MosaicDataset_single
    elif config['dataset']=='MosaicDataset_towx':
        DataSet = MosaicDataset_towx
    is_interp = config['train']['is_interp']
    #train_dataset = DataSet(config['data']['source_dir'],config['data']['train_dir'], masks, config['data']['norm'],is_interp,is_towx=False,fixed=config['data']['fixed'])
    train_dataset = DataSet(None,config['data']['train_dir'], masks, config['data']['norm'],is_interp,is_towx=False,fixed=config['data']['fixed'])
    valid_dataset = DataSet(None,config['data']['valid_dir'], masks, config['data']['norm'], is_interp,is_towx=False,fixed=config['data']['fixed'])


    logger.info("train_dataset: {}\\tvalid_dataset: {}".format(
                    len(train_dataset), len(valid_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['evaluate']['batch_size'], shuffle=False)

    device = torch.device(config['train']['device'])

    #model = Base_SCRBnet(8, 8, masks_tensor, scale=config['train']['sigmoid_scale'])  # DAMC(8, 8)
    #model = Res2Unet_PPIG(8, 8, masks_tensor.shape,sigmoid_scale=config['train']['sigmoid_scale'])
    #model = EHANnet(8, 8, 1,1)
    model = MSFAUnet_base(8,8,masks_tensor.shape,sigmoid_scale=config['train']['sigmoid_scale'])


    model_name = config['train']['model']
    if config['train']['pre_train'] != "None":
        model.load_state_dict(torch.load(config['train']['pre_train']))

    model.to(device)
    beat = torch.tensor(0.001).to(device)
    criterion = MixLoss(config['train']['loss_function']['losses'], config['train']['loss_function']['weights'])\
    #---PPIG
    #criterion = MixLoss_ppi(config['train']['loss_function']['losses'], config['train']['loss_function']['weights'])
    #optimizer = getattr(optim, config['train']['optimizer'])(model.parameters(), lr=config['train']['learning_rate'],betas=(0.5,0.99))
    optimizer = getattr(optim, config['train']['optimizer'])(model.parameters(), lr=config['train']['learning_rate'])
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=config['train']['learning_rate']*0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=config['train']['learning_rate']*0.1)
    best_score = 0.0  # psnr + 1/sam
    for epoch in tqdm(range(config['train']['max_epochs']), desc="train epoch: "):
        model.train()
        train_epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc="train batch: "):    # batch: (batch_size, C, H, W)
            optimizer.zero_grad()
            # norms, x_image_1,x_image_8, y_image = batch
            # x_image_1 = x_image_1.float().to(device)
            # x_image_8 = x_image_8.float().to(device)
            # y_image = y_image.float().to(device)
            # pred_imag = model(x_image_1,x_image_8)
            norms, x_image_8, y_image = batch
            x_image_8 = x_image_8.float().to(device)
            y_image = y_image.float().to(device)
            pred_imag = model(x_image_8)
            #scale = model.get_scale()
            #loss = criterion(y_image, pred_imag,model.get_scale(),beat)
            loss = criterion(y_image, pred_imag)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        valid_epoch_psnr = 0.0
        valid_peoch_sam = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="valid batch: "):  # batch: (batch_size, C, H, W)
                norms, x_image_8, y_image = batch
                x_image_8 = x_image_8.float().to(device)
                y_image = y_image.float().to(device)
                pred_imag = model(x_image_8)

                # norms, x_image_1, x_image_8, y_image = batch
                # x_image_1 = x_image_1.float().to(device)
                # x_image_8 = x_image_8.float().to(device)
                # y_image = y_image.float().to(device)
                # pred_imag = model(x_image_1, x_image_8)

                valid_epoch_psnr += getPSNR(y_image, pred_imag)
                valid_peoch_sam += getSAM(y_image, pred_imag)

            train_epoch_loss, valid_epoch_psnr, valid_peoch_sam = train_epoch_loss / len(train_dataloader), \
                                                                    valid_epoch_psnr / len(valid_dataloader), \
                                                                    valid_peoch_sam / len(valid_dataloader)
            #获取学习率
            learning_rate = optimizer.param_groups[0]['lr']

        logger.info("Epoch: {}, LearningRate: {:.5f}, TrainLoss: {:.5f}, ValidPSNR: {:.5f}, ValidSAM: {:.5f}".format(
                                epoch+1,learning_rate, train_epoch_loss, valid_epoch_psnr, valid_peoch_sam))

        if best_score < (valid_epoch_psnr+1/valid_peoch_sam):
            best_score = (valid_epoch_psnr+1/valid_peoch_sam)
            torch.save(model.state_dict(), '{}/{}-best-epoch-{}-psnr-{:.5f}-sam-{:.5f}-{}.pth'.format(
                checkpoint_save_dir,model_name, epoch+1, valid_epoch_psnr, valid_peoch_sam, time.strftime('%m-%d')))
        elif config['log']['save_frequency']>0 and (epoch+1)%config['log']['save_frequency']==0:
            torch.save(model.state_dict(), '{}/{}-epoch-{}-psnr-{:.5f}-sam-{:.5f}-{}.pth'.format(
                checkpoint_save_dir, model_name, epoch + 1, valid_epoch_psnr, valid_peoch_sam, time.strftime('%m-%d')))
