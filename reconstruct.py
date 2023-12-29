from model.EHANnet import EHANnet
from utils.dataset import MosaicDataset
from utils.dataset import MosaicDataset_single
from utils.dataset import MosaicDataset_enhance
from torch.utils.data import DataLoader
from utils.metrics import getPSNR, getSAM
import yaml
import numpy as np
from model.DAMCNet import DAMC
from model.SCRBnet import SCRBnet
from model.base_SCRBnet import Base_SCRBnet
from utils.dataset import MosaicDataset_enhance
from model.SCRBnetV3 import SCRBnetV3
from model.SCRBnetV2 import SCRBnetV2
from model.Res2Unet import MSFAUnet
from model.Res2Unet_baseV2 import MSFAUnet_baseV2
from model.Res2Unet_PPI import Res2Unet_PPIG
import torch
import os
import shutil
from utils.utils import *

if __name__ == "__main__":
    """
    测试模型
    """
    config_path = 'mosaick.yaml'
    masks = np.load("./masks.npy")
    masks_tensor = torch.from_numpy(masks)  #.unsqueeze(0)
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    #指定dataset
    DataSet = MosaicDataset_enhance
    if config['dataset'] == 'MosaicDataset_single':
        DataSet = MosaicDataset_single
    elif config['dataset'] == 'MosaicDataset':
        DataSet = MosaicDataset
    #训练时候采用传统差值，与训练时保持一致
    is_interp = config['train']['is_interp']
    #加载数据
    #test_dataset = DataSet(None,config['test']['test_dir'], masks, config['test']['norm'], is_interp,is_towx= True,fixed=config['test']['fixed'], test=True)
    test_dataset = DataSet(None,config['test']['test_dir'], masks, config['test']['norm'], is_interp,is_towx= False,fixed=config['test']['fixed'], test=True)

    # 保存恢复后的图像的路径
    recovered_dir = config['test']['recovered_dir']
    make_sure_path_exists(recovered_dir)
    #model_name = config['test']['model']


    device = torch.device(config['test']['device'])
    #model = Base_SCRBnet(8, 8, masks_tensor, scale=config['test']['sigmoid_scale'])   # DAMC(8, 8)
    #model = MSFAUnet_baseV2(8, 8, masks_tensor.shape,sigmoid_scale=config['test']['sigmoid_scale'])   # DAMC(8, 8)
    #model = Res2Unet_PPIG(8, 8, masks_tensor.shape,sigmoid_scale=config['test']['sigmoid_scale'])   # DAMC(8, 8)
    model = EHANnet(8, 8, 1, 1)
    model_name = model.__class__.__name__
    logger = get_logger_ckpdir_reconstruct(config, model_name)
    logger.info(config)
    logger.info("test_dataset: {}".format(len(test_dataset)))

    model.load_state_dict(torch.load(config['test']['pre_train'], map_location=device))
    model.to(device)

    sum_psnr= 0
    sum_sam = 0
    sum_time = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataset:
            hdr_file, norms, x_image_8, y_image = data
            #hdr_file, norms, x_image_1, x_image_8, y_image = data

            norms = [torch.tensor(norms[0]).unsqueeze(0)]
            # 输入图像
            #x_image_1 = torch.tensor(x_image_1[np.newaxis, :,:,:]).float().to(device)
            x_image_8 = torch.tensor(x_image_8[np.newaxis, :,:,:]).float().to(device)
            #真实图像 gt
            y_image = torch.tensor(y_image[np.newaxis, :,:,:]).float().to(device)
            # 模型输出
            import time

            torch.cuda.synchronize()
            time_start = time.time()  # 记录开始时间

            #pred_imag = model(x_image_1,x_image_8)
            pred_imag = model(x_image_8)

            torch.cuda.synchronize()
            time_end = time.time()  # 记录结束时间
            time_p = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            print(time_p)

            # 计算PSNR指标，四舍五入保存两位小数
            r_psnr = round(getPSNR(y_image, pred_imag).item(), 5)
            # sk_psnr = round(peak_signal_noise_ratio(y_image.detach().cpu().numpy(), pred_imag.detach().cpu().numpy()), 2)
            # "sk_psnr: ", sk_psnr
            r_sam = round(getSAM(y_image,pred_imag).item(),5)
            print(hdr_file, "  psnr: ", r_psnr)
            print(hdr_file, "  sam: ", r_sam)
            # 计算平均指标
            sum_psnr += r_psnr
            sum_sam += r_sam
            sum_time += time_p
            # 记录日志
            logger.info("hdr_file: {}, psnr: {}, sam: {}, time: {}".format(hdr_file,r_psnr,r_sam,time_p))
            # 反归一化
            deNorm_pred_imag = test_dataset.deNormlize(pred_imag, norms)

            recovered_uint16 = np.array(deNorm_pred_imag).astype(np.uint16)
            import datetime

            current_time = datetime.datetime.now()
            current_time_str = current_time.strftime("%Y-%m-%d")


            target_hdr = "target.hdr"
            save_img_path = os.path.join(recovered_dir, model_name,f"{model_name}_reconstruct_" + current_time_str +"_"+str(r_psnr).replace('.','-') + "_" + hdr_file.split('.')[0] + ".img")
            save_hdr_path = os.path.join(recovered_dir, model_name,f"{model_name}_reconstruct_" + current_time_str +"_"+str(r_psnr).replace('.','-') + "_" + hdr_file.split('.')[0] + ".hdr")
            make_sure_path_exists(os.path.dirname(save_img_path))
            make_sure_path_exists(os.path.dirname(save_hdr_path))

            recovered_uint16.tofile(save_img_path)
            shutil.copy(target_hdr, save_hdr_path)

            # 保存马赛克处理之后的图像，用于看效果
            deNorm_roi_imag = test_dataset.deNormlize(y_image, norms)
            mosaic_imag = deNorm_roi_imag * masks
            mosaic_imag_uint16 = np.array(mosaic_imag).astype(np.uint16)

            mosaic_save_img_path = os.path.join(recovered_dir,"mosaiced_" + str(r_psnr).replace('.', '-') + "_" + hdr_file.split('.')[0] + ".img")
            mosaic_save_hdr_path = os.path.join(recovered_dir,"mosaiced_" + str(r_psnr).replace('.', '-') + "_" + hdr_file.split('.')[0] + ".hdr")
            if not os.path.exists(mosaic_save_img_path):
                mosaic_imag_uint16.tofile(mosaic_save_img_path)
                shutil.copy(target_hdr, mosaic_save_hdr_path)
        logger.info("avg_psnr: {:.5f}, avg_sam: {:.5f}, avg_time: {:.5f}".format(sum_psnr/len(test_dataset), sum_sam/len(test_dataset), sum_time/len(test_dataset)))
        pass
