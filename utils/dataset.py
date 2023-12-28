import numpy as np
import os
from torch.utils.data import Dataset
from .normalize import normalizeMinMax, normalizeChannelMinMax, applyNormalizeMinMax, applyNormalizeChannelMinMax
from .normalize import deNormalizeMinMax, deNormalizeChannelMinMax
from .utils import load_envi_img
from .utils import apply_mask_and_slide
from .interpolation import Interpolation_V5_3
import random


class MosaicDataset_enhance(Dataset):
    def __init__(self, source_dir,data_dir, masks, norm_type,is_interp,is_towx = False,fixed="None",test=False):
        if norm_type == "global_max" or norm_type == "fixed_max":
            self.normlize, self.applyNormlize, self.deNormlize = normalizeMinMax, applyNormalizeMinMax, deNormalizeMinMax
        elif norm_type == "channel_max":
            self.normlize, self.applyNormlize, self.deNormlize = normalizeChannelMinMax, applyNormalizeChannelMinMax, deNormalizeChannelMinMax
        self.masks = masks
        self.hdr_files = [file for file in os.listdir(data_dir) if file.endswith('.hdr')]
        # 大图根目录
        self.source_dir =  source_dir
        self.source_files = []
        if source_dir is not None and not test :
            # 保存大图文件名
            self.source_files = [file for file in os.listdir(source_dir) if file.endswith('.hdr')]
        self.fixed = fixed
        self.test = test
        self.is_interp = is_interp
        self.is_twox = is_towx
        self.data = []
        for i, _ in enumerate(self.hdr_files):
            hdr_file_path = os.path.join(data_dir, self.hdr_files[i])
            target_img = load_envi_img(hdr_file_path).load().astype(np.uint16).transpose(2, 0, 1).astype(np.float64)
            # 测试花
            #target_img = np.take(target_img, [4,5,6,0,1,2,3,7], axis=0)
            input_img = target_img * masks
            #先验插值
            if is_interp:
                input_img = Interpolation_V5_3(input_img)
            # 获取归一化参数，并对马赛克处理后的图像进行归一化
            # 8通道图像
            norms, norm_input_img_8 = self.normlize(input_img, masks, fixed)
            # 单通道输入图像
            merge_norm_input_img_1 = np.expand_dims(np.sum(norm_input_img_8, axis=0), axis=0)
            # 对原始图像进行归一化
            norm_target_img = self.applyNormlize(target_img, norms)
            if not test:
                if is_towx:
                    self.data.append((norms,merge_norm_input_img_1, norm_input_img_8, norm_target_img))
                else:
                    self.data.append((norms, norm_input_img_8, norm_target_img))
            else:
                # 测试时需要hdr文件方便查看效果
                if is_towx:
                    self.data.append((self.hdr_files[i], norms, merge_norm_input_img_1,norm_input_img_8, norm_target_img))
                else:
                    self.data.append((self.hdr_files[i], norms,norm_input_img_8, norm_target_img))

    def __len__(self):
        #这里的len大小就是后面getitem中idx的范围
        if self.source_dir is not None and not self.test :
            # 训练时对大图裁切
            return len(self.data)+len(self.source_files)
        return len(self.data)

    def __getitem__(self, idx):
        #print(f"==========idx{idx}===============")
        if self.test:
            return self.data[idx]
        if idx <len(self.data):
            return self.data[idx]
        #print(f"==========进行裁剪===============")
        #cut_img = self.random_cut_p()
        cut_img = self.random_cut_index(idx)
        return cut_img

    def random_cut(self,target_img):
        # 进行随机裁剪
        target_h, target_w = target_img[0].shape
        masks_h, masks_w = self.masks[0].shape
        # 生成随机剪裁位置
        row = random.randint(0, target_h - masks_h)
        col = random.randint(0, target_w - masks_w)
        res_data = target_img[:, row:row + masks_h, col:col + masks_w]
        return res_data

    def random_cut_p(self):
        """
        随机剪裁的图像索引位置由随机函数给出
        :param is_twox: 输入是否包含单通道的马赛克图
        :return: 随机剪裁的图像
        """
        # 生成随机文件
        data = []
        source_files = [file for file in os.listdir(self.source_dir) if file.endswith('.hdr')]
        file_numbers = len(source_files)
        # 生成随机剪裁的文件索引
        random_number = random.randint(0, file_numbers-1)
        random_file_name = source_files[random_number]
        random_file_path = os.path.join(self.source_dir, random_file_name)
        target_img = load_envi_img(random_file_path).load().astype(np.uint16).transpose(2, 0, 1).astype(np.float64)
        # 进行随机裁剪
        target_h, target_w = target_img[0].shape
        masks_h, masks_w = self.masks[0].shape
        # 生成随机剪裁位置
        row = random.randint(0, target_h - masks_h)
        col = random.randint(0, target_w - masks_w)
        res_data = target_img[:, row:row + masks_h, col:col + masks_w]
        input_img = res_data * self.masks
        # 先验插值
        if self.is_interp:
            input_img = Interpolation_V5_3(input_img)
        # 获取归一化参数，并对马赛克处理后的图像进行归一化
        # 8通道图像
        norms, norm_input_img_8 = self.normlize(input_img, self.masks, self.fixed)
        # 单通道输入图像
        merge_norm_input_img_1 = np.expand_dims(np.sum(norm_input_img_8 * self.masks, axis=0), axis=0)
        # 对原始图像进行归一化
        norm_target_img = self.applyNormlize(res_data, norms)

        if self.is_twox:
            data.append((norms, merge_norm_input_img_1, norm_input_img_8, norm_target_img))
        else:
            data.append((norms, norm_input_img_8, norm_target_img))
        return data[0]

    def random_cut_index(self,idx):
        """
        随机剪裁的图像索引位置根据pytorch中的dataloader给出然后减去原数据量大小得到
        :param idx: dataloader随机生成的索引，范围是__len__函数的返回值
        :return: 随机剪裁的图像
        """
        # 生成随机文件
        data = []
        source_files = [file for file in os.listdir(self.source_dir) if file.endswith('.hdr')]
        #file_numbers = len(source_files)
        # 生成随机剪裁的文件索引
        #random_number = random.randint(0, file_numbers-1)
        # 将文件索引范围映射到需要剪裁的大图索引
        random_number = idx - len(self.data)
        random_file_name = source_files[random_number]
        random_file_path = os.path.join(self.source_dir, random_file_name)
        target_img = load_envi_img(random_file_path).load().astype(np.uint16).transpose(2, 0, 1).astype(np.float64)
        # 进行随机裁剪
        target_h, target_w = target_img[0].shape
        masks_h, masks_w = self.masks[0].shape
        # 生成随机剪裁位置
        row = random.randint(0, target_h - masks_h)
        col = random.randint(0, target_w - masks_w)
        res_data = target_img[:, row:row + masks_h, col:col + masks_w]
        input_img = res_data * self.masks
        # 先验插值
        if self.is_interp:
            input_img = Interpolation_V5_3(input_img)
        # 获取归一化参数，并对马赛克处理后的图像进行归一化
        # 8通道图像
        norms, norm_input_img_8 = self.normlize(input_img, self.masks, self.fixed)
        # 单通道输入图像
        merge_norm_input_img_1 = np.expand_dims(np.sum(norm_input_img_8 * self.masks, axis=0), axis=0)
        # 对原始图像进行归一化
        norm_target_img = self.applyNormlize(res_data, norms)
        if self.is_twox:
            data.append((norms,merge_norm_input_img_1,norm_input_img_8, norm_target_img))
        else:
            data.append((norms, norm_input_img_8, norm_target_img))
        return data[0]



class MosaicDataset_towx(Dataset):
    def __init__(self, data_dir, masks, norm_type, fixed="None", test=False):
        if norm_type == "global_max" or norm_type == "fixed_max":
            self.normlize, self.applyNormlize, self.deNormlize = normalizeMinMax, applyNormalizeMinMax, deNormalizeMinMax
        elif norm_type == "channel_max":
            self.normlize, self.applyNormlize, self.deNormlize = normalizeChannelMinMax, applyNormalizeChannelMinMax, deNormalizeChannelMinMax

        self.hdr_files = [file for file in os.listdir(data_dir) if file.endswith('.hdr')]
        self.data = []
        for i, _ in enumerate(self.hdr_files):
            hdr_file_path = os.path.join(data_dir, self.hdr_files[i])
            target_img = load_envi_img(hdr_file_path).load().astype(np.uint16).transpose(2, 0, 1).astype(np.float64)
            # 测试花
            #target_img = np.take(target_img, [4,5,6,0,1,2,3,7], axis=0)
            input_img = target_img*masks
            # 先验插值
            #interp_input_img = Interpolation_V5_3(input_img)
            # 获取归一化参数，并对马赛克处理后的图像进行归一化
            # 8通道图像
            norms, norm_input_img_8 = self.normlize(input_img, masks, fixed)
            #单通道输入图像
            merge_norm_input_img_1 = np.expand_dims(np.sum(norm_input_img_8, axis=0), axis=0)
            # 对原始图像进行归一化
            norm_target_img = self.applyNormlize(target_img, norms)
            if not test:
                self.data.append((norms, merge_norm_input_img_1, norm_input_img_8, norm_target_img))
            else:
                # 测试时需要hdr文件方便查看效果
                self.data.append((self.hdr_files[i], norms, merge_norm_input_img_1, norm_input_img_8, norm_target_img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass
        return self.data[idx]

class MosaicDataset_single(Dataset):
    def __init__(self, data_dir, masks, norm_type, fixed="None", test=False):
        if norm_type == "global_max" or norm_type == "fixed_max":
            self.normlize, self.applyNormlize, self.deNormlize = normalizeMinMax, applyNormalizeMinMax, deNormalizeMinMax
        elif norm_type == "channel_max":
            self.normlize, self.applyNormlize, self.deNormlize = normalizeChannelMinMax, applyNormalizeChannelMinMax, deNormalizeChannelMinMax

        self.hdr_files = [file for file in os.listdir(data_dir) if file.endswith('.hdr')]
        self.data = []
        for i, _ in enumerate(self.hdr_files):
            hdr_file_path = os.path.join(data_dir, self.hdr_files[i])
            target_img = load_envi_img(hdr_file_path).load().astype(np.uint16).transpose(2, 0, 1).astype(np.float64)
            # 测试花
            #target_img = np.take(target_img, [4,5,6,0,1,2,3,7], axis=0)
            input_img = target_img*masks
            # 先验插值
            #interp_input_img = Interpolation_V5_3(input_img)
            # 获取归一化参数，并对马赛克处理后的图像进行归一化

            norms, norm_input_img = self.normlize(input_img, masks, fixed)
            #得到单通道图像
            merge_norm_input_img = np.expand_dims(np.sum(norm_input_img, axis=0), axis=0)

            # 对原始图像进行归一化
            norm_target_img = self.applyNormlize(target_img, norms)
            if not test:
                self.data.append((norms, merge_norm_input_img, norm_target_img))
            else:
                # 测试时需要hdr文件方便查看效果
                self.data.append((self.hdr_files[i], norms, merge_norm_input_img, norm_target_img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class MosaicDataset(Dataset):
    def __init__(self, data_dir, masks, norm_type, fixed="None", test=False):
        if norm_type == "global_max" or norm_type == "fixed_max":
            self.normlize, self.applyNormlize, self.deNormlize = normalizeMinMax, applyNormalizeMinMax, deNormalizeMinMax
        elif norm_type == "channel_max":
            self.normlize, self.applyNormlize, self.deNormlize = normalizeChannelMinMax, applyNormalizeChannelMinMax, deNormalizeChannelMinMax

        self.hdr_files = [file for file in os.listdir(data_dir) if file.endswith('.hdr')]
        self.data = []
        for i, _ in enumerate(self.hdr_files):
            hdr_file_path = os.path.join(data_dir, self.hdr_files[i])
            target_img = load_envi_img(hdr_file_path).load().astype(np.uint16).transpose(2, 0, 1).astype(np.float64)
            # 测试花
            #target_img = np.take(target_img, [4,5,6,0,1,2,3,7], axis=0)
            input_img = target_img*masks
            # 先验插值
            interp_input_img = Interpolation_V5_3(input_img)
            # 获取归一化参数，并对马赛克处理后的图像进行归一化
            norms, norm_input_img = self.normlize(interp_input_img, masks, fixed)

            # 对原始图像进行归一化
            norm_target_img = self.applyNormlize(target_img, norms)
            if not test:
                self.data.append((norms, norm_input_img, norm_target_img))
            else:
                # 测试时需要hdr文件方便查看效果
                self.data.append((self.hdr_files[i], norms, norm_input_img, norm_target_img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


