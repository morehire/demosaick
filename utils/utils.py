import errno

import numpy as np
import random
import torch
import os
from .log import make_sure_path_exists, log_creater
from datetime import datetime
import spectral.io.envi as envi

def parse_hdr_info(input_file):
    hdr_path = os.path.splitext(input_file)[0] + ".hdr"

    try:
        with open(hdr_path) as f:
            content = f.readlines()
    except Exception as e:
        print("open hdr file error!", e)
        return None
    samples, lines, bands, data_type, interleave, band_names, wavelength, byte_order, exposure, gain = \
        (None, None, None, None, None, None, None, None, None, None)
    byte_order = 0

    for ll in content:
        line_element = ll.replace('\n', '').split("=")
        if line_element[0].strip() == ("samples"):
            samples = int(line_element[1].strip())
        elif line_element[0].strip() == ("lines"):
            lines = int(line_element[1].strip())
        elif line_element[0].strip() == ("bands"):
            bands = int(line_element[1].strip())
        elif line_element[0].strip() == ("data type"):
            data_type = int(line_element[1].strip())
        elif line_element[0].strip() == ("interleave"):
            interleave = line_element[1].strip()
        elif line_element[0].strip() == ("byte order"):
            byte_order = int(line_element[1].strip())
        elif line_element[0].strip() == ("band names"):
            band_names = line_element[1].strip()[1:-1].split(",")
        elif line_element[0].strip() == ("wavelength"):
            wavelength = line_element[1].strip()[1:-1].split(",")
        elif line_element[0].strip() == ("Time of exposure"):
            exposure = line_element[1].strip()[1:-1].split(",")
        elif line_element[0].strip() == ("Gain"):
            if line_element[1].strip().find(',') != -1:
                gain = line_element[1].strip()[1:-1].split(",")
            else:
                gain = [line_element[1].strip()]
    return samples, lines, bands, data_type, interleave, band_names, wavelength, byte_order, exposure, gain
def get_img_info(img_file_path, file_name, req_wave=np.array([450, 550, 650, 720, 750, 800, 850, 950])):
    # samples, lines, bands, data_type, interleave, band_names, wavelength, byte_order, exposure, gain
    hdr_info = parse_hdr_info(img_file_path)

    if hdr_info is None:
        return (None, None, None)

    samples = hdr_info[0]
    lines = hdr_info[1]
    bands = hdr_info[2]
    data_type = hdr_info[3]
    interleave = hdr_info[4]
    band_names = hdr_info[5]
    wavelength = hdr_info[6]
    for w in wavelength:
        if w.find('.') != -1:
            print("hdr 波段参数存在小数, 不符合要求, 请删除input下的文件！")
            return (None, None, None)
    wavelength = np.array(list(map(int, wavelength)))
    byte_order = hdr_info[7]

    exposure = hdr_info[8]
    if exposure is None:
        exposure = np.ones_like(wavelength).astype(np.float64)
    else:
        exposure = np.array(list(map(int, exposure)))
    gain = hdr_info[9]
    if gain is None:
        gain = np.ones_like(wavelength).astype(np.float64)
    else:
        gain = np.array(list(map(int, gain)))

    if len(exposure)>0:
        if exposure[0] == 0:
            exposure[0] = 1
        exposure = exposure / exposure[0]
    if len(exposure) != len(gain):
        gain = np.repeat(gain, len(exposure))

    req_wave_index = []
    for w in req_wave:
        index = np.where(w == wavelength)[0]
        if len(index) == 0:
            print(img_file_path, f"无法找到目标波段 {w} 的下标")
            return (None, None, None)
        req_wave_index.append(index[0])

    byte_num = 1
    if data_type == 12:
        byte_num = 2

    little_endian = True
    if byte_order == 1:
        little_endian = False

    if interleave == 'bsq':
        if byte_num == 1:
            img = np.fromfile(img_file_path, dtype=np.uint8).reshape(bands, lines, samples)
        elif byte_num == 2:
            if little_endian:
                img = np.fromfile(img_file_path, dtype='<u2').reshape(bands, lines, samples)
            else:
                img = np.fromfile(img_file_path, dtype='>u2').reshape(bands, lines, samples)
    else:
        return (None, None, None)

    hdr_path = os.path.splitext(file_name)[0] + ".hdr"
    req_imgs = img[req_wave_index]
    exposure, gain = exposure[req_wave_index], gain[req_wave_index]
    expanded_exposure = exposure[:, np.newaxis, np.newaxis]
    expanded_gain = gain[:, np.newaxis, np.newaxis]

    #处理增益和曝光
    #req_imgs = req_imgs / expanded_exposure / expanded_gain

    # update hdr_info
    hdr_info = list(hdr_info)
    hdr_info[0] = 636
    hdr_info[1] = 480
    hdr_info[2] = 8
    hdr_info[5] = [hdr_info[5][index] for index in req_wave_index]
    hdr_info[6] = [hdr_info[6][index] for index in req_wave_index]
    hdr_info[8] = list(expanded_exposure.squeeze())
    hdr_info[9] = list(expanded_gain.squeeze())
    return img, hdr_info, req_imgs


def parse_hdr_exp_gain(hdr_path):
    try:
        with open(hdr_path) as f:
            content = f.readlines()
    except Exception as e:
        print("open hdr file error!", e)
        return None
    exposure, gain = None, None

    for ll in content:
        line_element = ll.replace('\n', '').split("=")
        if line_element[0].strip() == ("exposure"):
            exposure = line_element[1].strip()[1:-1].split(",")
        elif line_element[0].strip() == ("gain"):
            if line_element[1].strip().find(',') != -1:
                gain = line_element[1].strip()[1:-1].split(",")
            else:
                gain = [line_element[1].strip()]

    exposure = np.array(list(map(float, exposure)))
    gain = np.array(list(map(float, gain)))
    return [exposure, gain]

def extract_hdr_values(hdr_filename):
    samples = None
    lines = None
    bands = None

    try:
        with open(hdr_filename, 'r') as hdr_file:
            for line in hdr_file:
                line = line.strip()
                if line.startswith('samples'):
                    samples = int(line.split('=')[1].strip())
                elif line.startswith('lines'):
                    lines = int(line.split('=')[1].strip())
                elif line.startswith('bands'):
                    bands = int(line.split('=')[1].strip())
    except FileNotFoundError:
        print(f"File '{hdr_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return samples, lines, bands


def apply_mask_and_slide(data, mask):
    mask_height, mask_width = mask[0].shape
    data_height, data_width = data[0].shape
    step_height, step_width = (mask_height, mask_width)

    result = []
    for row in range(0, data_height, step_height):
        row_result = []
        start_row = row + 0
        end_row = min(data_height, start_row + mask_height)
        if end_row - start_row >= mask_height // 2:
            start_row = end_row - mask_height
        else:
            continue

        for col in range(0, data_width, step_width):
            start_col = col + 0
            end_col = min(data_width, start_col + mask_width)
            if end_col - start_col >= mask_width // 2:
                start_col = end_col - mask_width
            else:
                continue

            result.append(data[:, start_row:end_row, start_col:end_col])
    return result


def random_seed(seed=666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger_ckpdir(config, model):


    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 将日期转换为字符串
    date_string = current_datetime.strftime("%m_%d")

    data_set = str(config['data']['data_set'])
    loss_str = '_'.join(list(map(str, config['train']['loss_function']['weights'])))
    sub_dir = os.path.join(model, str(config['train']['batch_size'])+"_"+\
                                   str(config['train']['learning_rate']) +"_"+ loss_str + '_'+data_set+'_'+date_string)
    sub_dir = sub_dir.replace('.', ',')
    checkpoint_save_dir = os.path.join(config['log']['ckp_dir'], sub_dir)
    log_save_dir = os.path.join(config['log']['log_dir'], sub_dir)
    make_sure_path_exists(checkpoint_save_dir)
    logger = log_creater(log_save_dir,'train_log')
    return checkpoint_save_dir, logger

def get_logger_ckpdir_reconstruct(config, model):
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 将日期转换为字符串
    date_string = current_datetime.strftime("%m_%d")

    data_set = str(config['test']['data_set'])
    loss_str = '_'.join(list(map(str, config['train']['loss_function']['weights'])))
    sub_dir = os.path.join(model, str(config['train']['batch_size']) + "_" + \
                           str(config['train']['learning_rate']) + "_" + loss_str + '_' + data_set + '_' + date_string)
    sub_dir = sub_dir.replace('.', ',')
    #checkpoint_save_dir = os.path.join(config['log']['ckp_dir'], sub_dir)
    log_save_dir = os.path.join(config['test']['log'], sub_dir)
    #make_sure_path_exists(checkpoint_save_dir)
    logger = log_creater(log_save_dir, 'test_log')

    return logger

def __fixHDRfile(filePath):
    """
    Append the required "byte order" property into the .hdr file to fix the error\n
    修复.hdr文件
    ENVI Header Format: https://www.l3harrisgeospatial.com/docs/enviheaderfiles.html
    """

    with open(filePath, encoding='utf-8') as hdrFile:
        hdrInfo = hdrFile.read()

    if not "byte order" in hdrInfo:
        with open(filePath, 'a', encoding='utf-8') as hdrFile:
            hdrFile.write('\nbyte order = 0')

    with open(filePath, encoding='utf-8') as hdrFile:
        hdrInfo = hdrFile.readlines()

    if not "ENVI" in hdrInfo[0] and 'Wayho' in hdrInfo[0]:
        hdrInfo[0] = 'ENVI'
        with open(filePath, 'w', encoding='utf-8') as hdrFile:
            hdrFile.writelines(hdrInfo)


def load_envi_img(hdr_path):
    try:
        img = envi.open(hdr_path)
    except:
        __fixHDRfile(hdr_path)
        img = envi.open(hdr_path)
    return img