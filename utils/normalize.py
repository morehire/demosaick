import numpy as np


def normalizeMinMax(image, masks, fixed="None"):
    if fixed != "None":
        max_value = fixed
    else:
        max_value = np.max(image[masks])

    norm_image = image / max_value

    norms = [max_value.astype(np.float64)]
    return norms, norm_image


def applyNormalizeMinMax(image, norms):
    """
    应用最大最小值归一化
    :param image: 模型输入图像
    :param norms: 获取的图像最大值列表
    :return: 归一化之后的图像
    """
    norm = norms[0]

    norm_image = image / norm
    return norm_image


def deNormalizeMinMax(norm_image, norms):
    if norm_image.is_cuda:
        norm_image = norm_image.detach().cpu()
    norm = norms[0]

    b, c, h, w = norm_image.shape
    norm_image = norm_image.reshape([b, c, h * w])
    image = norm_image * norm[:, np.newaxis, np.newaxis]

    image = image.reshape([b, c, h, w])
    return image


def normalizeChannelMinMax(image, masks, ignore="None"):
    c, h, w = image.shape
    image = image.reshape([c, h * w])
    masks = masks.reshape([c, h * w])
    max_values = np.max(image[masks].reshape(c, -1), axis=-1)
    max_values[max_values==0.0] = 1.0

    norm_image = (image / max_values[:, np.newaxis]).reshape([c, h, w])
    norms = [max_values.astype(np.float64)]
    return norms, norm_image


def applyNormalizeChannelMinMax(image, norms):
    norm = norms[0]
    c, h, w = image.shape
    image = image.reshape([c, h * w])

    norm_image = (image/ norm[:, np.newaxis]).reshape([c, h, w])
    return norm_image


def deNormalizeChannelMinMax(norm_image, norms):
    if norm_image.is_cuda:
        norm_image = norm_image.detach().cpu()
    norm = norms[0]

    b, c, h, w = norm_image.shape
    norm_image = norm_image.reshape([b, c, h * w])
    image = norm_image * norm[:, :, np.newaxis]

    image = image.reshape([b, c, h, w])
    return image