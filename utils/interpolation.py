import numpy as np


def Interpolation_V5_3(img):
    # (C,H,W)
    C, H, W = img.shape
    CH_NUM = 8

    CH_POS = [
        [(0, 0), (2, 2)],
        [(0, 2), (2, 0)],
        [(1, 1), (3, 3)],
        [(1, 3), (3, 1)],
        [(0, 1), (2, 3)],
        [(0, 3), (2, 1)],
        [(1, 0), (3, 2)],
        [(1, 2), (3, 0)]
    ]

    CH_NBR_UDLR = [
        [7, 6, 5, 4],
        [6, 7, 4, 5],
        [4, 5, 6, 7],
        [5, 4, 7, 6],
        [3, 2, 0, 1],
        [2, 3, 1, 0],
        [0, 1, 3, 2],
        [1, 0, 2, 3]
    ]

    # 斜向 \\ //
    ch_45 = [
        [2, 3],
        [3, 2],
        [0, 1],
        [1, 0],
        [7, 6],
        [6, 7],
        [5, 4],
        [4, 5]
    ]

    padding = 6

    img_out = np.zeros((img.shape[0], img.shape[1] + padding * 2, img.shape[2] + padding * 2), dtype=np.float32) + 0.01

    img_out[:, padding:padding + H, padding:padding + W] = img.astype(np.float32) + 0.01

    # 切割小矩阵
    # print(img_out.strides)
    kernel_size = (24, 24)
    stride = (12, 12)

    stride0, stride1, stride2 = img_out.strides
    # print(*img_out.strides[-2:])
    stride3, stride4 = img_out.strides[-2:]

    shape0 = img_out.shape[0]
    shape1 = int((img_out.shape[1] - (kernel_size[0] - 1)) / stride[0] + 1)
    shape2 = int((img_out.shape[2] - (kernel_size[1] - 1)) / stride[1] + 1)

    A1 = np.lib.stride_tricks.as_strided(img_out,
                                         shape=(shape0, shape1, shape2, *kernel_size),
                                         strides=(stride0, stride1 * stride[0], stride2 * stride[1], stride3, stride4))
    # print(A1)

    for ch in range(CH_NUM):
        ch_nbr_hv = CH_NBR_UDLR[ch]
        # 补上下左右
        for pos in CH_POS[ch]:  # 本通道真值的坐标
            r = padding + pos[0] * 3
            c = padding + pos[1] * 3

            # UP
            base_value = np.mean(A1[ch, :, :, r, c:c + 3], axis=2).reshape((40, 53, 1, 1))
            base_value = np.repeat(np.repeat(base_value, 3, axis=2), 3, axis=3)
            ch_peer = ch_nbr_hv[0]
            comp_value = np.mean(A1[ch_peer, :, :, r - 1, c:c + 3], axis=2).reshape((40, 53, 1, 1))
            comp_value = np.repeat(np.repeat(comp_value, 3, axis=2), 3, axis=3)
            A1[ch, :, :, r - 3:r - 3 + 3, c:c + 3] = A1[ch_peer, :, :, r - 3:r - 3 + 3,
                                                     c:c + 3] / comp_value * base_value

            # Down
            base_value = np.mean(A1[ch, :, :, r + 2, c:c + 3], axis=2).reshape((40, 53, 1, 1))
            base_value = np.repeat(np.repeat(base_value, 3, axis=2), 3, axis=3)
            ch_peer = ch_nbr_hv[1]
            comp_value = np.mean(A1[ch_peer, :, :, r + 3, c:c + 3], axis=2).reshape((40, 53, 1, 1))
            comp_value = np.repeat(np.repeat(comp_value, 3, axis=2), 3, axis=3)
            A1[ch, :, :, r + 3:r + 3 + 3, c:c + 3] = A1[ch_peer, :, :, r + 3:r + 3 + 3,
                                                     c:c + 3] / comp_value * base_value

            # Left
            base_value = np.mean(A1[ch, :, :, r:r + 3, c], axis=2).reshape((40, 53, 1, 1))
            base_value = np.repeat(np.repeat(base_value, 3, axis=2), 3, axis=3)
            ch_peer = ch_nbr_hv[2]
            comp_value = np.mean(A1[ch_peer, :, :, r:r + 3, c - 1], axis=2).reshape((40, 53, 1, 1))
            comp_value = np.repeat(np.repeat(comp_value, 3, axis=2), 3, axis=3)
            A1[ch, :, :, r:r + 3, c - 3:c - 3 + 3] = A1[ch_peer, :, :, r:r + 3,
                                                     c - 3:c - 3 + 3] / comp_value * base_value

            # Right
            base_value = np.mean(A1[ch, :, :, r:r + 3, c + 2], axis=2).reshape((40, 53, 1, 1))
            base_value = np.repeat(np.repeat(base_value, 3, axis=2), 3, axis=3)
            ch_peer = ch_nbr_hv[3]
            comp_value = np.mean(A1[ch_peer, :, :, r:r + 3, c + 3], axis=2).reshape((40, 53, 1, 1))
            comp_value = np.repeat(np.repeat(comp_value, 3, axis=2), 3, axis=3)
            A1[ch, :, :, r:r + 3, c + 3:c + 3 + 3] = A1[ch_peer, :, :, r:r + 3,
                                                     c + 3:c + 3 + 3] / comp_value * base_value

        # 补两个斜向
        ch_peer_1, ch_peer_2 = ch_45[ch]

        # \\\\\
        offset_list = CH_POS[ch_peer_1]  # 需要补的元素的坐标
        for r_off, c_off in offset_list:
            r = padding + r_off * 3
            c = padding + c_off * 3

            # 以斜向两个角点，按比例平均
            A1[ch, :, :, r, c] = A1[ch, :, :, r - 1, c - 1]
            d1 = A1[ch, :, :, r, c] / A1[ch_peer_1, :, :, r, c]
            A1[ch, :, :, r + 2, c + 2] = A1[ch, :, :, r + 3, c + 3]
            d2 = A1[ch, :, :, r + 2, c + 2] / A1[ch_peer_1, :, :, r + 2, c + 2]

            A1[ch, :, :, r + 1, c] = A1[ch_peer_1, :, :, r + 1, c] * d1
            A1[ch, :, :, r, c + 1] = A1[ch_peer_1, :, :, r, c + 1] * d1

            A1[ch, :, :, r + 2, c + 1] = A1[ch_peer_1, :, :, r + 2, c + 1] * d2
            A1[ch, :, :, r + 1, c + 2] = A1[ch_peer_1, :, :, r + 1, c + 2] * d2

            A1[ch, :, :, r, c + 2] = A1[ch_peer_1, :, :, r, c + 2] * (d2 + d1) / 2
            A1[ch, :, :, r + 1, c + 1] = A1[ch_peer_1, :, :, r + 1, c + 1] * (d2 + d1) / 2
            A1[ch, :, :, r + 2, c] = A1[ch_peer_1, :, :, r + 2, c] * (d2 + d1) / 2

            pass

        # /////
        offset_list = CH_POS[ch_peer_2]
        for r_off, c_off in offset_list:
            r = padding + r_off * 3
            c = padding + c_off * 3

            # value = (np.sum(img_out[ch, r-3:r-3+3, c+3:c+3+3]) +
            #              np.sum(img_out[ch, r+3:r+3+3, c-3:c-3+3])) / 18

            # img_j_ratio = img_out[ch_peer_2, r:r+3, c:c+3] /  np.mean(img_out[ch_peer_2, r:r+3, c:c+3])

            # img_out[ch, r:r+3, c:c+3] = img_j_ratio * value

            # 以斜向两个角点，按比例平均
            A1[ch, :, :, r, c + 2] = A1[ch, :, :, r - 1, c + 3]
            d1 = A1[ch, :, :, r, c + 2] / A1[ch_peer_2, :, :, r, c + 2]
            A1[ch, :, :, r + 2, c] = A1[ch, :, :, r + 3, c - 1]
            d2 = A1[ch, :, :, r + 2, c] / A1[ch_peer_2, :, :, r + 2, c]

            A1[ch, :, :, r + 1, c + 2] = A1[ch_peer_2, :, :, r + 1, c + 2] * d1
            A1[ch, :, :, r, c + 1] = A1[ch_peer_2, :, :, r, c + 1] * d1

            A1[ch, :, :, r + 2, c + 1] = A1[ch_peer_2, :, :, r + 2, c + 1] * d2
            A1[ch, :, :, r + 1, c] = A1[ch_peer_2, :, :, r + 1, c] * d2

            A1[ch, :, :, r, c] = A1[ch_peer_2, :, :, r, c] * (d2 + d1) / 2
            A1[ch, :, :, r + 1, c + 1] = A1[ch_peer_2, :, :, r + 1, c + 1] * (d2 + d1) / 2
            A1[ch, :, :, r + 2, c + 2] = A1[ch_peer_2, :, :, r + 2, c + 2] * (d2 + d1) / 2

            pass

        # 十字中心
        if ch % 2 == 0:
            ch_peer = ch + 1
        else:
            ch_peer = ch - 1

        offset_list = CH_POS[ch_peer]
        for r_off, c_off in offset_list:
            r = padding + r_off * 3
            c = padding + c_off * 3

            # 利用最接近的那三个元素
            d_h = np.sum(A1[ch, :, :, r:r + 3, c - 1], axis=2) - np.sum(A1[ch, :, :, r:r + 3, c + 3], axis=2)
            d_v = np.sum(A1[ch, :, :, r - 1, c:c + 3], axis=2) - np.sum(A1[ch, :, :, r + 3, c:c + 3], axis=2)
            direct = np.abs(d_h) - np.abs(d_v)

            value_v = (np.sum(A1[ch, :, :, r - 1, c:c + 3], axis=2) +
                       np.sum(A1[ch, :, :, r + 3, c:c + 3], axis=2)) / 6
            direct1 = np.zeros(direct.shape, dtype=np.int8)
            direct1[np.bitwise_and(np.abs(direct) >= 1e-9, direct > 0)] = 1
            value_v = value_v * direct1

            value_h = (np.sum(A1[ch, :, :, r:r + 3, c - 1], axis=2) +
                       np.sum(A1[ch, :, :, r:r + 3, c + 3], axis=2)) / 6
            direct2 = np.zeros(direct.shape, dtype=np.int8)
            direct2[np.bitwise_and(np.abs(direct) >= 1e-9, direct < 0)] = 1
            value_h = value_h * direct2

            # 水平和垂直梯度相等
            value_hv = (np.sum(A1[ch, :, :, r - 1, c:c + 3], axis=2) +
                        np.sum(A1[ch, :, :, r + 3, c:c + 3], axis=2) +
                        np.sum(A1[ch, :, :, r:r + 3, c - 1], axis=2) +
                        np.sum(A1[ch, :, :, r:r + 3, c + 3], axis=2)) / 12

            direct3 = np.zeros(direct.shape, dtype=np.int8)
            direct3[np.abs(direct) < 1e-9] = 1
            value_hv = value_hv * direct3

            value = value_v + value_h + value_hv
            value = np.repeat(np.repeat(value.reshape((40, 53, 1, 1)), 3, axis=2), 3, axis=3)

            mean_value = np.mean(A1[ch_peer, :, :, r:r + 3, c:c + 3], axis=(2, 3)).reshape((40, 53, 1, 1))
            mean_value = np.repeat(np.repeat(mean_value, 3, axis=2), 3, axis=3)

            A1[ch, :, :, r:r + 3, c:c + 3] = A1[ch_peer, :, :, r:r + 3, c:c + 3] / mean_value * value

            pass

    img_out[img_out > 65535] = 65535
    img_out[img_out < 0] = 0

    return img_out[:, padding:padding + H, padding:padding + W]