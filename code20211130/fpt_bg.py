import numpy as np
# import minpy.numpy as np2
import cv2

from fpt_consts import FIX_VIDEO_SIZE

BG_REF_FRAME_COUNT = 100
NORMAL_BLACK = 180 # TODO: 255

DROP_BG = True #!!!!

def sub_img(img, roi):
    return img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

def calc_bg(cap, start_frame=0, end_frame=0):
    from tqdm import tqdm
    print("calc bg...")
    if not end_frame:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    remove_frame = int((end_frame - start_frame) / 3)
    start_frame += remove_frame
    end_frame -= remove_frame
    step = max(1, int((end_frame - start_frame) / BG_REF_FRAME_COUNT))
    img_a = []
    for seq in tqdm(range(start_frame, end_frame, step)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, seq)
        ret, img = cap.read()
        if not ret:
            break
        img_gray = img[:, :, 1]
        if FIX_VIDEO_SIZE:
            img_gray = cv2.resize(img_gray, FIX_VIDEO_SIZE)
        img_a.append(img_gray)
        # img_bg = np.maximum(img_bg, img_gray)

    # img_bg = np.median(img_a, 0)
    img_bg = np.max(img_a, 0)
    return img_bg.astype(float)

def fill_center_black(img_bg):
    gray = np.median(img_bg)
    center_shape = (int(img_bg.shape[0] / 2.5), int(img_bg.shape[1] / 2.5))
    center = np.empty(center_shape)
    center.fill(gray)
    pad1 = int((img_bg.shape[0] - center_shape[0]) / 2)
    pad2 = int((img_bg.shape[1] - center_shape[1]) / 2)
    mask = np.pad(center, ((pad1, img_bg.shape[0] - pad1 - center_shape[0]), (pad2, img_bg.shape[1] - pad2 - center_shape[1])), "constant", constant_values=0)
    return np.maximum(img_bg, mask)

def remove_bg2(img_gray, img_bg):
    # return remove_bg(img_gray, img_bg)
    # return 255.0 - np.fabs(img_bg - img_gray.astype(float))
    return norm_subbg(np.fabs(img_bg - img_gray.astype(float)))

def remove_bg3(img_gray, img_bg):  # NOTE: dark food make fly body brighter, fixed by img_bg_center_uniform
    return norm_subbg(np.fabs(img_bg - img_gray))

MAX_DIFF, MIN_DIFF = 30, 10
DIFF_D = MAX_DIFF - MIN_DIFF
def remove_bg(img_gray, img_bg, need_pred=True):
    r_bg = 255.0 - img_bg  # <70
    r_gray = 255.0 - img_gray  # fg>70
    r_subbg = r_gray - r_bg
    return norm_subbg(r_subbg)
    p_bg = np.clip((MAX_DIFF - r_subbg) / DIFF_D, 0, 1)
    r_fg = r_gray - r_bg * p_bg #0.007s
    i_fg = norm_subbg(r_fg)
    if need_pred:
        img_pred = norm_subbg(r_subbg)
    else:
        img_pred = None
    return i_fg, img_pred # 0.014s

def norm_subbg(r_subbg):# 0.008s
    r_subbg = np.clip(r_subbg, 0, 255)
    black = np.max(r_subbg)
    r_norm = r_subbg * (NORMAL_BLACK / black)
    return 255.0 - r_norm

def norm_img(img):
    black = np.max(img)
    r_norm = img * (255.0 / black)
    return r_norm

def imshow_test(img, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import NoNorm
    plt.figure("test")
    plt.imshow(img, cmap=plt.cm.gray, norm=NoNorm())
    if path:
        plt.savefig(path)
    else:
        plt.show()
