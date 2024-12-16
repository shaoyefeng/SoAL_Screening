# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta

START_FRAME = 0  # test
FRAME_STEP = 0  # test
DURATION_LIMIT = 60 * 60  # 1 hour
FIX_VIDEO_SIZE = None#1920, 1080  # test
ARENA_ROWS = 4#2  # test
PRINT_FRAME = 1500
PRINT_FRAME1 = PRINT_FRAME - 1

VIDEO_TODO_DIR = r'/media/syf/DataRecovery/LC_screening/SYF/LC25/video'
ALL_VIDEO_DIR = ["_video_hrnet_finish", "_video_screen_finish", "video_suzukii_finish",
                 "video_hd", "_video_ort"]
MIN_DURATION = 50 * 60
# NOTE: config for PC
IMG_QUEUE_CAPACITY = 200  #60 # max BATCH_SIZE 15000 ~15000*20*128*102~32G
IMG_PROCESS_CAPACITY = 50  #30 # NOTE: for multi img process (fpt_1) 1280*1024
# NOTE: if cv2.MemoryError, lower gpu_options.per_process_gpu_memory_fraction
# NOTE: fpt_1
#   NOTE: limit 1xGeForce GTX 1060 (1,2,2,4) ~2000fly/s
#   NOTE: limit 2xGeForce RTX 2070 (2,4,6,4) ~4800fly/s
#   NOTE: limit 2xGeForce RTX 2060 (2,4,4,7) ~2400fly/s
GPU_N = 2 #2
PRED_PROCESS_NUM = 4 #4
MAX_TASK = 4  # NOTE: fpt:4~6, fpt_1:2
IMG_PROCESS_NUM = 4  # NOTE: for multi img process (fpt_1) max=20

POOL_SIZE = 16

# NOTE: config for server fpt_1
# IMG_QUEUE_CAPACITY = 60
# PRED_PROCESS_NUM = 6
# GPU_N = 2
# MAX_TASK = 6
# IMG_PROCESS_NUM = 4
# IMG_PROCESS_CAPACITY = 30
# USE_ASYNC = True

# NOTE: config for server fpt
# PRED_PROCESS_NUM = 6
# MAX_TASK = 18

REL_POLAR_T_OFFSET = 0

MODEL_W = 48
MODEL_FOLDER = "./nn_dlc/models/R2_48"  # 96

MODEL_SHAPE = (MODEL_W, MODEL_W)
MODEL_SHAPE_EXTEND = (int(MODEL_SHAPE[0] * 1.42 + 2), int(MODEL_SHAPE[1] * 1.42 + 2))  # NOTE: black point on the corner
# MODEL_SHAPE = 64, 48
# MODEL_SHAPE_EXTEND = (int(MODEL_SHAPE[0] * 1.5 + 3), int(MODEL_SHAPE[0] * 1.5 + 3))

FEAT_SCALE_NORMAL = MODEL_W / 4

# NOTE: 1 for Head/Thor/Abdo
FLY_NUM = 2

if FLY_NUM == 1:
    FEMALE_AVG_LEN_L = 0
else:
    FEMALE_AVG_LEN_L = 1.6
FEMALE_AVG_LEN_H = 3.0

FEMALE_AVG_LEN = 2.5
FLY_AVG_WID = 1
MALE_AVG_LEN = 2.2

CS_MEAN_FLY_LENGTH = 2.32

# during circling BODY_SIZE
# CS: (2.337, 0.958), (2.176, 0.899), (2.499, 1.017)
# IR: (2.298, 0.871), (2.155, 0.796), (2.441, 0.946)
# A: (2.325, 0.951), (2.159, 0.898), (2.491, 1.004)
BODY_SIZE = [(2.4, 0.8), (2.2, 0.78), (2.53, 0.87)]
BODY_LEN_CENTER_CIR = [2.3196883920443376, 2.155937577339063, 2.4834392067496123]  #NOTE: (M+F)/2, M, F
FLY_AVG_AREA = 1.6 #TODO
DIST_TO_FLY_FAR = 2.7  # 无接触距离
DIST_TO_FLY_NEAR1 = 2.1
DIST_TO_FLY_NEAR2 = 2.65
DIST_TO_FLY_NEAR3 = 2.65
DIST_TO_FLY_NEAR4 = 3.2
HEAD_MEDIAN_DIST = 3.9678  # head正对时平均距离
HEAD_MEDIAN_ANGULAR_SIZE = 12.5130  # head正对时angular size

DIST_TO_FLY_INNER_T = 2.5  # 内圈大小(dist_McFc)
DIST_TO_FLY_INNER_H = 3  # 内圈大小(dist_McFc)
# 挑蛹日期第一天, EXP_MALE_DAY1+"C1"[1]计算结果为羽化天数
# no_map: "_geno_days" "_TB>CS_4"
EXP_MALE_GENO_MAP = {
    #"c": ("CS", "20200701"),
    "l": ("LC10adX", "20201031"),
    "t": ("TBX", "20201127"),

    "C": ("Ctrl", "20201109"),
    "E": ("Exp", "20201109"),
    "B": ("Biar", "20201031"),
    "H": ("Head", "20210617"),
    "W": ("HeadW", "20210618"),
    "M": ("HeadM", "20210618"),
    "A": ("Abdo", "20210617"),
    "T": ("Thor", "20210617"),
}
EFFECTOR_MAP = {
    "c": "CS",
    "s": "Shi",
    "t": "Trp",
    "X": "Two",
    "Y": "One",
    "Z": "None",
	"S": "",
}
def code_to_geno(code):
    g = code[0]
    geno, day1 = EXP_MALE_GENO_MAP.get(g, (None, None))
    if not geno:
        return "", 0
    if len(code) > 1:
        eff = code[1]
        return geno + EFFECTOR_MAP[eff], day1
    else:
        return geno, day1

# NOTE: config for UI
DEFAULT_TWO_POINT_LEN = 20
DEFAULT_GRAY_THRESHOLD = 140
FPS = 66
VERSION = 20191026
ROUND_ARENA = True
POINT_NUM = 5

DIST_TO_CENTER_THRESHOLD = 7
DIST_TO_CENTER_THRESHOLD_FEMALE = 5.5
DIST_TO_CENTER_THRESHOLD_MALE = 6.5

TURN_BINS = 9
FINE_CIR_MALE_MIN_V = 1
FINE_CIR_FEMALE_MAX_V = 5#3#
FINE_CIR_FEMALE_MAX_AV = 120#60#

FA_CIR_FEMALE_MIN_V = 5#1
FA_CIR_FEMALE_MIN_AV = 120


def str2day(s):
    return datetime.strptime(s, "%Y%m%d")

def day_diff(d2, d1):
    return (d2 - d1).days

def day_str_diff(s2, s1):
    return day_diff(str2day(s2), str2day(s1))

def day_add_s(s1, i):
    return day2str(str2day(s1) + timedelta(days=i))

def day2str(d):
    return datetime.strftime(d, "%Y%m%d")

def time_now_str():
    return datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")

def tt_to_second(tt):
    return int(tt[0])*3600+int(tt[1])*60+int(tt[2])

def second_to_tt(s):
    return int(s/3600), int(s%3600/60), int(s%60)


