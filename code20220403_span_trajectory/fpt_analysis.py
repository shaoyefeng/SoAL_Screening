# -*- coding: utf-8 -*-

import os
import sys
import time
import shutil
import cv2
import seaborn as sns
import pandas as pd
from multiprocessing import Pool

from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.anova import anova_lm
from scipy.stats.mstats_basic import kruskalwallis
import scikit_posthocs as sp
from scipy import stats
from tqdm import tqdm

from fpt_frame_stat import STATE_SIDE_WALK
#from manuscript_geno import GENO_DIR
from fpt_circling import extract_avi, calc_all_cir_info, do_detect_behavior, do_extract_avi, do_extract_stat
from fpt_plot import *
from fpt_consts import DIST_TO_CENTER_THRESHOLD_MALE, DIST_TO_CENTER_THRESHOLD_FEMALE, time_now_str, \
    DIST_TO_FLY_INNER_H, \
    DIST_TO_FLY_INNER_T, MIN_DURATION, POOL_SIZE, BODY_LEN_CENTER_CIR

# DATA_DIR = "G:/data_hrnet/center"
# DATA_DIR = "F:/temp/data_part/center"
DATA_DIR = r"E:/all/center"
VIDEO_DIR = r"/media/syf/SYF/todo/"                 
DRIVER_L = ["CSXTrp31 T45XTrp31 T45XShi31"]
# DATA_DIR = "G:/data_screen/center"
# DATA_DIR = "G:/data3s/center"
# DATA_DIR = "G:/data_LPLC1/center"

GENO_DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "geno_data") #"E:/data2/geno_data"#
INFO_DIR = os.path.join(DATA_DIR, "_info")

MIN_BOUTS = 5
CAN_SKIP = False
PLOT_CIR = True
PLOT_WE = False
PLOT_DIFF = False
PLOT_WHOLE = False

PLOT_DRIVER_GROUP = 4
PLOT_DRIVER_SIG = True
PLOT_DRIVER_PVAL = False
PLOT_ALL_PVAL = False
USE_ADD_SUB_GENO_XLABEL = False

ONLY_UPDATE_GENO = []#"FBSNPXCS31", "LC16XCS31", "LC12XCS31",
                    #"FBSNPXShi31", "LC16XShi31", "LC12XShi31",
                    #"FBSNPXTrp31", "LC16XTrp31", "LC12XTrp31"]#["CS22", "IR22", "MW22", "FW22", "MWIR22", "A22", "AIR22"]
# ONLY_UPDATE_GENO = ["CSXCS31", "CSXTrp31", "CSXShi31", # CS31
#                     "LC24XCS31", "LC24XShi31", "LC24XTrp31",
#                     "LC26XCS31", "LC26XShi31", "LC26XTrp31",
#                     "LC6XCS31", "LC6XShi31", "LC6XTrp31",
#                     "LC18XCS31", "LC18XShi31", "LC18XTrp31"] #test WIPS
# ONLY_UPDATE_GENO = ["L11XCS31", "LC11XShi31", "LC11XTrp31",
#                     "LC20XCS31", "LC20XShi31", "LC20XTrp31",
#                     "LPLC2XCS31", "LPLC2XShi31", "LPLC2XTrp31"]
UPDATE_EXIST_STAT = False
LOAD_EXIST_STAT = True #test WIPS

SWARM_PALATTE = ["r", "g", "b", "y", "c"]
SWARM_WIDTH_SCALE = 1.2
SWARM_HEIGHT_SCALE = 1
DF_COLUMNS = [
    "0:dist_McFc", "2:rel_polar_h:r", "2:rel_polar_h:t", "2:rel_pos_h:x", "1:theta",  # dist
    "1:rel_polar_hh:t", "1:rel_polar_h:r", "1:rel_pos:x", "1:rel_pos:y", "1:rel_polar_h:t", "1:rel_polar:r", "1:rel_polar:t", "1:rel_polar_ht:t",  # angle
    "1:abs_acc", "1:acc_len", "1:abs_av", "1:abs_theta", "1:abs_v_dir", "1:v_len", "1:abs_vs", "1:vf",  # speed
    "1:we_l", "1:we_r", "1:we_lr", "1:wing_m", "1:we_ipsi",  # wing
    "1:e_maj", "2:e_maj", "2:on_edge", "2:v_len", "1:court", "1:dist_c", "2:dist_c",  # condition
    "1:rel_polar_hh:r", "1:rel_polar_ht:r", "1:walk", "1:nh", "1:nt", "1:on_edge",
    # "rel_pos:x", "rel_pos:y", "rel_polar:r", "rel_polar:t", "rel_pos_h:x", "rel_pos_h:y",
    # "rel_polar_h:r", "rel_polar_h:t", "rel_pos_t:x", "rel_pos_t:y", "rel_polar_t:r", "rel_polar_t:t",
    # "rel_polar_hh:r", "rel_polar_hh:t", "rel_pos_ht:x", "rel_pos_ht:y",
    # "rel_polar_ht:r", "rel_polar_ht:t", "rel_pos_th:x", "rel_pos_th:y", "rel_polar_th:r", "rel_polar_th:t",
]

# "0:dist_McFc", "2:rel_polar_h:r", "2:abs_rel_pos_h:x",  # dist
# "1:rel_polar_hh:t", "1:rel_polar_ht:t",  # angle
# "1:abs_acc", "1:abs_av", "1:abs_theta", "1:abs_v_dir", "1:v_len", "1:abs_vs", "1:vf",  # speed
# "1:we_l", "1:we_r", "1:we_lr", "1:wing_m", "1:we_ipsi",  # wing
# "1:e_maj", "2:e_maj",
KEY_TYPE_KEYS = {
    "00": ["cir_per_second", "cir_ratio", "cir_duration", "w_we_ratio", "cir_with_cond_ratio"],
    "01": ["cir_per_second", "cir_ratio", "cir_duration", "w_we_ratio", "cir_with_cond_ratio"],
    "02": ["w_static_ratio", "w_vf_walk_m", "w_male_len", "female_days", "male_days", "duration"],
    "mc": ["0:dist_McFc", "2:rel_polar_h:r", "1:rel_polar_ht:r", "1:nh", "1:nt", "1:nh_index", "1:dh_nh/dt_nt", "1:d_to_h/d_to_t", "1:p_cw", "1:p_to_h", "1:abs_acc", "1:abs_av", "1:v_len", "1:abs_vs", "1:vf", "1:we_l", "1:we_lr", "1:wing_m", "1:we_ipsi", "1:e_maj", "2:e_maj"],
    "mw": ["0:dist_McFc", "2:rel_polar_h:r", "1:rel_polar_ht:r", "1:nh", "1:nt", "1:abs_acc", "1:abs_av", "1:abs_theta", "1:v_len", "1:abs_vs", "1:vf", "1:we_l", "1:we_lr", "1:we_ipsi", "1:e_maj", "2:e_maj"],
    "me": ["0:dist_McFc", "2:rel_polar_h:r", "1:rel_polar_ht:r", "1:nh", "1:nt", "1:abs_acc", "1:abs_av", "1:abs_theta", "1:v_len", "1:abs_vs", "1:vf", "1:we_ipsi"],
    "mef": ["1:nh", "1:nt", "1:abs_acc", "1:abs_av", "1:abs_theta", "1:v_len", "1:we_ipsi"],
    "mnh": ["1:rel_polar:r", "1:sd_rel_polar:t", "1:rel_polar:t", "1:abs_acc", "1:abs_av", "1:abs_theta", "1:v_len", "1:abs_vs", "1:vf", "1:we_lr", "1:we_ipsi"],
    "mnt": ["1:rel_polar_ht:r", "1:rel_polar_ht:t", "1:abs_acc", "1:abs_av", "1:abs_theta", "1:v_len", "1:abs_vs", "1:vf", "1:we_lr", "1:we_ipsi"],
    "mih": ["1:rel_polar_hh:r", "1:rel_polar_hh:t", "1:abs_acc", "1:abs_av", "1:abs_theta", "1:v_len", "1:abs_vs", "1:vf", "1:we_lr", "1:we_ipsi"],
    "mit": ["1:rel_polar_ht:r", "1:rel_polar_ht:t", "1:abs_acc", "1:abs_av", "1:abs_theta", "1:v_len", "1:abs_vs", "1:vf", "1:we_lr", "1:we_ipsi"],
    "mnl": ["0:dist_McFc", "1:abs_acc", "1:abs_av", "1:abs_theta", "1:v_len", "1:abs_vs", "1:vf", "1:we_lr", "1:we_ipsi"],
    "mil": ["0:dist_McFc", "1:abs_acc", "1:abs_av", "1:abs_theta", "1:v_len", "1:abs_vs", "1:vf", "1:we_lr", "1:we_ipsi"],
}

DF_KEYS = [], [], []
for k in DF_COLUMNS:
    DF_KEYS[int(k[0])].append(k[2:])
# NOTE: mean, mean_angle, mean_nzero, median(default)
DF_MEAN_TYPE = {
    "rel_polar:t": "mean_angle",
    "rel_polar_h:t": "mean_angle",
    "rel_polar_hh:t": "mean_angle",
    "rel_polar_ht:t": "mean_angle",
    "we_ipsi": "mean_nzero",
    "we_l": "mean",
    "we_r": "mean",
    "we_lr": "mean",
    "nh": "mean",
    "nt": "mean",
    # "v_len": "mean",
}

SUMMARY_FIGURES = [
    "c_pos-female", #"c_pos_h-female", "c_pos_t-female", # "c_pos_nh-female", "c_pos_nt-female",
    "c_pos", "c_pos_h",  # "c_pos_t", # "c_pos_h_nh", "c_pos_h_nt",
    "fc_speed", "fc_abs_av", "fc_acc", "fc_wing_m",
    "fc_theta", #"fc_vf", "fc_abs_vs",  # "fc_vs_nh", "fc_vs_nt",
    "polar_d", "a_d", "polar_d_ud", "a_d_ud", "polar_d_nhnt", # "diff_a_d_ud",
    # "d_dist", "a_c_dev", "a_speed", "a_abs_vs", "a_e_maj", "a_wing_m",
    "wc_head", #"wc_roc_head", #"we_head_x_nh", "we_head_x_nt",
    "near_dmin_cc",
    "d_rel_polar_h", "d_se_polar-female", #"d_cir", #"se_pos", "c_cir_s-female", "c_cir_e-female",
    "switch_pos_mv", "d_switch",
    "time_d_align_all-male-rel_polar:r", "time_d_align_ud-male-rel_polar:r",
    "time_d_align_all-male-speed", "time_d_align_ud-male-speed",
    "time_d_align_all-male-we_ipsi", "time_d_align_ud-male-we_ipsi", #"time_d_align_all-male-abs_av", "time_d_align_all-male-abs_theta", "time_d_align_all-male-wing_m",
    # "trigger_stop-male-rel_polar:r", "trigger_stop-male-v_len", "trigger_stop-male-vf",# "trigger_stop-male-theta",
    # "c_we_ipsi", "c_we_contra", "r_theta_wing_t", "r_d_v_we",

    # "d_e_maj", #"d_e_min", "d_e_maj-female", "d_e_min-female", "r_vs_theta", "stop_trigger_track_len",
    # "stream_v_dir-female", "stream_dir-female"
    "fc_we_ipsi", "heat_theta_wing_toh", "heat_theta_wing_tot", "wci_dist",
]
SUMMARY_FIGURES = ['partial_track_len']#'fc_abs_theta'
#"c_pos-female", "c_pos_h", "c_pos_t", "fc_speed", "fc_abs_av", "fc_acc", "fc_wing_m", "fc_theta", "fc_vf", "fc_abs_vs", "fc_we_lr", "fc_contact",
# "d_speed", "d_av", "d_theta", "t_polar_h", "d_wing", "c_track", "se_pos-female",
# SUMMARY_FIGURES = ["c_pos-female", "fc_speed"]

"""
[swarm]
mc: median during circling
me: median during wing extension
mih/mil/mit: median inner head/lateral/tail
mnh/mnl/mnt: median near(45 degree range) head/lateral/tail

[fly]
0: other
1: male
2: female

[property]
2:rel_polar_h:r
    female centric male"s head distance (dist_MhFc)
1:rel_polar_ht:r
    male centric male"s head to female"s tail distance (dist_MhFt)
1:rel_polar_hh:r
    male centric male"s head to female"s head distance (dist_MhFh)
v_len/v_dir
    speed magnitude/direction
vs/vf
    side/forward velocity
we_l
    ratio of left wing extension
we_lr
    ratio of bilateral wing extension
wing_m
    max(left wing theta, right wing theta)
we_ipsi
    ipsi ratio - contra ratio (1 when ipsi, -1 when contra, 0 other)
e_maj   
    ellipse major axis length (body length)
    
[directory structure]
exp(D:\exp)
   |-data
        |-geno_data
            |-CS31
                |-_CS31_all.txt
                |-_CS31_center.txt
                    |-m_mw_{}, m_mc_{}
                    |-pairs:
                        |-20191009_140000_A_0: mw_{}(mean_whole), mc_{}(mean_cir), pair_info
                |-_CS31_cir_all.csv
                |-_CS31_cir_center.csv
                |-20191009_140000_A_0
                    |-20191009_140000_A_0_all.txt
                    |-20191009_140000_A_0_center.txt
                        |-cir_per_second, ..., mc_..., ...
                    |-20191009_140000_A_0_cir.avi
                    |-20191009_140000_A_0_cir_center.avi
                    |-20191009_140000_A_0_cir_info.csv
                    |-20191009_140000_A_0_cir_info_center.csv
                    |-20191009_140000_A_0_cir_meta.txt
                    |-20191009_140000_A_0_cir_meta_center.txt
                    |-20191009_140000_A_0_cir_meta.txt
                    |-20191009_140000_A_0_cir_info.csv
                    |-20191009_140000_A_0_stat0.pickle
                    |-20191009_140000_A_0_stat1.pickle
                    |-20191009_140000_A_0_stat2.pickle
                    |-20191009_140000_A_0_cir_stat0.pickle
                    |-20191009_140000_A_0_cir_stat1.pickle
                    |-20191009_140000_A_0_cir_stat2.pickle
                    |-20191009_140000_A_0_cir_center_stat0.pickle
                    |-20191009_140000_A_0_cir_center_stat1.pickle
                    |-20191009_140000_A_0_cir_center_stat2.pickle
                |-20191009_140000_A_1
                |-...
            |-...
       |-all
       |-center
            |-_cir (cir_bar, cor)
            |-_etho (etho)
            |-_driver (driver_bar,swarm)
            |-_pair
                |-CS31-summary.png
                |-...
                |-we
                    |-CS31-wesummary.png
                    |-...
                |-compare
                    |-a_abs_vs.png
                    |-...
                    |-diff
                        |-a_abs_vs_diff.png
                        |-...
       |-good
"""

def get_geno_str(meta):
    return meta["ROI"]["male_geno"].replace(">", "X").replace("/", "") + str(meta["temperature"])

def update_all_cir_info():
    for f in os.listdir(DATA_DIR):
        p = os.path.join(DATA_DIR, f)
        if os.path.isdir(p) and not f.startswith("_"):
            for ff in os.listdir(p):
                if ff.startswith("20"):
                    update_cir_info(p, ff)
        clean_save_cache()

def update_cir_info(parent, pair_name):
    pair_folder = os.path.join(parent, pair_name)
    return update_geno_folder_cir_info(parent, pair_name, load_dataframe(os.path.join(pair_folder, pair_name + "_cir_info.csv")))
    geno = os.path.basename(parent)
    dfs, bouts, n = load_dfs_bouts(pair_folder)
    cir_info = calc_all_cir_info(dfs, {"cir_bouts1": bouts})
    cir_info["pair"] = pair_name
    cir_info["geno"] = geno
    cir_info["cir_with_cond"] = len(bouts)
    cir_info.to_csv(os.path.join(pair_folder, pair_name + "_cir_info.csv"))
    update_geno_folder_cir_info(parent, pair_name, cir_info)

def update_from_raw_data(pair_folder_l, force=True):
    mkdir(GENO_DATA_DIR)
    log_path = os.path.join(GENO_DATA_DIR, "log.csv")
    log = pd.read_csv(log_path, index_col=0).to_dict("index") if os.path.exists(log_path) else {}
    for pair_folder in tqdm(pair_folder_l):
        exp_folder, pair_no = os.path.split(pair_folder)
        exp_name = os.path.basename(exp_folder)
        # if exp_name.startswith("2020"):
        #     print("skip2", exp_name)
        #     continue
        pair_name = exp_name + "_" + pair_no
        prefix = os.path.join(pair_folder, pair_name)
        if log.get(pair_name) and not force:
            print("skip0 in log:", pair_name)
            continue
        # NOTE: copy files
        meta = load_dict(prefix + "_meta.txt")
        geno_str = get_geno_str(meta)
        if ONLY_UPDATE_GENO and geno_str not in ONLY_UPDATE_GENO:
            print("skip1 not in ONLY_UPDATE_GENO:", geno_str)
            continue
        geno_pair_folder = os.path.join(GENO_DATA_DIR, geno_str, pair_name)
        print("update_from_raw_data", geno_pair_folder)
        os.makedirs(geno_pair_folder)
        stat_prefix = os.path.join(geno_pair_folder, pair_name)
        if os.path.exists(stat_prefix + "_stat0.pickle"):
            print("skip2 already exists:", stat_prefix)
            continue
        if os.path.exists(prefix + "_cir_meta.txt"):
            shutil.copy2(prefix + "_cir_meta.txt", geno_pair_folder)
            shutil.copy2(prefix + "_stat0.pickle", geno_pair_folder)
            shutil.copy2(prefix + "_stat1.pickle", geno_pair_folder)
            shutil.copy2(prefix + "_stat2.pickle", geno_pair_folder)
            dfs, cir_meta = None, None
        else:
            feat_file = prefix + "_feat.csv"
            dfs, cir_meta = do_detect_behavior(feat_file, stat_prefix)
        modify_update_log(log, pair_name, geno_pair_folder, 1)

        # if ex_avi: # NOTE: extract cir
        #     if len(meta["cir_bouts1"]):
        #         video = get_video_in_dir(os.path.dirname(pair_folder))
        #         if os.path.exists(video):
        #             prefix_d = os.path.join(geno_pair_folder, pair_name)
        #             extract_avi(video, meta["cir_bouts1"], meta["ROI"]["roi"], prefix_d + "_cir.avi", extend_pixel=20)

def update_one_pair(pair_folder, force=UPDATE_EXIST_STAT):
    exp_folder, pair_no = os.path.split(pair_folder)
    exp_name = os.path.basename(exp_folder)
    # if exp_name.startswith("2020"):
    #     print("skip0", exp_name)
    #     continue
    pair_name = exp_name + "_" + pair_no
    prefix = os.path.join(pair_folder, pair_name)
    meta = load_dict(prefix + "_meta.txt")
    if not meta:
        return
    geno_str = get_geno_str(meta)
    if ONLY_UPDATE_GENO and geno_str not in ONLY_UPDATE_GENO:
        print("skip1 not in ONLY_UPDATE_GENO:", geno_str)
        return
    geno_pair_folder = os.path.join(GENO_DATA_DIR, geno_str, pair_name)
    print("update_one_pair", geno_pair_folder)
    os.makedirs(geno_pair_folder, exist_ok=True)
    stat_prefix = os.path.join(geno_pair_folder, pair_name)
    if pair_name in "20201231_144143_E_9,20201231_154730_F_13,20210101_160524_E_14,20210101_160524_F_8,20210105_153721_F_9,20210105_153721_F_11".split(","):
        print("skip3---------")
        return
    if os.path.exists(stat_prefix + "_stat2.pickle") and not force:
        print("skip2 already exists:", stat_prefix)
        return
    feat_file = prefix + "_feat.csv"
    dfs, cir_meta = do_detect_behavior(feat_file, stat_prefix, not LOAD_EXIST_STAT)  # NOTE: save stat0 and cir_meta (disk busy)
    dfs = dfs_calc_abs(dfs)
    if FLY_NUM > 1:
        dfs = dfs_scale_pos(dfs)  # NOTE: calc_abs and scale_pos by female length

    # NOTE: optional
    # do_extract_avi(stat_prefix + "_cir_meta.txt", cir_meta)
    # do_extract_stat(stat_prefix + "_cir_meta.txt", cir_meta, dfs)

    update_ana_data_pair(geno_pair_folder, geno_str, pair_name, dfs, cir_meta)  # NOTE: save cir_stat0...

def update_geno_one_pair(cir_meta_file):
    do_extract_avi(cir_meta_file)
    # do_extract_stat(cir_meta_file)

def get_fps(meta_file):
    meta = load_dict(meta_file)
    return meta["FPS"]
    # update FPS
    meta["FPS"] = 66
    save_dict(meta_file, meta)
    return meta["FPS"]

def update_ana_data_pre(geno=None):
    geno_l = [geno] if geno else os.listdir(GENO_DATA_DIR)
    for geno in tqdm(geno_l):
        geno_folder = os.path.join(GENO_DATA_DIR, geno)
        if os.path.isdir(geno_folder):
            for pair_name in os.listdir(geno_folder):
                geno_pair_folder = os.path.join(geno_folder, pair_name)
                if not os.path.isdir(geno_pair_folder):
                    continue
                print((geno, geno_pair_folder, pair_name))
                prefix = os.path.join(geno_pair_folder, pair_name)
                cir_meta = load_dict(prefix + "_cir_meta.txt")
                cop = cir_meta["copulate"]
                cir_meta_center = load_dict(prefix + "_cir_meta_center.txt")
                cir_meta_center["copulate"] = cop
                save_dict(prefix + "_cir_meta_center.txt", cir_meta_center)

                center = load_dict(prefix + "_center.txt")
                center["cop_bouts"] = cop
                save_dict(prefix + "_center.txt", center)

                center = load_dict(prefix + "_all.txt")
                center["cop_bouts"] = cop
                save_dict(prefix + "_all.txt", center)
                # dfs = load_dfs(prefix + "_stat0.pickle")
                # for fly in (1, 2):
                #     dfs[fly]["rel_polar:t"] = lim_dir_a(dfs[fly]["rel_polar:t"]-90)
                #     dfs[fly]["rel_polar_h:t"] = lim_dir_a(dfs[fly]["rel_polar_h:t"]-90)
                #     dfs[fly]["rel_polar_t:t"] = lim_dir_a(dfs[fly]["rel_polar_t:t"]-90)
                #     dfs[fly]["rel_polar_hh:t"] = lim_dir_a(dfs[fly]["rel_polar_hh:t"]-90)
                #     dfs[fly]["rel_polar_ht:t"] = lim_dir_a(dfs[fly]["rel_polar_ht:t"]-90)
                # fps = get_fps(prefix + "_cir_meta.txt")
                # d = int(fps/30 + 0.5)
                # fps_scale = fps / (d*2)
                # df = dfs[1]
                # if "acc_len" in df.keys():  # NOTE: skip
                #     continue
                # i = df.index
                # dfb = df.reindex(i - d, method="nearest")
                # dff = df.reindex(i + d, method="nearest")
                # xb, yb = np.array(dfb["pos:x"]), np.array(dfb["pos:y"])
                # xf, yf = np.array(dff["pos:x"]), np.array(dff["pos:y"])
                # vx, vy = xf - xb, yf - yb
                # v_len = np.sqrt(vx**2 + vy**2) * fps_scale
                # v_dir = np.rad2deg(np.arctan2(vy, vx))
                # accx, accy = (vx[d:] - vx[:-d]), (vy[d:] - vy[:-d])
                # acc_len = np.sqrt(accx**2, accy**2) * fps_scale
                #
                # df["v_len"] = v_len
                # df["v_dir"] = v_dir
                # theta = df["theta"]
                # theta_r = np.deg2rad(theta)
                # df["vs"] = v_len * np.sin(theta_r)
                # df["vf"] = v_len * np.cos(theta_r)
                # df["av"] = angle_diff(dff["dir"], dfb["dir"]) * fps_scale
                # df["acc"] = np.hstack([[np.nan]*d, (v_len[d:] - v_len[:-d]) * fps_scale])
                # df["acc_dir"] = np.hstack([[np.nan]*d, np.rad2deg(np.arctan2(accy, accx))])
                # df["acc_len"] = np.hstack([[np.nan]*d, acc_len])
                # df["walk"] = walk_state(v_len, abs(theta))

                # save_dfs(dfs, prefix)

# G = {}
def update_ana_data(geno=None):
    # log_path = os.path.join(GENO_DATA_DIR, "log.csv")
    # log = pd.read_csv(log_path, index_col=0).to_dict("index")
    geno_l = [geno] if geno else os.listdir(GENO_DATA_DIR)
    # geno_l = geno_l[geno_l.index("LC6XTrp31")-1:]  # test
    pool = Pool(POOL_SIZE)
    for geno in tqdm(geno_l):
        geno_folder = os.path.join(GENO_DATA_DIR, geno)
        if os.path.isdir(geno_folder):
            # G[0] = load_dict(os.path.join(geno_folder, "_%s_all.txt" % geno))
            for pair_name in os.listdir(geno_folder):
                geno_pair_folder = os.path.join(geno_folder, pair_name)
                if not os.path.isdir(geno_pair_folder):
                    continue
                # if CAN_SKIP and log.get(pair_name, {}).get("stage") == 2:
                #     print("skip", pair_name)
                #     continue
                # if pair_name.startswith("2020"):
                #     print("update_ana_data skip2", pair_name)
                #     continue
                if USE_ANA_ASYNC:
                    pool.apply_async(update_ana_data_pair, (geno, geno_pair_folder, pair_name))
                else:
                    update_ana_data_pair(geno, geno_pair_folder, pair_name)
                # modify_update_log(log, pair_name, geno, 2)
    pool.close()
    pool.join()

def pre_bouts(bouts, fps, duration=1):
    frames = fps * duration
    ret = []
    for s, e in bouts:
        if s >= frames:
            ret.append((s - frames + 1, s + 1))
    return ret

def remove_bouts(meta):
    ks = [k for k in meta.keys() if k.endswith("bouts")]
    for k in ks:
        del meta[k]

def update_ana_data_pair(geno_pair_folder, geno=None, pair_name=None, dfs=None, cir_meta=None, only_update=False):
    pair_name = pair_name or os.path.basename(geno_pair_folder)
    geno = geno or os.path.basename(os.path.dirname(geno_pair_folder))
    print("update_ana_data_pair", pair_name)
    # NOTE: calc cir info
    stat_prefix = os.path.join(geno_pair_folder, pair_name)
    # if True:
    #     dict_name = stat_prefix + "_all.txt"
    #     if load_dict(dict_name)["cir_with_cond_ratio"] < 1:
    #         print(dict_name)
    #         save_dict(dict_name, G[0]["pairs"][pair_name])
    #     return
    # if os.path.exists(stat_prefix + "_cir_stat0.pickle"):  #test
    #     mtime = os.stat(stat_prefix + "_cir_stat0.pickle").st_mtime
    #     if time.time() - mtime < 3600*12:
    #         print("skip3 already exists:", stat_prefix)
    #         return
    # dict_name = stat_prefix + "_all.txt"
    # pair_info = load_dict(dict_name)
    # pair_info["ici"] = calc_ici(pair_info["cir_bouts"]) / pair_info["fps"]
    # return save_dict(dict_name, pair_info)
    cir_meta = cir_meta or load_dict(stat_prefix + "_cir_meta.txt")
    dfs = dfs or load_dfs(stat_prefix + "_stat0.pickle", need_abs_and_scale=True)  # NOTE: correct pos by female length

    cir_info = calc_all_cir_info(dfs, cir_meta)
    cir_bouts1 = cir_meta["cir_bouts1"]
    cir_info["pair"] = pair_name
    cir_info["geno"] = geno
    cir_info["cir_with_cond"] = len(cir_bouts1)
    cir_info.to_csv(stat_prefix + "_cir_info.csv")
    # NOTE: center cir
    cir_bouts2, cir_info2 = cir_with_condition(cir_bouts1, cir_info, "center")
    meta2 = cir_meta.copy()
    remove_bouts(meta2)
    meta2["cir_bouts1"] = cir_bouts2
    meta2["DIST_TO_CENTER_THRESHOLD_MALE"] = DIST_TO_CENTER_THRESHOLD_MALE
    meta2["DIST_TO_CENTER_THRESHOLD_FEMALE"] = DIST_TO_CENTER_THRESHOLD_FEMALE
    meta3 = meta2.copy()
    cir_bouts3 = pre_bouts(cir_bouts2, cir_meta["FPS"], 1)
    meta3["cir_bouts1"] = cir_bouts3
    dfs1 = dfs_bouts(dfs, cir_bouts1)
    dfs2 = dfs_bouts(dfs, cir_bouts2)
    dfs3 = dfs_bouts(dfs, cir_bouts3)
    # save_dfs(dfs3, stat_prefix + "_cir_center_pre")  # NOTE: cir_center_pre_stat0.pickle
    # save_dict(stat_prefix + "_cir_meta_center_pre.txt", meta3)
    # return
    dfs_no_touch_we = dfs_no_touch(dfs_we(dfs))
    if not only_update:
        # NOTE: save center.avi
        cir_bouts2_v = np.array(squeeze_cir_bouts(cir_bouts1))[list(cir_info2.index)].tolist()  # NOTE: for extract avi
        video_cir = stat_prefix + "_cir.avi"
        if os.path.exists(video_cir):
            extract_avi(video_cir, cir_bouts2_v, None, stat_prefix + "_cir_center.avi", need_text=False)
        # return # test
        # NOTE: save cir_info_center.csv
        save_dict(stat_prefix + "_cir_meta_center.txt", meta2)
        save_dict(stat_prefix + "_cir_meta_center_pre.txt", meta3)
        cir_info2["cir_with_cond"] = len(cir_bouts2)
        cir_info2.to_csv(stat_prefix + "_cir_info_center.csv")
        # NOTE: save stat.pickle
        save_dfs(dfs1, stat_prefix + "_cir")  # NOTE: cir_stat0.pickle
        save_dfs(dfs2, stat_prefix + "_cir_center")  # NOTE: cir_center_stat0.pickle
        save_dfs(dfs3, stat_prefix + "_cir_center_pre")  # NOTE: cir_center_pre_stat0.pickle
        save_dfs(dfs_no_touch_we, stat_prefix + "_we")  # NOTE: we_stat0.pickle
        if PLOT_WHOLE:
            plot_summary_by_name(["d_dist_whole"], dfs, stat_prefix + "_whole_summary.png", col=1, save_svg=False)
    # NOTE: calc pair info
    dfs = dfs_columns(dfs, DF_KEYS)
    pair_info = {}
    if not only_update:
        calc_cir_info_mean(dfs, "mw_", pair_info)
        calc_cir_info_mean(dfs_no_touch_we, "me_", pair_info)
        calc_cir_info_mean(dfs_far(dfs_no_touch_we), "mef_", pair_info)

    for d, m, cb, data_type in [[dfs1, cir_meta, cir_bouts1, "all"], [dfs2, meta2, cir_bouts2, "center"]]:
        dict_name = stat_prefix + "_%s.txt" % data_type  # NOTE: all.txt, center.txt
        if os.path.exists(dict_name) and only_update:
            print("update", dict_name)
            pair_info = load_dict(dict_name)
            # TODO: input update code here
            pair_info = calc_pair_info_meta(dfs, m, len(cir_bouts1), len(cb), pair_info)
            # calc_cir_info_mean(dfs_far(dfs_no_touch_we), "mef_", pair_info)
            if cb and data_type == "center":
                dfs_c = dfs_columns(d, DF_KEYS)
                calc_cir_info_mean_extra(dfs_c, cb, pair_info)
        else:
            pair_info = calc_pair_info_meta(dfs, m, len(cir_bouts1), len(cb), pair_info)
            if cb:
                dfs_c = dfs_columns(d, DF_KEYS)
                calc_cir_info_mean(dfs_c, "mc_", pair_info)
                dfs_nh = dfs_head_quadrant(dfs_c)
                dfs_nt = dfs_tail_quadrant(dfs_c)
                dfs_nl = dfs_lateral(dfs_c)
                calc_cir_info_mean(dfs_nh, "mnh_", pair_info)
                calc_cir_info_mean(dfs_nt, "mnt_", pair_info)
                calc_cir_info_mean(dfs_nl, "mnl_", pair_info)
                calc_cir_info_mean(dfs_inner(dfs_nh, DIST_TO_FLY_INNER_H), "mih_", pair_info)
                calc_cir_info_mean(dfs_inner(dfs_nt, DIST_TO_FLY_INNER_T), "mit_", pair_info)
                calc_cir_info_mean(dfs_inner(dfs_nl, DIST_TO_FLY_INNER_T), "mil_", pair_info)
                pair_info["mc_1:dh_nh/dt_nt"] = pair_info["mnh_2:rel_polar_h:r"] / pair_info["mnt_2:rel_polar_h:r"]

                dfs_to_h = dfs_towards_head(dfs_c, True)
                dfs_to_t = dfs_towards_head(dfs_c, False)
                pair_info["mc_1:d_to_h/d_to_t"] = dfs_to_h[2]["rel_polar_h:r"].median() / dfs_to_t[2]["rel_polar_h:r"].median()

                s = (len(dfs_to_h[0]) + len(dfs_to_t[0]))
                pair_info["mc_1:p_to_h"] = (len(dfs_to_h[0]) / s) if s > 20 else np.nan
                pair_info["mc_1:p_cw"] = np.count_nonzero(dfs_c[1]["theta"] > 0) / len(dfs_c[1])
        save_dict(dict_name, pair_info)

def update_ana_data_post(geno=None, data_type="center"):  # use cir_center_stat0.pickle update center.txt
    geno_l = [geno] if geno else os.listdir(GENO_DATA_DIR)
    for geno in tqdm(geno_l):
        geno_folder = os.path.join(GENO_DATA_DIR, geno)
        if os.path.isdir(geno_folder):
            for pair_name in os.listdir(geno_folder):
                geno_pair_folder = os.path.join(geno_folder, pair_name)
                if not os.path.isdir(geno_pair_folder):
                    continue
                print((geno, geno_pair_folder, pair_name))
                prefix = os.path.join(geno_pair_folder, pair_name)

                ret = load_dict(prefix + "_%s.txt" % data_type)
                if data_type == "center":
                    dfs, bouts, n = load_dfs_bouts(geno_pair_folder, "_cir_center_stat0.pickle", "_cir_meta_center.txt")
                else:
                    # dfs, bouts, n = load_dfs_bouts(geno_pair_folder, "_cir_stat0.pickle", "_cir_meta.txt")
                    dfs = load_dfs(geno_pair_folder, only=1)  # NOTE: w_
                    bouts, n = None, None
                # prefix_d = os.path.join(geno_pair_folder, pair_name)
                # meta = load_dict(prefix_d + "_cir_meta.txt")
                # TODO: input update code here
                calc_cir_info_mean_extra(dfs, bouts, ret)

                save_dict(prefix + "_%s.txt" % data_type, ret)

def update_geno_info(geno=None):
    # log_path = os.path.join(GENO_DATA_DIR, "log.csv")
    # log = pd.read_csv(log_path, index_col=0).to_dict("index")
    geno_l = [geno] if geno else os.listdir(GENO_DATA_DIR)
    for geno in tqdm(geno_l):
        geno_folder = os.path.join(GENO_DATA_DIR, geno)
        if os.path.isdir(geno_folder):
            for data_type in ["all", "center"]:
                geno_txt = os.path.join(geno_folder, "_%s_%s.txt" % (geno, data_type))
                cir_csv = os.path.join(geno_folder, "_%s_cir_%s.csv" % (geno, data_type))
                os.path.exists(geno_txt) and os.remove(geno_txt)
                os.path.exists(cir_csv) and os.remove(cir_csv)

            for pair_name in os.listdir(geno_folder):
                geno_pair_folder = os.path.join(geno_folder, pair_name)
                if os.path.isdir(geno_pair_folder) and pair_name.startswith("2"):
                    print("update_geno_info", pair_name)
                    # if CAN_SKIP and log.get(pair_name, {}).get("stage") != 2: # NOTE: cant skip
                    #     print("skip", pair_name)
                    #     continue
                    prefix_d = os.path.join(geno_pair_folder, pair_name)

                    pair_info = load_dict(prefix_d + "_all.txt")
                    if pair_info["duration"] >= MIN_DURATION:
                        cir_info = load_dataframe(prefix_d + "_cir_info.csv")
                        update_geno_folder(geno_folder, pair_name, pair_info, cir_info, "all")

                        pair_info = load_dict(prefix_d + "_center.txt")
                        pair_info["roi"] = load_dict(prefix_d + "_cir_meta_center.txt").get("ROI", {}).get("roi")
                        cir_info = load_dataframe(prefix_d + "_cir_info_center.csv")
                        update_geno_folder(geno_folder, pair_name, pair_info, cir_info, "center")
                        # modify_update_log(log, pair_name, geno, 3)
                    else:
                        print("remove!!!", pair_name)
                        dst = geno_pair_folder.replace("geno_data", "remove_geno")
                        dst_p = os.path.dirname(dst)
                        dst_pp = os.path.dirname(dst_p)
                        mkdir(dst_pp)
                        mkdir(dst_p)
                        shutil.move(geno_pair_folder, dst)
        clean_save_cache()

def dfs_columns(dfs, keys_l):
    return dfs[0][keys_l[0]], dfs[1][keys_l[1]], dfs[2][keys_l[2]]

def modify_update_log(log, pair_name, geno, stage):
    log[pair_name] = {"update": time_now_str(), "geno": geno, "stage": stage}
    pd.DataFrame.from_dict(log, orient="index").to_csv(os.path.join(GENO_DATA_DIR, "log.csv"))

def cir_with_condition(cir_bouts, cir_info, data_type):
    if not len(cir_info):
        return cir_bouts, cir_info
    if data_type == "center":
        # NOTE: both female and male is not on edge
        cir_info2 = cir_info.query("dist_c < %d and dist_c2 < %d" % (DIST_TO_CENTER_THRESHOLD_MALE, DIST_TO_CENTER_THRESHOLD_FEMALE))
    else:  # TODO: good
        cir_info2 = cir_info
    cir_bouts2 = np.array(cir_bouts)[list(cir_info2.index)].tolist()
    return cir_bouts2, cir_info2

def update_geno_folder(geno_folder, pair_name, pair_info, cir_info, data_type=""):
    if not pair_info:
        print("no circling!!!", pair_name)
        return

    geno = os.path.basename(geno_folder)
    geno_txt = os.path.join(geno_folder, "_%s_%s.txt" % (geno, data_type))
    geno_info = load_with_cache(geno_txt) or {}
    geno_info.setdefault("pairs", {})
    geno_info["pairs"][pair_name] = pair_info
    # df = pd.DataFrame(list(geno_info["pairs"].values()))
    # for k, v in df.mean().to_dict().items():
    #     geno_info["m_" + k] = v
    save_with_cache(geno_txt, geno_info)
    update_geno_folder_cir_info(geno_folder, pair_name, cir_info, data_type)

def update_geno_folder_cir_info(geno_folder, pair_name, cir_info, data_type=""):
    geno = os.path.basename(geno_folder)
    cir_csv = os.path.join(geno_folder, "_%s_cir_%s.csv" % (geno, data_type))
    cir_info_geno = load_with_cache(cir_csv)
    if cir_info_geno is None:
        cir_info_geno = cir_info
    else:
        if np.count_nonzero(cir_info_geno["pair"] == pair_name) > 0:
            cir_info_geno.drop(np.nonzero(cir_info_geno["pair"] == pair_name)[0], inplace=True)
        cir_info_geno = cir_info_geno.append(cir_info, ignore_index=True, sort=False)
    cir_info_geno.sort_values(by="cir_with_cond", inplace=True)
    # cir_info_all[cir_csv] = cir_info_geno
    save_with_cache(cir_csv, cir_info_geno)

save_cache = {}
def load_with_cache(filename):
    if filename in save_cache:
        return save_cache[filename]
    if filename.endswith(".txt"):
        return load_dict(filename)
    return load_dataframe(filename)

def save_with_cache(filename, info):
    save_cache[filename] = info

def clean_save_cache():
    global save_cache
    for filename, info in save_cache.items():
        if filename.endswith(".txt"):
            save_dict(filename, info)
        else:
            save_dataframe(info, filename)
    save_cache = {}

def calc_pair_info_meta(dfs, meta, cir_count=1, cir_with_cond=1, ret=None):
    if not ret:
        ret = {}
    frames = len(dfs[0])
    fps = meta["FPS"]
    ret["fps"] = fps
    ret["duration"] = frames / fps
    ret["female_days"] = meta["female_days"]
    ret["male_days"] = meta["ROI"]["male_days"]
    ret["cir_per_second"], ret["cir_ratio"], ret["cir_duration"] = get_cir_percent(meta["cir_bouts1"], ret["duration"], frames, fps)
    ret["w_we_ratio"] = np.count_nonzero(dfs[1]["court"] > 0) / frames
    ret["w_nh_ratio"] = dfs[1]["nh"].mean()

    ids_c = (dfs[2]["dist_c"] < DIST_TO_CENTER_THRESHOLD_MALE) & (dfs[1]["dist_c"] < DIST_TO_CENTER_THRESHOLD_FEMALE)
    ret["w_sw30_ratio"] = np.count_nonzero(dfs[1]["walk"] == STATE_SIDE_WALK & ids_c) / len(ids_c)
    ret["w_edge_ratio"] = dfs[1]["on_edge"].mean()
    ret["w_center_ratio"] = np.count_nonzero(ids_c) / frames

    walk_idx = dfs[1]["walk"] != STATE_STATIC
    ret["w_static_ratio"] = np.count_nonzero(walk_idx) / frames
    ret["w_vf_walk_m"] = dfs[1][walk_idx]["vf"].mean()
    ret["w_male_len"] = dfs[1]["e_maj"].mean()
    ret["w_female_len"] = dfs[2]["e_maj"].mean()
    ret["cir_bouts"] = meta["cir_bouts1"]
    ret["cop_bouts"] = meta.get("copulate", [])
    ret["cir_count"] = cir_count or 1
    ret["cir_with_cond"] = cir_with_cond or 1
    ret["cir_with_cond_ratio"] = ret["cir_with_cond"] / ret["cir_count"]
    ret["ici"] = calc_ici(ret["cir_bouts"]) / fps
    return ret

def calc_ici(bouts):
    ss = np.array([s for s, e in bouts])
    ee = np.array([e for s, e in bouts])
    return (ss[1:] - ee[:-1]).mean()

def calc_cir_info_mean(dfs, key_prefix, pair_info):
    # NOTE: mean, median, mean_angle
    for i in [0, 1, 2]:
        for k in DF_KEYS[i]:
            pw = key_prefix + "%d:" % i
            mean_type = DF_MEAN_TYPE.get(k)
            if mean_type == "mean":
                pair_info[pw + k] = dfs[i][k].mean()
            elif mean_type == "mean_angle":
                pair_info[pw + k] = mean_angle(dfs[i][k])
            elif mean_type == "mean_nzero":
                pair_info[pw + k] = dfs[i][k].sum()/np.count_nonzero(dfs[i][k])
            else:
                pair_info[pw + k] = dfs[i][k].median()
    pair_info[key_prefix + "1:nh_index"] = nan_div(dfs[1]["nh"].sum(), dfs[1]["nt"].sum() + dfs[1]["nh"].sum())
    if key_prefix == "mnh_":
        dfs_nh = dfs
        pair_info[key_prefix + "1:sd_rel_polar_hh:t"] = sd_angle(dfs_nh[1]["rel_polar_hh:t"]) if len(dfs_nh[1]) > 20 else np.nan
        pair_info[key_prefix + "1:sd_rel_polar_h:t"] = sd_angle(dfs_nh[1]["rel_polar_h:t"]) if len(dfs_nh[1]) > 20 else np.nan
        pair_info[key_prefix + "1:sd_rel_polar:t"] = sd_angle(dfs_nh[1]["rel_polar:t"]) if len(dfs_nh[1]) > 20 else np.nan
        pair_info[key_prefix + "1:sd_rel_pos:x"] = dfs_nh[1]["rel_pos:x"].std() if len(dfs_nh[1]) > 20 else np.nan
        pair_info[key_prefix + "1:sd_rel_pos:y"] = dfs_nh[1]["rel_pos:y"].std() if len(dfs_nh[1]) > 20 else np.nan
    return pair_info

def sd_angle(angles):
    md = mean_angle(angles)
    d = lim_dir_a(angles - md)
    return np.sqrt((d ** 2).sum() / (len(angles) - 1))

def calc_cir_info_mean_extra(dfs, bouts, ret):
    frames = len(dfs[1])
    if not frames:
        return
    # ids_c = (dfs[2]["dist_c"] < DIST_TO_CENTER_THRESHOLD_MALE) & (dfs[1]["dist_c"] < DIST_TO_CENTER_THRESHOLD_FEMALE)
    # ret["w_sw30_ratio"] = np.count_nonzero(dfs[1]["walk"] == STATE_SIDE_WALK & ids_c) / len(ids_c)
    # ret["w_edge_ratio"] = dfs[1]["on_edge"].mean()
    # ret["w_center_ratio"] = np.count_nonzero(ids_c) / frames
    # if not len(bouts):
    #     return
    # ret["ici"] = calc_ici(ret["cir_bouts"]) / ret["fps"]
    # ret["mc_1:p_cw"] = np.count_nonzero(dfs[1]["theta"] > 0) / len(dfs[1])
    # dfs_to_h = dfs_towards_head(dfs, True)
    # dfs_to_t = dfs_towards_head(dfs, False)
    # s = (len(dfs_to_h[0]) + len(dfs_to_t[0]))
    # ret["mc_1:p_to_h"] = (len(dfs_to_h[0]) / s) if s > 20 else np.nan
    # NOTE: nh_index
    # ret["mc_1:nh_index"] = nan_div(dfs[1]["nh"].sum(), dfs[1]["nt"].sum() + dfs[1]["nh"].sum())
    # NOTE: sd_rel_polar_hh:t
    key_prefix = "mnh_"
    dfs_nh = dfs_head_quadrant(dfs)
    # ret[key_prefix + "1:sd_rel_polar_hh:t"] = sd_angle(dfs_nh[1]["rel_polar_hh:t"]) if len(dfs_nh[1]) > 20 else np.nan
    # ret[key_prefix + "1:sd_rel_polar_h:t"] = sd_angle(dfs_nh[1]["rel_polar_h:t"]) if len(dfs_nh[1]) > 20 else np.nan
    # ret[key_prefix + "1:sd_rel_polar:t"] = sd_angle(dfs_nh[1]["rel_polar:t"]) if len(dfs_nh[1]) > 20 else np.nan
    ret[key_prefix + "1:sd_rel_pos_h:x"] = dfs_nh[1]["rel_pos_h:x"].std() if len(dfs_nh[1]) > 20 else np.nan
    ret[key_prefix + "1:sd_rel_pos_h:y"] = dfs_nh[1]["rel_pos_h:y"].std() if len(dfs_nh[1]) > 20 else np.nan
    # ret["mnh_1:rel_polar:r"] = mean_value(dfs_nh[1]["rel_polar:r"])
    # ret["mnh_1:rel_polar:t"] = mean_angle(dfs_nh[1]["rel_polar:t"])
    # ret["mnh_1:rel_polar_h:r"] = mean_value(dfs_nh[1]["rel_polar_h:r"])
    # ret["mnh_1:rel_polar_h:t"] = mean_angle(dfs_nh[1]["rel_polar_h:t"])

    # ret["mc_1:dh_nh/dt_nt"] = ret["mnh_2:rel_polar_h:r"] / ret["mnt_2:rel_polar_h:r"]
    # dfs_to_h = dfs_towards_head(dfs, True)
    # dfs_to_t = dfs_towards_head(dfs, False)
    # ret["mc_1:d_to_h/d_to_t"] = dfs_to_h[2]["rel_polar_h:r"].median() / dfs_to_t[2]["rel_polar_h:r"].median()

def nan_div(a, b):
    if b == 0:
        return None
    return a/b

def get_cir_percent(cir_bouts, duration, frames, fps):
    cir_per_second = len(cir_bouts) / duration
    cir_frames = [e - s for s, e in cir_bouts]
    cir_ratio = np.sum(cir_frames) / frames
    return cir_per_second, cir_ratio, np.mean(cir_frames) / fps

# df_cache = {}
# def load_dfs_folder(geno_folder):
#     df_cache[geno_folder] = pd.DataFrame()
#     def cb(f):
#         global df_cache
#         df_cache[geno_folder] = df_cache[geno_folder].append(load_dfs(f), ignore_index=True, sort=False)
#     traverse_folder(geno_folder, cb)
#
# def load_dfs_one(pair_folder):
#     df_cache[pair_folder] = load_dfs(pair_folder)

def get_cir_keys(cir_info):
    all_keys = list(cir_info.columns)
    all_keys.sort()
    all_keys.pop(0)
    for k in ["s", "e", "geno", "pair", "i_swap_r", "cir_with_cond"]:
        all_keys.remove(k)
    all_keys.insert(0, "cir_with_cond")
    return all_keys

def pearson_corrcoef(x, y):
    return np.corrcoef(x, y)[0][1]

def plot_enter_beh_by_data(ax, bouts_l, color):
    s_l = []
    n = len(bouts_l)
    for bouts in bouts_l:
        if bouts:
            s_l.append(bouts[0][0])
    s_l.sort()
    s_l.insert(0, 0)
    ys = list(range(len(s_l)))
    s_l.append(ETHOGRAM_TIME_RANGE)
    ys.append(ys[-1])
    fig_info = [s_l, np.array(ys, dtype=float)/n, None, (0, ETHOGRAM_TIME_RANGE), (0, 1), color, "-", None]
    plot_line_by_info(ax, fig_info)
    return fig_info

def plot_etho_for_geno(geno, data_type="all", save_path=None, save_pickle=True):
    geno_folder = geno#GENO_DIR.get(geno, geno)
    if isinstance(geno_folder, list):
        geno_folder_l = geno_folder
    else:
        geno_folder_l = [geno_folder]
    cir_bouts_t = []
    for geno_folder in geno_folder_l:
        geno = os.path.basename(geno_folder)
        geno_info_f = os.path.join(geno_folder, "_%s_%s.txt" % (geno, data_type))
        geno_info = load_dict(geno_info_f)
        for pair, info in geno_info["pairs"].items():
            if info["duration"] < DURATION_LIMIT - 60:
                print("dur<59min", pair)
                # if os.path.exists(os.path.join(geno_folder, pair)):
                #     shutil.move(os.path.join(geno_folder, pair), os.path.join("G:/data_hrnet/remove_geno", geno, pair))
                continue
            fps = get_real_fps(pair) or info["fps"]
            cir_bouts = info.get("cir_bouts", [])
            cop_bouts = info.get("cop_bouts", [])
            cir_b = [[cir[0]/fps, cir[1]/fps] for cir in cir_bouts]
            cop_b = [[cop[0]/fps, cop[1]/fps] for cop in cop_bouts]
            cop_len = np.sum([cop[1] - cop[0] for cop in cop_b])
            if cop_len > 300:  # NOTE: copulation > 5min
                print("cop>5min", pair)
                if cop_len > 3000:
                    print("cop>50min", pair)
                    # if os.path.exists(os.path.join(geno_folder, pair)):
                    #     shutil.move(os.path.join(geno_folder, pair), os.path.join("G:/data_hrnet/remove_geno", geno, pair))
                    continue
            cir_bouts_t.append([pair, cir_b, cop_b])
    # cir_bouts_t.sort(key=lambda x: len(x[1]))
    cir_bouts_t.sort(key=lambda x: x[0])
    bins, cpm_l = plot_ethogram_by_data([t[0] for t in cir_bouts_t], [t[1] for t in cir_bouts_t], [t[2] for t in cir_bouts_t])
    save_path and save_and_open(save_path)

    if data_type == "all":
        plt.figure()
        ax = plt.gca()
        fig_info = plot_enter_beh_by_data(ax, [t[1] for t in cir_bouts_t], "b")
        if save_pickle:
            pf = open(save_path.replace(FIGURE_FILE_EXTENSION, "_enter.pickle"), "wb")
            pickle.dump(to_list(["line", 1, fig_info]), pf)
            pf.close()
        plot_enter_beh_by_data(ax, [t[2] for t in cir_bouts_t], "r")
        save_path and save_and_open(save_path.replace(FIGURE_FILE_EXTENSION, "_enter" + FIGURE_FILE_EXTENSION))

    plt.figure()
    ax = plt.gca()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Circling per minute")
    cpm, err_l = lines_to_err_band(cpm_l)
    fig_info = [bins, cpm, err_l, (0, ETHOGRAM_TIME_RANGE), None, "k", "-", None]
    plot_line_by_info(ax, fig_info)
    save_path and save_and_open(save_path.replace(FIGURE_FILE_EXTENSION, "_dist" + FIGURE_FILE_EXTENSION))
    if save_pickle:
        pf = open(save_path.replace(FIGURE_FILE_EXTENSION, "_dist.pickle"), "wb")
        pickle.dump(to_list(["line", 1, fig_info]), pf)
        pf.close()

    # ici_ll = []
    # bin_step = 600
    # bins = ETHOGRAM_TIME_RANGE // bin_step
    # for pair, cir_b, _ in cir_bouts_t:
    #     ici_l = [[] for i in range(bins)]
    #     last_e = 0
    #     for s, e in cir_b:
    #         if last_e and s - last_e < 200:
    #             bin_n = int(s) // bin_step
    #             # if bin_n < bins:
    #             ici_l[bin_n].append(s - last_e)
    #         last_e = e
    #     ici_ll.append([np.mean(ii) for ii in ici_l])
    # xs = np.linspace(0, ETHOGRAM_TIME_RANGE + 1, bins)
    # # for ici_l in ici_ll:
    # #     plt.plot(xs, ici_l)
    # plt.plot(xs, np.nanmean(ici_ll, axis=0))
    # plt.show()

def plot_etho_for_video(video_folder):
    names, cir_b, cop_b = [], [], []
    video_name = os.path.basename(video_folder)
    for pair_no in os.listdir(video_folder):
        cir_meta = os.path.join(video_folder, pair_no, "%s_%s_cir_meta.txt" % (video_name, pair_no)) # "all"
        if os.path.exists(cir_meta):
            meta = load_dict(cir_meta)
            fps = meta["FPS"]
            names.append("%s_%s" % (video_name, pair_no))
            cir_b.append([[cir[0]/fps, cir[1]/fps] for cir in meta["cir_bouts1"]])
            cop_b.append([[cop[0]/fps, cop[1]/fps] for cop in meta["copulate"]])
    bins, cpm_l = plot_ethogram_by_data(names, cir_b, cop_b)
    save_and_open(os.path.join(video_folder, video_name + "_etho.png"))

def plot_for_geno(geno_folder, data_type=None):
    # etho
    geno = os.path.basename(geno_folder)
    data_type = data_type or os.path.basename(DATA_DIR)
    plot_etho_for_geno(geno_folder, data_type, os.path.join(DATA_DIR, "_etho/%s_etho_%s.png" % (geno, data_type)))
    # # cir
    cir_info_f = os.path.join(geno_folder, "_%s_cir_%s.csv" % (geno, data_type))
    cir_info = load_dataframe(cir_info_f)
    # NOTE: unnamed:0 = idx
    # _plot_swarm(cir_info, get_cir_keys(cir_info), "_cir/_" + geno + "_cir.png", "pair", xtick_off=True, swarm=False, width=8)

    # fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    # axes = axes.flatten()
    # for i, (x, y) in enumerate([("e_maj", "dist_McFc"), ("e_maj", "dist_MhFh"), ("e_maj", "dist_MhFt"), ("e_maj", "speed"),
    #                           ("speed", "dist_McFc"), ("accum_move", "angle_range"), ("start_angle", "end_angle"),]):
    #     sns.scatterplot(x, y, data=cir_info, ax=axes[i])
    #     axes[i].set_title("r=%.3f" % pearson_corrcoef(cir_info[x], cir_info[y]))
    # plt.suptitle(geno)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # save_and_open(os.path.join(DATA_DIR, "_cir/_" + geno + "_cor.png"), False)

def get_all_keys(df):
    all_keys = list(df.columns)
    for k in ["geno", "pair"]:
        all_keys.remove(k)
    return all_keys

def get_keys_by_type(pr):
    if pr.startswith("0"):
        return KEY_TYPE_KEYS[pr]
    return [pr + "_" + k for k in KEY_TYPE_KEYS[pr]]

def get_key_color(k):
    if k.find("e_maj") >= 0:
        return "y"
    if k.find("we") >= 0 or k.find("wing") >= 0:
        return "r"
    if k.find("dist") >= 0 or k.find(":x") >= 0 or k.find(":r") >= 0:
        return "m"
    if k.find(":t") >= 0:
        return "g"
    return "b"

def plot_for_driver_group2(geno_l, width=2, prefix="img/", group=3, update=True):
    global PLOT_DRIVER_GROUP
    PLOT_DRIVER_GROUP = group
    plot_for_driver_group(geno_l, width=width, prefix=prefix, update=update)
    plot_for_driver_group(geno_l, width=width, prefix=prefix, data_type="all", update=update)

def get_palette(geno_l):
    ret = []
    for g in geno_l:
        if g.startswith("CS") or g.endswith("CS31"):
            ret.append("gray")
            # ret.append("g")
        elif g.endswith("Trp31"):
            ret.append("r")
        elif g.endswith("Shi31"):
            ret.append("b")
        elif g.startswith("IR"):
            ret.append("r")
        elif g.startswith("A"):
            ret.append("b")
        elif g.startswith("Ort"):
            ret.append("g")
    return ret

def plot_for_driver_group(geno_l, geno_folders_l=None, data_type="center", prefix="img/swarm/", width=2, violin=False, update=True):
    csv_name = "%s_%s.csv" % (prefix, data_type)
    mkdir(os.path.dirname(prefix))
    if os.path.exists(csv_name) and not update:
        df = pd.read_csv(csv_name)
    else:
        if not geno_folders_l:
            geno_folders_l = [GENO_DIR[g] for g in geno_l]
        geno_d = []
        for geno, geno_folders in zip(geno_l, geno_folders_l):
            if not isinstance(geno_folders, list):
                geno_folders = [geno_folders]
            for geno_folder in geno_folders:
                geno_info_f = os.path.join(geno_folder, "_%s_%s.txt" % (os.path.basename(geno_folder), data_type))
                geno_info = load_dict(geno_info_f)
                if geno_info:
                    for pair, info in geno_info["pairs"].items():
                        info["geno"] = geno
                        info["pair"] = pair
                        del info["cir_bouts"]
                        info["female_days"] = int(info["female_days"])
                        geno_d.append(info)
        df = pd.DataFrame(geno_d)
        df.to_csv(csv_name, index=False)
    df["cir_per_minute"] = df["cir_per_second"] * 60
    df_filtered = df.query("cir_count > %d" % MIN_BOUTS)

    swarm_type = "violin" if violin else "swarm"
    f1 = prefix + "_%s_%s_cir" % (swarm_type, data_type)
    f2 = prefix + "_%s_%s_filtered" % (swarm_type, data_type)
    # if geno_l[0].endswith("CS31") and len(geno_l) > 5:
    #     # palette = [*"bbbbrgbrgbrgbrgbrg"]
    #     palette = [*"ggggrbgrbgrbgrbgrb"]
    palette = get_palette(geno_l)
    if data_type == "center":
        # keys = ["cir_with_cond_ratio"]
        # _plot_swarm(df, keys, f1, suptitle="_swarm_%s_cir" % data_type, width=width, violin=violin)
        # keys = ["mc_0:dist_McFc", "mnh_1:rel_polar:r", #"mnh_1:rel_polar_hh:r", "mnh_1:rel_polar_h:r",
        #         "mnh_1:rel_polar:t", #"mnh_1:rel_polar_hh:t", "mnh_1:rel_polar_h:t",
        #         "mnh_1:sd_rel_polar:t", "mnh_1:sd_rel_pos:x", "mnh_1:sd_rel_pos:y",#"mnh_1:sd_rel_polar_hh:t", "mnh_1:sd_rel_polar_h:t",
        #         "mc_1:we_ipsi", "me_1:we_ipsi", "mef_1:we_ipsi", #"mnh_1:we_ipsi", "mnt_1:we_ipsi",
        #         "mc_1:v_len", "mc_1:abs_av", "cir_per_second",
        #         "mc_1:nh_index", "mc_1:nh", #"mc_1:d_to_h/d_to_t", "mc_1:dh_nh/dt_nt",
        #         ]
        # keys = ["mc_0:dist_McFc", "mc_1:wing_m", "mc_1:we_ipsi", "mc_1:nh_index",
        #         "mnh_1:rel_polar:r",
        #         "me_1:wing_m",
        #         "me_1:we_ipsi", "mef_1:we_ipsi",
        #         "mc_1:nh", "mc_1:nt",
        #         "mnh_1:sd_rel_polar_h:t", "mnh_1:sd_rel_pos_h:x", "mnh_1:sd_rel_pos_h:y",
        #         ]  # test WIPS
        # _plot_swarm(df_filtered, keys, f2, suptitle="_swarm_%s_filtered(cir > %d)" % (data_type, MIN_BOUTS), width=width, col=2, violin=violin, palette=palette)

        _plot_swarm(df_filtered, ["mc_2:v_len"], prefix + "_fv", ylabel="Female velocity", width=width, col=1, palette=palette, swarm=True)
        _plot_swarm(df_filtered, ["mc_1:v_len"], prefix + "_mv", ylabel="Male velocity", width=width, col=1, palette=palette, swarm=True)
        _plot_swarm(df_filtered, ["mc_0:dist_McFc"], prefix + "_distance", ylabel="Male-female distance", width=width, col=1, palette=palette, swarm=True)
        _plot_swarm(df_filtered, ["mnh_1:sd_rel_polar:t"], prefix + "_sdt", ylabel="SD of $\\alpha^F$ (°)", width=width, col=1, palette=palette, swarm=True)
        _plot_swarm(df_filtered, ["mc_1:we_ipsi"], prefix + "_wing_ipsi", ylabel="Wing choice index", width=width, col=1, palette=palette, swarm=True)
        _plot_swarm(df_filtered, ["me_1:we_ipsi"], prefix + "_wing_ipsi_we", ylabel="Wing choice index (we)", width=width, col=1, palette=palette, swarm=True)
        _plot_swarm(df_filtered, ["mef_1:we_ipsi"], prefix + "_wing_ipsi_far", ylabel="Wing choice index (we,far)", width=width, col=1, palette=palette, swarm=True)

        ## _plot_swarm(df_filtered, ["mc_1:nh"], prefix + "_nh", ylabel="Percentage of time\nnear head", width=width, col=1, palette=palette, swarm=False)
        ## _plot_swarm(df_filtered, ["mc_1:nt"], prefix + "_nt", ylabel="Percentage of time\nnear tail", width=width, col=1, palette=palette, swarm=False)
        _plot_swarm(df, ["cir_duration"], prefix + "_dur", ylabel="Duration (s)", width=width, palette=palette, swarm=True)
    else:
        pass
    #     keys = ["cir_duration", "ici", "cir_per_minute", "mc_1:p_to_h", "w_static_ratio", "w_we_ratio", "w_sw30_ratio", "w_edge_ratio", "w_center_ratio"] #"cir_ratio",
        # _plot_swarm(df, keys, f1, suptitle="_swarm_%s_cir" % data_type, width=width, col=2, violin=violin, palette=palette)

        _plot_swarm(df, ["cir_per_minute"], prefix + "_cpm", ylabel="Bouts/min", width=width, palette=palette, swarm=True)
        _plot_swarm(df, ["w_we_ratio"], prefix + "_we_ratio", ylabel="Percentage of time\nin wing extension", width=width, palette=palette, swarm=False)


def plot_for_driver(geno_folders, use_pickle=False, geno_folders_out_d=None, driver=None):
    data_type = os.path.basename(DATA_DIR)
    if driver:  #len(geno_folders) > 24:
        prefix = os.path.join(DATA_DIR, "_driver/" + driver)
    else:
        driver = geno_folders[0][:-2].replace("XCS", "")
        prefix = os.path.join(DATA_DIR, "_driver/" + ",".join(geno_folders))
    if CAN_SKIP and os.path.exists(prefix + "_bar.png"):
        print("skip", prefix)
        return
    pickle_name = prefix + ".pickle"
    if use_pickle and os.path.exists(pickle_name):
        df = load_dataframe(pickle_name)
    else:
        geno_d = []
        for geno in geno_folders:
            geno_info_f = os.path.join(GENO_DATA_DIR, geno, "_%s_%s.txt" % (geno, data_type))
            geno_info = load_dict(geno_info_f)
            if geno_info:
                for pair, info in geno_info["pairs"].items():
                    info["geno"] = geno
                    info["pair"] = pair
                    if info.get("cir_bouts"):
                        del info["cir_bouts"]
                    info["female_days"] = int(info.get("female_days", 0))
                    geno_d.append(info)
        if geno_folders_out_d:  # NOTE: data in other GENO_DATA_DIR
            for geno_data_dir, geno_folders_out in geno_folders_out_d.items():
                data_name = os.path.basename(geno_data_dir) + "_"
                for geno_l in geno_folders_out:
                    for geno in geno_l.split():
                        geno_info_f = os.path.join(geno_data_dir, "geno_data", geno, "_%s_%s.txt" % (geno, data_type))
                        print(geno)
                        geno_info = load_dict(geno_info_f)
                        if geno_info:
                            geno2 = data_name + geno
                            for pair, info in geno_info["pairs"].items():
                                info["geno"] = geno2
                                info["pair"] = pair
                                if info.get("cir_bouts"):
                                    del info["cir_bouts"]
                                info["female_days"] = int(info["female_days"])
                                geno_d.append(info)
                            geno_folders.append(geno2)
            geno_folders.pop(0)
        if not geno_d:
            return
        df = pd.DataFrame(geno_d)
        save_dataframe(df, pickle_name)
    df_filtered = df.query("cir_count > 5")   # NOTE: only take active flies
    # fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    # sns.scatterplot("male_days", "cir_ratio", data=df, ax=axes[0])
    # sns.scatterplot("male_days", "we_ratio", data=df, ax=axes[1])
    # plt.tight_layout()
    # save_and_open(prefix + "_scatter0.png")

    info_l = [] #{"pairs": [], "cir_count": [], "cir_with_cond": [], "cir_with_cond_ratio": []}
    # cir_df = pd.DataFrame()
    for geno in geno_folders:
        geno_df = df.query("geno == \"%s\"" % geno)

        pairs = len(geno_df)
        if not pairs:
            pairs = 1
        cir_count = geno_df["cir_count"].sum() or 1
        cir_with_cond = geno_df["cir_with_cond"].sum()
        inactive_pairs = np.count_nonzero(geno_df["cir_count"] <= 5)
        # active_pairs = pairs - inactive_pairs
        info_l.append({
            "geno": geno,
            "pairs": pairs,
            "cir_count": cir_count,
            "cir_per_pair": cir_count / pairs,
            "cir_less_than_6": inactive_pairs,
            "cir_with_cond": cir_with_cond,
            "cir_with_cond_ratio": cir_with_cond / cir_count,
            "cir_less_than_6_ratio": inactive_pairs / pairs,
        })
        # NOTE: for cir swarm
        # cir_info_f = os.path.join(DATA_DIR, geno, "_%s_cir_%s.csv" % (geno, data_type))
        # cir_info = load_dataframe(cir_info_f)
        # cir_df = cir_df.append(cir_info, ignore_index=True, sort=False)
    info_df = pd.DataFrame(info_l)
    key_l = ["pairs", "cir_count", "cir_with_cond_ratio", "cir_per_pair", "cir_less_than_6_ratio"]
    fig, axes = plt.subplots(1, len(key_l), figsize=(len(key_l) * 5, 3))
    for ax, k in zip(axes, key_l):
        sns.barplot(x="geno", y=k, data=info_df, ax=ax, palette=SWARM_PALATTE[:PLOT_DRIVER_GROUP], alpha=0.5)
    sns.barplot(x="geno", y="cir_less_than_6", data=info_df, ax=axes[0], color="r")
    sns.barplot(x="geno", y="cir_with_cond", data=info_df, ax=axes[1], color="r")
    for ax in axes:
        ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], rotation=90)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plot_bar_shade3(ax)
    plt.suptitle(driver + "_bar")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_and_open(prefix + "_bar" + FIGURE_FILE_EXTENSION)

    width = len(geno_folders)
    # if width > 3:
    # width = 0.4 * width
    # sns.swarmplot(ax=ax, x="geno", y="cir_count", data=df, size=5)  # test plot single
    for pr in ["00", "mw", "me", "mef"]:
        keys = get_keys_by_type(pr)
        keys and _plot_swarm(df, keys, prefix + "_swarm_%s" % pr + FIGURE_FILE_EXTENSION, suptitle=driver + "_swarm_" + pr + " (%s)" % data_type, width=width)
    for pr in ["01", "02", "mc", "mnh", "mnt"]:#, "mnl", "mih", "mit", "mil"]:
        keys = get_keys_by_type(pr)
        keys and _plot_swarm(df_filtered, keys, prefix + "_swarm_%s" % pr + FIGURE_FILE_EXTENSION, suptitle=driver + "_swarm_" + pr + "(%s c>5)" % data_type, width=width)

    # cir_keys = get_cir_keys(cir_df)
    # _plot_swarm(cir_df, cir_keys, prefix + "_swarm4.png", swarm=False, suptitle=driver)

def short_geno(geno):
    return geno[:geno.find("X")]

def get_row_col(count):
    row = int(np.sqrt(count))
    col = int(count / row)
    if row * col < count:
        col += 1
    return row, col

def anova_df(df, key, key_category="geno"):
    anova_results = anova_lm(ols("%s~C(%s)" % (key, key_category), df).fit())
    print(anova_results)
    mc = MultiComparison(df[key], df[key_category])
    mc.tukeyhsd(alpha=0.05)
    mc.allpairtest(stats.ttest_rel)
    mc.allpairtest(stats.mannwhitneyu)

def process_geno_xlabels(xlabels):
    tl = [la.get_text().replace("31", "") for la in xlabels]
    dn, en = [], []
    for t in tl:
        d, e = t.split("X")
        if d not in dn:
            dn.append(d)
        if e not in en:
            en.append(e)
    en.remove("CS")
    dn.extend(en)
    ret = []
    for t in tl:
        s = ["-"] * len(dn)
        d, e = t.split("X")
        s[dn.index(d)] = "+"
        s[dn.index(e)] = "+"
        ret.append("\n" + "\n".join(s))
    return ret


def _plot_swarm(df, keys, path, key_x="geno", swarm=True, violin=False, width=2.6, suptitle="", col=None, palette=None, ylabel=None):
    if len(df) == 0 or len(keys) == 0:
        return
    count = len(keys)
    if col:
        row = int(count/col)
        if row * col < count:
            row += 1
    else:
        row, col = get_row_col(count)
    i = 0
    fig, axes = plt.subplots(row, col, dpi=300) #figsize=(col*width*SWARM_WIDTH_SCALE, row*width*SWARM_HEIGHT_SCALE), #,sharex=True),
    plt.subplots_adjust(left=0.3)
    if count > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    palette = palette or SWARM_PALATTE#[:PLOT_DRIVER_GROUP]
    for ax in axes:
        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        ax.tick_params(labelsize=FONT_SIZE)
        if i < count:
            k = keys[i]
            i += 1
            if is_not_numeric(df, k):
                ax.set_ylabel(k + "(error)")
                continue
            swarm and sns.swarmplot(ax=ax, x=key_x, y=k, data=df, size=2, zorder=100, palette=palette)#, alpha=0.3)
            if violin:
                sns.violinplot(ax=ax, x=key_x, y=k, data=df, alpha=0.5, inner="box", zorder=0, ci="sd", estimator=np.mean, linewidth=.2, saturation=.5, palette=palette)
            else:
                if USE_SEM:
                    sns.barplot(ax=ax, x=key_x, y=k, data=df, alpha=0.5, capsize=.5, zorder=200, ci=None, estimator=np.mean, palette=palette) #, edgecolor=".2"
                    all_names = df[key_x].unique()
                    group = df.groupby(key_x)[k]
                    md = group.mean()
                    print("mean:", md)
                    sd = group.std()
                    md_f = [md[n] for n in all_names]
                    se_f = [(sd[n] / np.sqrt(len(group.get_group(n)))) for n in all_names]
                    ax.errorbar(list(range(len(all_names))), md_f, se_f, fmt="none", color="k", capsize=8, zorder=300)
                else:
                    sns.barplot(ax=ax, x=key_x, y=k, data=df, alpha=0.5, capsize=.5, zorder=200, ci="sd", estimator=np.mean, edgecolor=".2", palette=palette) #color=get_key_color(k),

            limy = STAT_LIMIT_SWARM.get(k)
            if PLOT_DRIVER_SIG:
                test_l = pair_hypo_test(key_x, k, df, PLOT_DRIVER_GROUP)
                # NOTE: p_m[0]: ids, names, p_value_matrix, kw_p_value
                plot_significant2(ax, test_l, ylim=limy)
                # plot_significant(ax, p_m[1], PLOT_DRIVER_GROUP)
                if PLOT_DRIVER_PVAL:
                    plot_pval_matrix(test_l, path, k)
            if USE_ADD_SUB_GENO_XLABEL:
                ax.set_xticklabels(process_geno_xlabels(ax.get_xticklabels()))
            else:
                xlabels = ax.get_xticklabels()
                if len(xlabels) == 3:
                    if xlabels[0]._text == "CS":
                        ax.set_xticklabels(["WT", "IR", "AME"])
                    else:
                        ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            if k == "cir_per_minute":
                ax.set_yticks([0, 1, 2, 3])
                ax.set_yticklabels([0, 1, 2, 3])
            if path.endswith("_dist") or k.endswith("rel_polar:r") or k.endswith("dist_McFc"):
                if path.endswith("_dist"):
                    mlen = BODY_LEN_CENTER_CIR[0]
                else:
                    mlen = (df["mc_1:e_maj"].mean() + df["mc_2:e_maj"].mean()) / 2
                    print(df["mc_1:e_maj"].mean(), df["mc_2:e_maj"].mean(), mlen)
                items = len(ax.get_xticklabels())
                ax.plot([-0.5, items - 0.5], [mlen, mlen], "k--")
            limy and ax.set_ylim(limy[0], limy[1])
            if not USE_ADD_SUB_GENO_XLABEL:
                plot_bar_shade3(ax)
        else:
            ax.axis("off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("")
        ylabel and ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
        plt.savefig(path+str(i)+'.png')
    suptitle and plt.suptitle(suptitle)
    # ax.set_ylabel("")
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # save_and_open(path)

def plot_bar_shade3(ax):
    xticks = ax.get_xticks()
    inter = PLOT_DRIVER_GROUP * 2
    if len(xticks) >= inter:
        yticks = ax.get_yticks()
        max_y = yticks[-1]
        for x_lim in xticks[PLOT_DRIVER_GROUP::inter]:
            ax.add_patch(plt.Rectangle((x_lim - 0.5, 0), PLOT_DRIVER_GROUP, max_y, alpha=0.06, color="k", linewidth=0))

def plot_pval_matrix(test_l, path, key):
    fig_h, cols, n = 4, len(test_l), 0
    fig, axes = plt.subplots(1, cols, num="p_value_matrix", figsize=(fig_h * cols, fig_h))
    for ids, names, p_value_matrix, kw_p_value in test_l:
        if p_value_matrix is not None:
            m = -np.log10(p_value_matrix)
            ax = axes[n]
            vmin, vmax = 1, 4
            im = ax.matshow(m, cmap="Reds", vmin=vmin, vmax=vmax)
            for i, row in enumerate(p_value_matrix):
                for j, col in enumerate(row):
                    ax.text(i - 0.5, j, "%.4f" % col, fontsize=12, color="gray")
            ticks = list(range(len(m)))
            ax.set_xticks(ticks)
            ax.set_xticklabels(names)
            ax.set_yticks(ticks)
            ax.set_yticklabels(names, rotation=90)
            # plot_colorbar(ax, im=im)
        n += 1
    # plt.tight_layout()
    parent = os.path.dirname(path)
    save_and_open(os.path.join(parent, "pv", key.replace(":", ",") + ".png"))

def plot_significant2(ax, test_l, need_text=False, ylim=None):
    ylim = ylim or ax.get_ylim()
    gap = (ylim[-1] - ylim[0]) * 0.03
    cur_y = ylim[-1] - gap*1.5
    init_y = cur_y
    sig_i = 0
    for ids, names, p_value_matrix, kw_p_value in test_l:
        need_text and ax.text(ids[int(PLOT_DRIVER_GROUP / 2) - 1], ylim[-1], get_pv_str(kw_p_value), fontsize=FONT_SIZE*0.8,
                    color="r" if kw_p_value < 0.05 else "gray")
        if p_value_matrix is not None:  # kw_p_value < 0.05:
            l = len(p_value_matrix)
            for i in range(0, 1):#l):
                for j in range(i, l)[::-1]:
                    v = p_value_matrix[i][j]
                    if v < 0:
                        continue
                    if PLOT_ALL_PVAL or (0 < v < 0.05):
                        cur_x1, cur_x2 = ids[i], ids[j]
                        # ax.plot([cur_x1, cur_x1, cur_x2, cur_x2], [cur_y, cur_y + gap, cur_y + gap, cur_y], linewidth=1, color="k")
                        ax.add_line(plt.Line2D([cur_x1, cur_x1, cur_x2, cur_x2], [cur_y, cur_y + gap, cur_y + gap, cur_y], linewidth=1, color="k"))
                        vc = "gray" if v > 0.05 else "k"
                        ax.text(cur_x1, cur_y + 2*gap, str(v), fontsize=FONT_SIZE*0.6, color=vc)#get_pv_str(v)
                        sig_i += 1
                        if sig_i > PLOT_DRIVER_GROUP-2:
                            sig_i = 0
                            cur_y = init_y
                        else:
                            cur_y -= gap*4

def get_pv_str(pv):
    if pv < 0.001:
        return "$P < 10^{%d}$" % np.log10(pv)
    elif pv < 0.05:
        return "$P < 0.05$"
    return "%.3f" % pv

def plot_significant(ax, p_matrix, group_count):
    c = len(p_matrix)
    if c < group_count:
        return
    yticks = ax.get_yticks()
    cur_y = yticks[-1]
    gap = (cur_y - yticks[0]) * 0.02
    for i in range(0, c, group_count):
        row = p_matrix[i]
        for j in range(i, min(i+group_count, c)):
            v = row[j]
            if v < 0.05:
                cur_y -= gap
                ax.plot([i, j], [cur_y, cur_y], "|-", linewidth=0.5, markersize=2, alpha=0.5)
                ax.text(j, cur_y, "%.3f" % v, fontsize=8, alpha=0.5)
                
def pair_hypo_test(key_group, key_value, df, group_count):
    df_key_group = df[key_group]
    all_names = df_key_group.unique()
    c = len(all_names)
    ret = []
    for i in range(0, c, group_count):  # test 1
        names, groups = [], []
        ids = range(i, min(c, i + group_count))
        for j in ids:
            name = all_names[j]
            names.append(name)
            idx = df_key_group == name
            groups.append(df[key_value][idx].tolist())
        pvm, p = pairwise_kruskalwallis(groups)
        ret.append([ids, names, pvm, p])
    return ret

def pairwise_kruskalwallis(list_groups):
    pvm = None
    try:
        h, p = kruskalwallis(*list_groups)
        if PLOT_ALL_PVAL or p < 0.05:
            # pvm = sp.posthoc_dunn(list_groups, p_adjust="bonferroni").values  # as_matrix()
            pvm = sp.posthoc_mannwhitney(list_groups, p_adjust="bonferroni").values  # as_matrix()
    except:
        print("pairwise_kruskalwallis except")
        h, p = 1, 1
    return pvm, p

def pairwise_manwitneyu(groups, df_key_group, key_value, df):
    ret = []
    for g1 in groups:
        row = []
        for g2 in groups:
            d1 = df[df_key_group == g1][key_value]
            d2 = df[df_key_group == g2][key_value]
            try:
                s, p = mannwhitneyu(d1, d2, alternative="two-sided")
            except:
                print("pairwise_manwitneyu except")
                p = np.nan
            row.append(p)
        ret.append(row)
    return [groups.tolist(), ret]

def plot_for_pair(pair_folder):
    name = os.path.basename(pair_folder)
    f = find_file(pair_folder, "_stat0.pickle")
    if f:
        dfs, bouts, n = load_dfs_bouts(pair_folder)
        # plot_by_name(dfs, "_pair/" + name + "-heat-male-rel_pos,x-rel_pos,y")  # test plot single
        plot_summary_by_name(SUMMARY_FIGURES, dfs, os.path.join(DATA_DIR, "_pair/" + name + "-summary"), bouts, n=n, save_svg=False, save_pickle=True, )

def plot_for_pair_geno(geno_folder, data_type):
    name = os.path.basename(geno_folder)
    summary_prefix = os.path.join(DATA_DIR, "_pair/" + name)
    if CAN_SKIP and os.path.exists(summary_prefix + "-summary.png"):
        print("skip", name)
        return
    if PLOT_CIR:
        if data_type == "center":
            dfs, bouts, n = load_folder_dfs_bouts(geno_folder, "_cir_center_stat0.pickle")
        else:
            dfs, bouts, n = load_folder_dfs_bouts(geno_folder)
        plot_summary_by_name(SUMMARY_FIGURES, dfs, summary_prefix + "-summary", bouts, n=n, save_svg=False, save_pickle=True, info_dir=INFO_DIR, need_title=True)
    if data_type == "center" and PLOT_WE:
        summary_prefix = os.path.join(DATA_DIR, "_pair/we/" + name)
        dfs, bouts, n = load_folder_dfs_bouts(geno_folder, "_we_stat0.pickle")
        dfs = dfs_no_touch(dfs)
        plot_summary_by_name(SUMMARY_FIGURES, dfs, summary_prefix + "-wesummary", bouts, n=n, save_svg=False, save_pickle=True, info_dir=INFO_DIR)

def plot_all_pair(all_geno, data_type):
    out_parent = os.path.join(DATA_DIR, "_pair/compare/")
    # out_parent3 = os.path.join(DATA_DIR, "_pair/compare3/")
    out_diff = os.path.join(DATA_DIR, "_pair/compare/diff/")
    for f in SUMMARY_FIGURES:
        if PLOT_CIR:
            if PLOT_DIFF:
                if f.startswith("c_") or f.startswith("fc_"):
                    plot_multi_diff_heat(f, all_geno, INFO_DIR, out_diff)
                elif f.startswith("a_") or f.startswith("d_"):
                    plot_multi_diff_hist(f, all_geno, INFO_DIR, out_diff)
            try:
                plot_multi_info([["%s-summary_%s.pickle" % (geno, f.replace(":", ",")) for geno in gg] for gg in all_geno], INFO_DIR, out_parent + f.replace(":", ",") + FIGURE_FILE_EXTENSION)
            except:
                print("error: trace on", f)
        if data_type == "center" and PLOT_WE:
            plot_multi_info([["%s-wesummary_%s.pickle" % (geno, f.replace(":", ",")) for geno in gg] for gg in all_geno], INFO_DIR, out_parent + "we-" + f.replace(":", ","))

        # for i in range(0, len(all_geno), 6):
        #     geno3 = all_geno[i], all_geno[i + 1], all_geno[i + 2]
        #     d = f + "_" + geno3[2][:geno3[2].find("X")]
        #     plot_multi_info(["%s-summary_%s.pickle" % (geno, f) for geno in geno3], parent, out_parent3 + d, col=1)
        #     if data_type == "center":
        #         plot_multi_info(["%s-wesummary_%s.pickle" % (geno, f) for geno in geno3], parent, out_parent3 + "we-" + d, col=1)

def plot_all_pair_out(all_geno_d, data_type, figure_name, out_file=None):
    multi_info = []
    for data_dir, all_geno in all_geno_d.items():
        all_geno = np.concatenate([a.split() for a in all_geno])
        for geno in all_geno:
            pk = "%s/%s/_info/%s-summary_%s.pickle" % (data_dir, data_type, geno, figure_name)
            multi_info.append(pk)
    multi_info = np.array(multi_info).reshape((-1, 3)).T
    plot_multi_info(multi_info, "", os.path.join(DATA_DIR, "multi-%s.png" % figure_name), col=len(multi_info.T))
    # multi_info1 = np.reshape(multi_info, (-1, 3)).T.flatten()
    # plot_multi_info(multi_info1, "", out_file or os.path.join(DATA_DIR, "multi-%s.png" % figure_name), col=int(len(multi_info1)/3))

def plot_multi_diff_heat(f, all_geno, parent, out_parent):
    is_weight_heat = f.startswith("fc_")
    rows, cols = len(all_geno[0]), 4 if is_weight_heat else 2
    FIGURE_W = 12
    fz = (cols * FIGURE_W, rows * FIGURE_W)
    fig, axes = plt.subplots(rows, cols, figsize=fz)
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0, wspace=0)
    for i in range(0, rows * 3, 3):
        geno_l = all_geno[0][i], all_geno[1][i], all_geno[2][i]
        geno_f = ["%s-summary_%s.pickle" % (geno, f) for geno in geno_l]
        info_l = [load_fig_info(os.path.join(parent, fi)) for fi in geno_f]
        if not info_l[0]:
            continue
        row = int(i / 3)
        for col in [0, 1]:
            ax = axes[row, col]
            if is_weight_heat:
                plot_heat_diff_p_value(ax, info_l[col + 1][2], info_l[0][2])
                ax.set_title("%s-%s p_val" % (geno_l[col + 1], geno_l[0]), fontsize=23)
                ax2 = axes[row, col + 2]
                plot_heat_diff(ax2, info_l[col + 1][2][:-1], info_l[0][2][:-1])
                ax2.set_title("%s-%s" % (geno_l[col + 1], geno_l[0]), fontsize=23)
            else:
                plot_heat_diff(ax, info_l[col + 1][2], info_l[0][2], fix_lim_heat=0.004)
                ax.set_title("%s-%s" % (geno_l[col + 1], geno_l[0]), fontsize=23)
    plt.suptitle(f)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_and_open(out_parent + f + "_diff")

def plot_hist_diff(ax, f, fig_info, fig_info2):
    ax.tick_params(labelsize=23)
    if f.startswith("a_"):
        xs, ys, bins, is_polar, color, lim_heat = fig_info
        xs2, ys2 = fig_info2[:2]
        range_x = [-180, 180]
        if lim_heat:
            diff_heat = (lim_heat[1] - lim_heat[0]) / 5
            lim_heat = [-diff_heat, diff_heat]

        vs_l1, bin_l1 = bin_data(ys, xs, bins=bins, range_a=range_x)
        vs_l2, bin_l2 = bin_data(ys2, xs2, bins=bins, range_a=range_x)
        hist_f1 = np.array([np.median(vs) for vs in vs_l1])
        hist_f2 = np.array([np.median(vs) for vs in vs_l2])
        pval = [mannwhitneyu(vs1, vs2, alternative="two-sided")[1] for vs1, vs2 in zip(vs_l1, vs_l2)]
        pval_c = pval_to_color(pval)

        hist_f = hist_f1 - hist_f2
        ax.scatter(bin_l1, hist_f, c=pval_c, cmap="Reds", marker="o", s=100)
        plot_adist(bin_l1, hist_f, "-", color, lim_heat, ax=ax)
    elif f.startswith("d_"):
        xs, range_x, bins, is_polar, color, x = fig_info
        xs2 = fig_info2[0]
        ys = np.ones((len(xs), ))
        ys2 = ys

        vs_l1, bin_l1 = bin_data(ys, xs, bins=bins, range_a=range_x)
        vs_l2, bin_l2 = bin_data(ys2, xs2, bins=bins, range_a=range_x)
        hist_f1 = np.array([len(vs) for vs in vs_l1])
        hist_f2 = np.array([len(vs) for vs in vs_l2])

        hist_f = hist_f1 - hist_f2
        ax.plot(bin_l1, hist_f, "o-", c=color)

def plot_multi_diff_hist(f, all_geno, parent, out_parent):
    rows, cols = int(len(all_geno) / 3), 2
    FIGURE_W = 12
    fz = (cols * FIGURE_W, rows * FIGURE_W)
    fig, axes = plt.subplots(rows, cols, figsize=fz)
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0, wspace=0)
    for i in range(0, rows * 3, 3):
        geno_l = all_geno[i], all_geno[i + 1], all_geno[i + 2]
        geno_f = ["%s-summary_%s.pickle" % (geno, f) for geno in geno_l]
        info_l = [load_fig_info(os.path.join(parent, fi)) for fi in geno_f]
        if not info_l[0]:
            continue
        is_list = isinstance(info_l[0][0], list)
        row = int(i / 3)
        for col in [0, 1]:
            ax = axes[row, col]
            if is_list:  # diff_a_d_ud
                for ii in range(len(info_l[0])):
                    plot_hist_diff(ax, f, info_l[col + 1][ii][2], info_l[0][ii][2])
            else:
                plot_hist_diff(ax, f, info_l[col + 1][2], info_l[0][2])
            ax.set_title("%s-%s" % (geno_l[col + 1], geno_l[0]), fontsize=23)
    plt.suptitle(f)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_and_open(out_parent + f + "_diff")

def plot_geno(geno, f, postfix="_cir_center_stat0.pickle"):
    # NOTE: pg CS31 fc_speed-male
    #    LC20XCS31 fc_wing_m-male
    #    EPGXTrp31/20191006_134922_A_5 c_pos-male
    #    IR22 stream_v_dir-female
    # NOTE: postfix: "_cir_center_stat0.pickle", "_cir_stat0.pickle", "_we_stat0.pickle", "_stat0.pickle"
    if geno.find("/20") >= 0:
        dfs, bouts, n = load_dfs_bouts(os.path.join(GENO_DATA_DIR, geno), postfix)
    else:
        dfs, bouts, n = load_folder_dfs_bouts(os.path.join(GENO_DATA_DIR, geno), postfix)
        if postfix == "_we_stat0.pickle":
            dfs = dfs_no_touch(dfs)
    plot_by_name(dfs, os.path.join(DATA_DIR, "_img/%s-%s" % (geno.replace("/", ","), f)), bouts=bouts)

def plot_geno_diff(geno_l, f):
    rows, cols = int(len(geno_l)), 3 if f.startswith("fc_") else 2
    FIGURE_W = 12
    fz = (cols * FIGURE_W, rows * FIGURE_W)
    fig, axes = plt.subplots(rows, cols, figsize=fz)
    geno_info0 = None
    geno0 = geno_l[0]
    for row, geno in enumerate(geno_l):
        ax = axes[row][0]
        fig_info_path = os.path.join(DATA_DIR, "_img/%s-%s.pickle" % (geno.replace("/", ","), f))
        if not os.path.exists(fig_info_path):
            plot_geno(geno, f)
        geno_info = load_fig_info(fig_info_path)
        plot_fig_info(geno_info, ax, no_color_bar=False)
        ax.set_title(geno, fontsize=23)

        if row == 0:
            geno_info0 = geno_info
            for c in range(1, cols):
                axes[row][c].axis("off")
        else:
            if f.startswith("c_") or f.startswith("fc_"):
                plot_heat_diff(axes[row][1], geno_info[2][:-1], geno_info0[2][:-1])
                if f.startswith("fc_"):
                    plot_heat_diff_p_value(axes[row][2], geno_info[2], geno_info0[2])
                    axes[row][2].set_title("%s-%s pval" % (geno, geno_l[0]), fontsize=23)
            elif f.startswith("a_") or f.startswith("d_"):
                plot_hist_diff(axes[row][1], f, geno_info0, geno_info)
            axes[row][1].set_title("%s-%s" % (geno, geno0), fontsize=23)
    plt.suptitle(f)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_and_open(os.path.join(DATA_DIR, "_img/%s-%s" % (geno0[:geno0.find("X")], f)))

def plot_hist_in_folder(folder, figsize=(5, 4)):
    import glob, pickle
    plt.figure(figsize=figsize)
    ax = plt.gca()
    xs_l = []
    xs_r = []
    for p in glob.glob(folder + "/*.pickle"):
        (name1, bouts1, fig_info1), (name2, bouts2, fig_info2) = pickle.load(open(p, "rb"))
        xs1, range_x, bins, is_polar, color1, x, line = fig_info1
        xs2, range_x, bins, is_polar, color2, x, line = fig_info2
        xs_l.extend(xs1)
        xs_r.extend(xs2)
    plot_hist_by_info(ax, [xs_l, range_x, bins, is_polar, color1, x, line])
    plot_hist_by_info(ax, [xs_r, range_x, bins, is_polar, color2, x, line])
    save_and_open(os.path.join(folder, "all"))

def load_dfs_bouts(pair_folder, postfix="_cir_center_stat0.pickle", postfix_bouts="_cir_meta_center.txt"):
    pair = os.path.basename(pair_folder)
    prefix = os.path.join(pair_folder, pair)

    dfs = load_dfs(prefix + postfix)
    bouts = load_bouts(prefix + postfix_bouts)
    return dfs, bouts, 1

POSTFIX_BOUTS = {
    "_stat0.pickle": "_cir_meta.txt",  #_meta.txt
    "_cir_stat0.pickle": "_cir_meta.txt",
    "_we_stat0.pickle": "_cir_meta.txt",
    "_cir_center_stat0.pickle": "_cir_meta_center.txt",
    "_cir_center_pre_stat0.pickle": "_cir_meta_center_pre.txt",
}
def load_folder_dfs_bouts(folder, postfix="_cir_center_stat0.pickle", we_min_pickle=False):
    dfs = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    bouts = []
    count = [0]
    def cb(path):
        pair = os.path.basename(path)
        prefix = os.path.join(path, pair)
        if os.path.exists(prefix + postfix):
            if we_min_pickle:
                dfs_p = load_dfs(prefix + postfix, only=None)
                dfs_p = dfs_we_min_pickle(dfs_p)
            else:
                dfs_p = load_dfs(prefix + postfix)
            dfs_p[0]["pair"] = pair
            bouts_p = squeeze_cir_bouts(load_bouts(prefix + POSTFIX_BOUTS[postfix]))
            merge_dfs(dfs, dfs_p)
            merge_bouts(bouts, bouts_p)
            count[0] += 1
    if isinstance(folder, list):
        for f in folder:
            traverse_folder(f, cb)
    else:
        traverse_folder(folder, cb)
    dfs[0].reset_index(inplace=True, drop=True)
    dfs[1].reset_index(inplace=True, drop=True)
    dfs[2].reset_index(inplace=True, drop=True)
    return dfs, bouts, count[0]

def merge_dfs(dfs, dfs_p):
    dfs[0] = dfs[0].append(dfs_p[0], ignore_index=True, sort=False)
    dfs[1] = dfs[1].append(dfs_p[1], ignore_index=True, sort=False)
    dfs[2] = dfs[2].append(dfs_p[2], ignore_index=True, sort=False)

def merge_bouts(bouts, bouts_p):
    if not bouts:
        last = 0
    else:
        last = bouts[-1][1]
    for b in bouts_p:
        bouts.append([b[0] + last, b[1] + last])

def is_not_numeric(df, k):
    try:
        np.asarray(df[k], dtype=np.float)
    except Exception:
        return True
    return False

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def get_all_geno31(driver_l, postfix="31"):
    ret = []
    for geno_l in driver_l:
        for g in geno_l.split():
            if g.endswith(postfix):
                ret.append(g)
    return ret

def ana_main(cmd=None, data_i=None):
    global PLOT_DRIVER_GROUP, VIDEO_ANA_DIR, DATA_DIR, GENO_DATA_DIR, INFO_DIR, SWARM_PALATTE
    if data_i:
        video_i = data_i#[vd for vd in ALL_VIDEO_DIR if vd.startswith("video" + data_i + "_")][0]
        VIDEO_ANA_DIR = "G:/" + video_i
        DATA_DIR = "D:/exp/data%s/center" % data_i
        GENO_DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "geno_data")
        INFO_DIR = os.path.join(DATA_DIR, "_info")

    mkdir(DATA_DIR)
    mkdir(os.path.join(DATA_DIR, "_etho"))
    # mkdir(os.path.join(DATA_DIR, "_cir"))
    mkdir(os.path.join(DATA_DIR, "_pair"))
    mkdir(os.path.join(DATA_DIR, "_pair/compare"))
    # mkdir(os.path.join(DATA_DIR, "_pair/we"))
    mkdir(os.path.join(DATA_DIR, "_info"))
    mkdir(os.path.join(DATA_DIR, "_driver"))

    data_n = DATA_DIR.split("/")[-2]
    if data_n == "data0":
        driver_l = ["CS22"]
    elif data_n == "data1":
        driver_l = ["CS22 IR22"]
    elif data_n == "data2":
        driver_l = [
            "CS22 CS25 CS31",
            # "CS31 CSXTrp31 CSXShi31", "CS22 CSXTrp22 CSXShi22",
            # "LC10XCS31 LC10XTrp31 LC10XShi31", "LC10XCS22 LC10XTrp22 LC10XShi22",
            # # "CS25",
            # "TBXTrp31 TBXShi31 TBXTNTE231", "TBXShi22 TBXTNTE222",
            # "LC10adXTrp31 LC10adXShi31 LC10adXKir31", "LC10adXTrp22 LC10adXShi22",
            # "LC10bcXTrp31 LC10bcXShi31 LC10XKir31", "LC10bcXTrp22 LC10bcXShi22",
            # "LC10XKir22 LC10XKir25 LC10XKir31 LC10XKir222",
            # "LC10adXKir31 LC10adXKir222 LC10adXTNTE231 LC10adXTNTE222",
        ]
    # elif data_n == "data3s":
        # driver_l = [
        #     "CS31 CSXTrp31 CSXShi31 CS22 CSXTrp22 CSXShi22",
            # "TBXCS31 TBXTrp31 TBXShi31 TBXCS22 TBXTrp22 TBXShi22",
            # "LC10adXCS31 LC10adXTrp31 LC10adXShi31 LC10adXCS22 LC10adXTrp22 LC10adXShi22",
            # "EPGXCS31 EPGXTrp31 EPGXShi31 EPGXCS22 EPGXTrp22 EPGXShi22",
            # "FBSNPXCS31 FBSNPXTrp31 FBSNPXShi31 FBSNPXCS22 FBSNPXTrp22 FBSNPXShi22",
            # "LC16XCS31 LC16XTrp31 LC16XShi31 LC16XCS22 LC16XTrp22 LC16XShi22",
            # "LC12XCS31 LC12XTrp31 LC12XShi31 LC12XCS22 LC12XTrp22 LC12XShi22",
            # "ZFBSNPXShi31 ZFBSNPXCS31",
        # ]
        # driver_l = [
        #     "CSXShi31 CSXTrp31",
        #     "LC16XCS31 LC16XTrp31",
        #     "LC12XCS31 LC12XTrp31",
        #     "EPGXCS31 EPGXTrp31",
        #     "FBSNPXCS31 FBSNPXTrp31",
        #     "LC10adXCS31 LC10adXTrp31",
        #     "TBXCS31 TBXTrp31",
        # ]
    elif data_n == "data4":
        # driver_l = [
        #     # "CSXCS31 CSXTrp31 CSXShi31 CS22 CSXTrp22 CSXShi22",
        #     "CSXCS31 CSXTrp31 CSXShi31", "CS22 CSXTrp22 CSXShi22",
        #     "LC12XCS31 LC12XTrp31 LC12XShi31", "LC12XCS22 LC12XTrp22 LC12XShi22",
        #     "LC6XCS31 LC6XTrp31 LC6XShi31",
        #     "LC18XCS31 LC18XTrp31 LC18XShi31",
        #     "LC20XCS31 LC20XTrp31 LC20XShi31",
        #     "LC11XCS31 LC11XTrp31 LC11XShi31",
        #     "T45XCS31 T45XTrp31 T45XShi31",
        #     "LC26XCS31 LC26XTrp31 LC26XShi31",
        #     "LC24XCS31 LC24XTrp31 LC24XShi31",
        #     "LPLC2XCS31 LPLC2XTrp31 LPLC2XShi31",
        #     "CS31 MW31 FW31",
        #     # "FBcXShi31 FBcXTrp31",
        # ]
        driver_l = [
            "LC16XCS31 LC16XShi31 LC16XTrp31",
            "LC12XCS31 LC12XShi31 LC12XTrp31",
            "LC6XCS31 LC6XShi31 LC6XTrp31",
            "LC18XCS31 LC18XShi31 LC18XTrp31",
            "LC20XCS31 LC20XShi31 LC20XTrp31",
            "LC11XCS31 LC11XShi31 LC11XTrp31",
            "T45XCS31 T45XShi31 T45XTrp31",
            "LC26XCS31 LC26XShi31 LC26XTrp31",
            "LC24XCS31 LC24XShi31 LC24XTrp31",
            "LPLC2XCS31 LPLC2XShi31 LPLC2XTrp31",
        ]
    elif data_n.startswith("data5"):
        driver_l = ["CS22 FW22 MW22 FWMW22"]
    elif data_n.startswith("data7"):
        # driver_l = ["CS22 IRCS22 IRDCS22", "CSIR22 IR22 IRD22"]
        # driver_l = ["CS22 A22", "IR22 AIR22", "IRCS22 AIRCS22"]
        # driver_l = ["A22 At22"]
        driver_l = ["A22 At22 CS22", "AIRCS22 IRCS22 IRDCS22",
                    "AIR22 IR22 IRD22", "CSIR22"]
    elif data_n == "data8" or data_n == "data8s":
        driver_l = ["B22 BD22 BDB22"]
    elif data_n == "data9":
        driver_l = ["IR22 MWIR22"]
    elif data_n == "data10":
        driver_l = ["CS22 MW22"]
    elif data_n == "data4s":
        driver_l = ["FBSNPXCS31 FBSNPXShi31 FBSNPXTrp31",
                    "LC12XCS31 LC12XShi31 LC12XTrp31",
                    "LC16XCS31 LC16XShi31 LC16XTrp31",]
        # driver_l = ["CSXCS31 CSXShi31 CSXTrp31",
        #             "SNPXCS31 SNPXShi31 SNPXTrp31",
        #             "LC12XCS31 LC12XShi31 LC12XTrp31",
        #             "LC16XCS31 LC16XShi31 LC16XTrp31",
        #             "LC6XCS31 LC6XShi31 LC6XTrp31",
        #             "LC20XCS31 LC20XShi31 LC20XTrp31",
        #             "LC18XCS31 LC18XShi31 LC18XTrp31",
        #             "LC11XCS31 LC11XShi31 LC11XTrp31",
        #             "T45XCS31 T45XShi31 T45XTrp31",
        #             "LC26XCS31 LC26XShi31 LC26XTrp31",
        #             "LC24XCS31 LC24XShi31 LC24XTrp31",
        #             "LPLC2XCS31 LPLC2XShi31 LPLC2XTrp31",]
    elif data_n == "data_hrnet":
        driver_l = ["CS22 FW22 A22", "IR22 MW22 MWIR22 AIR22"]
    elif data_n == "data_suzukii_mated":
        driver_l = ["CtrlTwo25", "ExpTwo25", "CtrlOne25", "ExpOne25", "ExpNone25", "Biar25"]
    elif data_n == "data_suzukii":
        driver_l = ["CtrlTwo25", "ExpTwo25", "ExpOne25", "ExpNone25"]
    elif data_n == "data_screen":
        driver_l = ["TBXCS31 TBXShi31 TBXTrp31", "LC10adXCS31 LC10adXShi31 LC10adXTrp31",
                    "LC15XCS31 LC15XShi31 LC15XTrp31", "LC22XCS31 LC22XShi31 LC22XTrp31"]
    elif data_n == "data_LPLC1":
        driver_l = ["LPLC1XCS31 LPLC1XShi31 LPLC1XTrp31"]
    elif data_n == "data_Ort":
        # driver_l = ["OrtXCS25", "OrtXTNTE25", "OrtXTNTin25", "OrtXKO25"]
        driver_l = ["CS25", "OrtXKO25"]
    elif data_n == "data_part":
        driver_l = ["Head25 HeadS25 HeadW25 HeadM25 HeadSim25 Abdo25"]
    elif data_n == "data_TB2":
        # driver_l = ["Head25", "HeadS25", "HeadM25", "HeadSim25"]
        driver_l = ["TBXCS31 TBXShi31 CSXShi31"]
    elif data_n == "data_lz":
        driver_l = ["fore2courtXCsC_Nb25b5r6.850Hz10ms fore3courtXCsC_Nb25b5r6.850Hz10ms hind2courtXCsC_Nb25b5r6.850Hz10ms middle2courtXCsC_Nb25b5r6.850Hz10ms WTXCsC_Nb25b5r6.850Hz10ms"]
    else:
        #driver_l = ["CS22"]
        # driver_l = ["CS25"]
        driver_l = DRIVER_L
    PLOT_DRIVER_GROUP = len(driver_l[0].split())
    geno_l = np.concatenate([driver.split() for driver in driver_l])
    if cmd:
        argv = ("1 " + cmd).split()
    else:
        argv = sys.argv
    cmd = argv[1]
    data_type = os.path.basename(DATA_DIR)
    if cmd == "p":
        for geno in tqdm(sorted(set(geno_l))):
            geno_folder = os.path.join(GENO_DATA_DIR, geno)
            if os.path.exists(geno_folder):
                clear_align_cache()
                plot_for_pair_geno(geno_folder, data_type)
                plt.close("all")
    elif cmd == "pe":
        # update_geno_info()
        for geno in tqdm(sorted(set(geno_l))):
            geno_folder = os.path.join(GENO_DATA_DIR, geno)
            if os.path.exists(geno_folder):
                plot_for_geno(geno_folder, "all")
                plot_for_geno(geno_folder, "center")
                plt.close("all")
    elif cmd == "pd":
        plot_for_driver(geno_l, driver="A")
        # plot_for_driver([
        #     "CSXCS31", "CSXShi31", "CSXTrp31",
        #     "LC24XCS31", "LC24XShi31", "LC24XTrp31",
        #     "LC26XCS31", "LC26XShi31", "LC26XTrp31"], driver="LC24_LC26")
        # plot_for_driver([
        #     "CSXCS31", "CSXShi31", "CSXTrp31",
        #     "LC6XCS31", "LC6XShi31", "LC6XTrp31",
        #     "LC18XCS31", "LC18XShi31", "LC18XTrp31"], driver="LC6_LC18")
        # PLOT_DRIVER_GROUP = 2
        # plot_for_driver([
        #     "CSXCS31", "CS31", "CSXShi31", "CSXTrp31",
        #     "T45XCS31", "T45XShi31", "T45XTrp31"], driver="T4T5")
        # plot_for_driver(get_all_geno31(driver_l), driver="A")
        # plot_for_driver(get_all_geno31(driver_l, "CS31"), driver="C")
    elif cmd == "pdt":  # NOTE: multi info cross data1~4
        # driver_out_d = {"E:/data1": ["CS31", "IR22"], "E:/data4": ["FW31"]}
        # driver_out_d = {"E:/data1": ["CS31", "IR22", ], "E:/data2": ["CS22"], "E:/data3": ["CS22"], "E:/data4": ["CS31", "FW31"]}
        # driver_out_d = {"G:/data_screen": ["TBXCS31", "TBXShi31", ], "F:/exp/data_TB2": ["TBXCS31", "TBXShi31", "CSXShi31"]}
        driver_out_d = {"E:/LC_screening/NJ/CS": ["CSXCS31 CSXTrp31 CSXShi31"],
                        "E:/LC_screening/NJ/TB": ["TBXCS31 TBXTrp31 TBXShi31"],
                        "E:/LC_screening/NJ/T45": ["T45XCS31 T45XTrp31 T45XShi31"],
                        "E:/LC_screening/NJ/LPLC1": ["LPLC1XCS31 LPLC1XTrp31 LPLC1XShi31"],
                        "E:/LC_screening/NJ/LPLC2": ["LPLC2XCS31 LPLC2XTrp31 LPLC2XShi31"],
                        "E:/LC_screening/SYF/LC4": ["CSXTrp31 LC4XTrp31 LC4XShi31"],
                        "E:/LC_screening/NJ/LC6": ["LC6XCS31 LC6XTrp31 LC6XShi31"],
                        "E:/LC_screening/SYF/LC9": ["CSXTrp31 LC9XTrp31 LC9XShi31"],
                        "E:/LC_screening/NJ/LC10ad": ["LC10adXCS31 LC10adXTrp31 LC10adXShi31"],
                        "E:/LC_screening/NJ/LC11": ["LC11XCS31 LC11XTrp31 LC11XShi31"],
                        "E:/LC_screening/NJ/LC12": ["LC12XCS31 LC12XTrp31 LC12XShi31"],
                        "E:/LC_screening/SYF/LC13": ["CSXTrp31 LC13XTrp31 LC13XShi31"],
                        "E:/LC_screening/NJ/LC15": ["LC15XCS31 LC15XTrp31 LC15XShi31"],
                        "E:/LC_screening/NJ/LC16": ["LC16XCS31 LC16XTrp31 LC16XShi31"],
                        "E:/LC_screening/SYF/LC17": ["CSXTrp31 LC17XTrp31 LC17XShi31"],
                        "E:/LC_screening/NJ/LC18": ["LC18XCS31 LC18XTrp31 LC18XShi31"],
                        "E:/LC_screening/NJ/LC20": ["LC20XCS31 LC20XTrp31 LC20XShi31"],
                        "E:/LC_screening/SYF/LC21": ["CSXTrp31 LC21XTrp31 LC21XShi31"],
                        "E:/LC_screening/NJ/LC22": ["LC22XCS31 LC22XTrp31 LC22XShi31"],
                        "E:/LC_screening/NJ/LC24": ["LC24XCS31 LC24XTrp31 LC24XShi31"],
                        "E:/LC_screening/SYF/LC25": ["CSXTrp31 LC25XTrp31 LC25XShi31"],
                        "E:/LC_screening/NJ/LC26": ["LC26XCS31 LC26XTrp31 LC26XShi31"]}
        SWARM_PALATTE = [a for a in "krb" * 22]
        PLOT_DRIVER_GROUP = 3
        # plot_for_driver(["ALL"], geno_folders_out_d=driver_out_d)
        for f in SUMMARY_FIGURES:
           plot_all_pair_out(driver_out_d, "center", f)
        #plot_all_pair_out(driver_out_d, "center", "c_pos-female")
        # driver_out_d = {
        #     "G:/data4s": ["CSXCS31 CSXShi31 CSXTrp31",
        #                   "LC24XCS31 LC24XShi31 LC24XTrp31",
        #                   "LC26XCS31 LC26XShi31 LC26XTrp31",
        #                   "LC6XCS31 LC6XShi31 LC6XTrp31",
        #                   "LC11XCS31 LC11XShi31 LC11XTrp31",
        #                   "LC18XCS31 LC18XShi31 LC18XTrp31",
        #                   "T45XCS31 T45XShi31 T45XTrp31"],
        #     "G:/data_screen": ["TBXCS31 TBXShi31 TBXTrp31",
        #                        "LC10adXCS31 LC10adXShi31 LC10adXTrp31",
        #                        "LC15XCS31 LC15XShi31 LC15XTrp31",
        #                        "LC22XCS31 LC22XShi31 LC22XTrp31"]
        # }
        # plot_all_pair_out(driver_out_d, "center", "fc_speed")
        # plot_by_info(r"G:\data4s\center\_info\LC24XShi31-summary_fc_speed.pickle")
    elif cmd == "pa":
        geno_l = [driver.split() for driver in driver_l]
        # if len(geno_l[0]) > 3:
        plot_all_pair(np.array(geno_l), data_type)
        # else:
        #     plot_all_pair(np.array(geno_l).T, data_type)
    elif cmd == "pg":  # NOTE: center
        # NOTE: pg CS31 fc_speed-male
        plot_geno(argv[2], argv[3])#, "_we_stat0.pickle")
    elif cmd == "pgd":
        # NOTE: pgd LC20XCS31 LC20XTrp31 fc_speed-male
        plot_geno_diff(argv[2:-1], argv[-1])
    elif cmd == "pi":  # individual
        # NOTE: pi LC20XTrp31 c_pos_nt-female
        # NOTE: pi CS22 track_overlap-female-rel_pos,x-rel_pos,y
        geno = argv[2]
        parent = os.path.join(GENO_DATA_DIR, geno)
        # mkdir(os.path.join(DATA_DIR, "_img", argv[3]))
        info_l = []
        for pair in os.listdir(parent):
            pair_folder = os.path.join(parent, pair)
            if os.path.isdir(pair_folder) and not pair_folder.endswith("remove"):
                dfs, bouts, n = load_dfs_bouts(pair_folder)
                info_l.append([pair, dfs, bouts, len(bouts)])
        info_l.sort(key=lambda x: x[-1], reverse=True)

        rows, cols = int(len(info_l) / 6 + 0.99), 6
        FIGURE_W = 12
        fz = (cols * FIGURE_W, rows * FIGURE_W)
        fig, axes = plt.subplots(rows, cols, figsize=fz)
        axes = axes.flatten()
        HEAT_CONFIG["min_bin"] = 0
        HEAT_CONFIG["color_bar"] = 0
        for i, info in enumerate(info_l):
            pair, dfs, bouts, cir_count = info
            plot_by_name(dfs, os.path.join(DATA_DIR, "_img/%s,%s-%s" % (geno, pair, argv[3])), bouts=bouts, save_pickle=False, ax=axes[i])
            axes[i].set_title(pair + " n%d" % len(bouts), fontsize=23)
        plt.suptitle("%s,%s" % (argv[3], geno))
        save_and_open(os.path.join(DATA_DIR, "_img/%s,%s" % (argv[3], geno)))
    elif cmd == "pt":
        # NOTE: pt CS22 r_alpha_fab
        geno = argv[2]
        parent = os.path.join(GENO_DATA_DIR, geno)
        for pair in os.listdir(parent):
            pair_folder = os.path.join(parent, pair)
            if os.path.isdir(pair_folder) and not pair_folder.endswith("remove"):
                dfs = load_dfs(pair_folder)
                cir_meta = load_dict(get_all_cir_meta(pair_folder)[0])
                plt.figure()
                ax = plt.gca()
                ret = plot_figure(ax, dfs, 1, argv[3], cir_meta=cir_meta)
                save_name = "img/%s/%s" % (argv[3], pair)
                save_and_open(save_name)
                pf = open(save_name + ".pickle", "wb")
                pickle.dump(to_list(ret), pf)
                pf.close()
    elif cmd == "upcir":
        update_all_cir_info()
    elif cmd == "geno":
        # NOTE: geno CSXCS31\20190913_141035_2_3
        # update_one_pair("G:/_video_hrnet_finish/20200602_144311_1/11")
        mp_process_files(update_geno_one_pair, get_all_cir_meta(os.path.join(GENO_DATA_DIR, argv[2])), 0)
    elif cmd == "u":
        # 20190724_161718_1_18-19, 20190727_151557_1_13-15
        # update_from_raw_data(get_all_pair_folder(VIDEO_ANA_DIR))
        # update_ana_data_pre()
        geno = argv[2] if len(argv) > 2 else None
        # update_ana_data(geno)
        # update_ana_data_post(geno, data_type="center")
        update_geno_info(geno)

        # update_from_raw_data(get_all_pair_folder(r"E:\video3_fps30\20190916_162500_B"), force=True)
        # for ge in ["LC6", "LC20", "LC18", "LC11", "T45", "LC26", "LC24", "LPLC2", "LC12", "FBSNP", ]:
        # for ge in ONLY_UPDATE_GENO:
        #     update_ana_data_post(ge, data_type="all")
        #     update_geno_info(ge)
        # update_ana_data_pre()
    elif cmd == "all":
        # mp_process_files(update_one_pair, get_all_pair_folder("G:/video2_cross"), POOL_SIZE)
        # mp_process_files(update_one_pair, get_all_pair_folder("G:/video3_fps30"), POOL_SIZE)
        # mp_process_files(update_one_pair, get_all_pair_folder("G:/_video_screen"), POOL_SIZE)
        # mp_process_files(update_one_pair, get_all_pair_folder("G:/_video_suzukii"), POOL_SIZE)
        # mp_process_files(update_one_pair, get_all_pair_folder("G:/_video_hrnet_finish"), POOL_SIZE)
        # mp_process_files(update_one_pair, get_all_pair_folder(r"G:\_video_LPLC1"), POOL_SIZE)
        # mp_process_files(update_one_pair, get_all_pair_folder(r"G:\_video_dlc\video3_LC10_LC12_fps30"), POOL_SIZE)
        # mp_process_files(update_one_pair, get_all_pair_folder(r"G:\_video_todo"), POOL_SIZE)
        # mp_process_files(update_one_pair, get_all_pair_folder(r"G:\_video_dlc\video4_LC16_LPLC2_fps66"), 0)
        # mp_process_files(update_one_pair, get_all_pair_folder(r"F:\exp\_video_part_todo"), 8)
        mp_process_files(update_one_pair, get_all_pair_folder(VIDEO_DIR), 16)
        # mp_process_files(update_one_pair, get_all_pair_folder(r"G:\video_old\old3_fix"), POOL_SIZE)
        # update_geno_info("CSXShi31")
        update_geno_info()
        # ana_main("pd")
        # ana_main("pe")
        # ana_main("p")
        # plot_for_pair_geno(os.path.join(GENO_DATA_DIR, "CSXShi31"), "center")

        # ana_main("pa")
        # ana_main("pdt")


if __name__ == "__main__":
    # NOTE: all 7
    # NOTE: all video7_ir
    # NOTE: all video7_ir/20200602_144311_A
    # NOTE: geno CS22
    # NOTE: u CSXTrp31
    ana_main()
    # mp_process_files(update_one_pair, get_all_pair_folder("G:/video3_fps30"), POOL_SIZE)  # test WIPS
    # ana_mai[n("pt CS22 r_alpha_fab")
    # plot_hist_in_folder("img/r_alpha_fab")
    # ana_main("p")
    # ana_main("u", "9")
    # plot_etho_for_video(r"F:\temp\video_todo\20200602_144311_1")
    # plot_etho_for_video(r"F:\temp\video_todo\20200602_144312_2")
    # plot_etho_for_video(r"G:\video7_ir\20200602_144311_1")
    # plot_etho_for_video(r"G:\video7_ir\20200602_144312_2")
    # ana_main("pi LC20XTrp31 c_pos_nt-female")
    # ana_main("pi LC20XShi31 c_pos_nt-female")
    # ana_main("pi LC20XCS31 c_pos_nt-female")
    # ana_main("pi LC20XTrp31 c_pos-female")
    # ana_main("pi LC20XShi31 c_pos-female")
    # ana_main("pi LC20XCS31 c_pos-female")
    # ana_main("pi LC20XTrp31 c_pos_nh-female")
    # ana_main("pi LC20XShi31 c_pos_nh-female")
    # ana_main("pi LC20XCS31 c_pos_nh-female")
