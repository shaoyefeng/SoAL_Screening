# -*- coding: utf-8 -*-

import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import NoNorm
from scipy.stats import mannwhitneyu

from fpt_consts import FLY_AVG_WID, DIST_TO_FLY_INNER_H, DIST_TO_FLY_INNER_T, BODY_SIZE
from fpt_frame_stat import ANGLE_WE_MIN
from fpt_util import *

SHOW = False
PLOT_VIDEO = True
FONT_SIZE = 6
FIG_SCALE = FONT_SIZE/12
USE_SEM = True
USE_MEAN_PAIR = True

FIGURE_FILE_EXTENSION = ".png"
FIGURE_VEC_EXTENSION = ".png"
SAVE_SVG = True

HEAT_CONFIG = {
    "min_bin": 5, #10
    "color_bar": 0,
    "fill_fig": 0,
    "axis_off": 1,
    "only_plot_arrow": 0,
    "plot_mean": 0,
    "body_line": 100000,
    "plot_scatter_axis": 0,
    "fly_body_ellipse": 0,
    "fly_body_alpha": 0.5,
    "female_fast": None,
}
HEAT_BINS = 50
STREAM_BINS = 20
HEAT_BLANK = False
PLOT_HEAT_MEAN = True
PLOT_HEAT_HALF = True
PLOT_HEAT_ACCUM = False
ALIGN_HEAT_MEAN = False
PLOT_FLY_BODY = True
CENTER_FLY = 2
WEIGHT_HEAT_X = "rel_pos:x"  #_h
WEIGHT_HEAT_Y = "rel_pos:y"  #_h
# WEIGHT_HEAT_X = "rel_polar:t"  #_hh
# WEIGHT_HEAT_Y = "rel_polar:r"  #_hh
if WEIGHT_HEAT_X.startswith("rel_polar"):
    PLOT_HEAT_MEAN = True
    PLOT_FLY_BODY = False
else:
    PLOT_HEAT_MEAN = False
    PLOT_HEAT_HALF = False
if WEIGHT_HEAT_X.endswith("_h:x"):
    PLOT_FLY_BODY = False

ONLY_CALC_PLOT = False
USE_HIST_LINE = False
PRINT_HEAT_XY_VIDEO = None#(-4.8, 3.12)#(0, 3)


COLOR_HEAD = "#863285"#(134, 50, 133)#"purple"
COLOR_SIDE = "#808080"
COLOR_SIDE_TO_HEAD = "lightsteelblue"
COLOR_SIDE_TO_TAIL = "burlywood"
COLOR_NEAR = "#308014"
COLOR_FAR = "#82D900"
COLOR_TAIL = "#E89023"#(232, 144, 35)#"orange"

COLOR_FORWARD = COLOR_HEAD
COLOR_BACKWARD = COLOR_TAIL

COLOR_LEFT = "#b70e5e"
COLOR_RIGHT = "#6AC7E2"#0EB767
# COLOR_RIGHT = "#CFDB00"#"#4D4398"
# COLOR_LEFT = "#00a0da"
# COLOR_RIGHT = "#9c462f"
# COLOR_LEFT = "#00567c"
# COLOR_RIGHT = "#bf7ab1"
# COLOR_LEFT = "#00a0da"
# COLOR_RIGHT = "#bf7ab1"

COLOR_VS = "#004b9a"
COLOR_VF = "#2b9aad"
COLOR_ON_LEFT = "g"
COLOR_ON_RIGHT = "y"
COLOR_ON_LEFT_CIR = "#143c3b"#"#197679"
COLOR_ON_RIGHT_CIR = "#ab7c22"#"#b4b47a"

COLOR_START = "deepskyblue"
COLOR_END = "red"

COLOR_FAST = "g"
COLOR_SLOW = "m"

COLOR_MAP = {
    "head": "blueviolet",
    "side": "gray",
    "tail": "orangered",
    "left": "orange",
    "right": "purple",
}

ARENA_RANGE_X = (0, 20)
ARENA_RANGE = (ARENA_RANGE_X, ARENA_RANGE_X)
CENTER_RANGE_R = 6# 7.5
CENTER_RANGE_X = (-CENTER_RANGE_R, CENTER_RANGE_R)  # NOTE: 6 (c_pos, fc, se, switching)
CENTER_RANGE = (CENTER_RANGE_X, CENTER_RANGE_X)
CENTER_RANGE_R2 = 7.5
CENTER_RANGE_X2 = (-CENTER_RANGE_R2, CENTER_RANGE_R2)  # NOTE: 7.5 (stream)
CENTER_RANGE2 = (CENTER_RANGE_X2, CENTER_RANGE_X2)
CENTER_OVERLAP_RANGE_X = [-10, 10]  # NOTE: 10 (track_overlap)
CENTER_H_RANGE_X = [-3, 3]  # NOTE: 3 (c_pos_h, c_pos_h_nh/nt)
CENTER_H_RANGE_Y = [-1.2, 4.8]
CENTER_T_RANGE_X = [-3.6, 3.6]  # NOTE: 3.6 (c_pos_t, c_pos_t_nh/nt)
CENTER_T_RANGE_Y = [-1.2, 6]
CENTER_T2_RANGE_X = [-3.8, 3.8]  # NOTE: c_cw
CENTER_T2_RANGE_Y = [-1.2, 6.4]

CENTER_WE_RANGE_X = [-4.7, 4.7]  # NOTE: 4 (wc)
CENTER_WE_RANGE_Y = [-1.4, 8]
POLAR_RANGE = ((-180, 180), (0, 6))
COLOR_FOR_FLY = {0: "y", 1: "b", 2: "r", 3: "gray"}
FLY_ID = {"male": 1, "female": 2, "1": 1, "2": 2, "0": 0}
STAT_LIMIT = {
    0: {"dist_McFc": [0, 6],},
    1: {
        "rel_pos:x": CENTER_RANGE_X,
        "rel_pos:y": CENTER_RANGE_X,
        "rel_pos_h:x": [-4, 4],
        "rel_pos_t:x": [-4, 4],
        "rel_pos_h:y": [0, 6],
        "rel_pos_t:y": [0, 6],
        "pos:x": ARENA_RANGE_X,
        "pos:y": ARENA_RANGE_X,
        "v_len": [0, 10],  # 10
        "av": [-400, 400],
        "abs_av": [0, 400],
        "dist_McFc": [0, 6],
        "rel_polar_hh:r": [0, 4.5],
        "rel_polar_ht:r": [0, 4.5],
        "rel_polar_h:t": [0, 180],#[-180, 360],
        "theta": [-180, 180],
        "wing_l": [-100, 100],
        "wing_r": [-100, 100],
        "wing_m": [0, 120],
        "e_maj": [1.6, 3],
        "e_min": [0.5, 1.5],
        "vs": [-20, 20],
        "rel_polar:r": [0, 6],
        "rel_polar_h:r": [0, 4.5],
    },
    2: {
        "rel_pos_h:x": [-4, 4],
        "rel_pos_h:y": [-6, 6],
        "e_maj": [1.6, 3],
        "e_min": [0.5, 1.5],
        "v_len": [0, 8],
        "rel_polar_hh:r": [0, 4.5],
        "rel_polar_ht:r": [0, 4.5],
        "rel_polar:r": [0, 6],
        "rel_polar_h:r": [0, 4.5],
    },
    "male_len": [0, 2.5],
    "mc_0:dist_McFc": [0, 6],
    "me_0:dist_McFc": [0, 6],
    "mc_1:abs_acc": [0, 100],
    "me_1:abs_acc": [0, 100],
    "mnh_1:abs_acc": [0, 100],
    "mnt_1:abs_acc": [0, 100],
    "mih_1:abs_acc": [0, 100],
    "mit_1:abs_acc": [0, 100],
    "mnl_1:abs_acc": [0, 100],
    "mil_1:abs_acc": [0, 100],
    "mc_1:abs_av": [0, 200],
    "me_1:abs_av": [0, 200],
    "mnh_1:abs_av": [0, 200],
    "mnt_1:abs_av": [0, 200],
    "mih_1:abs_av": [0, 200],
    "mit_1:abs_av": [0, 200],
    "mnl_1:abs_av": [0, 200],
    "mil_1:abs_av": [0, 200],
    "mc_1:v_len": [0, 8],
    "me_1:v_len": [0, 4],
    "mnh_1:v_len": [0, 12],
    "mnt_1:v_len": [0, 8],
    "mih_1:v_len": [0, 12],
    "mit_1:v_len": [0, 8],
    "mnl_1:v_len": [0, 8],
    "mil_1:v_len": [0, 8],
    "mc_1:abs_vs": [0, 8],
    "me_1:abs_vs": [0, 2],
    "mnh_1:abs_vs": [0, 12],
    "mnt_1:abs_vs": [0, 8],
    "mih_1:abs_vs": [0, 12],
    "mit_1:abs_vs": [0, 8],
    "mnl_1:abs_vs": [0, 12],
    "mil_1:abs_vs": [0, 12],
    "mc_1:vf": [0, 4],
    "me_1:vf": [0, 2],
    "mnh_1:vf": [0, 4],
    "mnt_1:vf": [0, 4],
    "mih_1:vf": [0, 4],
    "mit_1:vf": [0, 4],
    "mnl_1:vf": [0, 4],
    "mil_1:vf": [0, 4],
    "mw_1:vf": [0, 4],
    "mc_1:we_lr": [0, 0.1],
    "mnh_1:we_lr": [0, 0.1],
    "mnt_1:we_lr": [0, 0.02],
    "mih_1:we_lr": [0, 0.1],
    "mit_1:we_lr": [0, 0.02],
    "mnl_1:we_lr": [0, 0.1],
    "mil_1:we_lr": [0, 0.1],
    "mw_1:we_lr": [0, 0.002],
    "me_2:rel_polar_h:r": [0, 4],
    "me_1:rel_polar_ht:r": [0, 3],
    "mnh_1:rel_polar_hh:r": [0, 3],
    "mnt_1:rel_polar_ht:r": [0, 2],
    "me_1:nh": [0, 0.5],
}
STAT_LIMIT_SWARM = {
    "cir_per_second": [0, 0.03],
    "cir_per_minute": [0, 3],
    # "cir_duration": [0, 1.5],
    "cir_ratio": [0, 0.1],
    "ici": [0, 400],
    "mc_1:p_to_h": [0, 1],

    "mc_0:dist_McFc": [0, 4],
    "mc_1:nh_index": [0, 1],
    "mc_1:nh": [0, 1],
    "mc_1:nt": [0, 1],
    "mc_1:wing_m": [0, 100],
    "me_1:wing_m": [0, 100],
    "mnh_1:rel_polar:r": [0, 6],
    "w_we_ratio": [0, 0.3],
    "mc_1:we_ipsi": [-1, 1],
    "me_1:we_ipsi": [-1, 1],
    "mef_1:we_ipsi": [-1, 1],

    "Standard deviation\nof $\\alpha^{McFc}$ (°)": [0, 100],
}
STAT_POINT_SIZE = {
    # "rel_pos:x": 2,
    "pos:x": 15,
    "theta": 0.2,
}
HEAT_LIMIT = {
    0: {
        "dist_McFc": [0, 5],
    },
    1: {
        "pos:x": [0, 0.005],
        "rel_pos:x": [0, 0.026],
        "rel_pos_h:x": [0, 0.025],
        "rel_pos_t:x": [0, 0.025],
        "rel_polar:r": [0, 4],
        "nh": [0, 0.8],
        "nt": [0, 0.8],
        "we_ipsi": [0, 0.5],
        "wing_l": [-100, 100],
        "wing_r": [-100, 100],
        "wing_m": [30, 85],

        "v_len": [0, 9],
        "theta": [-50, 50],
        "acc": [-10, 10],
        "av": [30, 180],

        "vf": [-1, 4],
        "vs": [-7.5, 7.5],
        "acc_len": [0, 1.5],
        "e_maj": [1.6, 2.5],
    },
    2: {
        # "pos:x": [0, 0.008],
        "rel_pos:x": [0, 0.006],
        # "rel_pos_h:x": [0, 0.02],
        "rel_polar_h:r": [0, 3.5],
        # "rel_pos_t:x": [0, 0.0025],
    },
    "head": {
        "rel_polar:r": [2, 4.5],
        "abs_rel_polar:t": [0, 180],
        "rel_polar:t": [-180, 180],
        "v_len": [0, 9],
        "speed": [0, 9],
        "av": [0, 180],
        "abs_av": [0, 180],
        "wing_m": [30, 90],
        "theta": [-60, 60],
        "abs_theta": [50, 100],
        "vf": [-1, 4],
        "vs": [-7.5, 7.5],
        "abs_vs": [0, 15],
        "acc": [-7, 7],
        "abs_acc": [15, 40],
        "we_ipsi": [-0.5, 1],
    },
    "start": {
        "rel_polar:r": [2, 6],
        "abs_rel_polar:t": [0, 180],
        "rel_polar:t": [-180, 180],
        "v_len": [0, 15],
        "av": [0, 220],
        "wing_m": [0, 100],
        "theta": [-60, 60],
        "abs_theta": [50, 100],
        "vf": [-1, 4],
        "vs": [-7.5, 7.5],
        "abs_vs": [0, 15],
        "acc": [-7, 7],
        "abs_acc": [15, 40],
        "speed": [0, 15],
        "abs_av": [80, 220],
    },
}
STAT_ALIAS = {
    "dist_MhFh": "rel_polar_hh:r",
    "dist_MhFt": "rel_polar_ht:r",
    "dist_MhFc": "rel_polar_h:r",
    "dist_MtFc": "rel_polar_t:r",
    "speed": "v_len",
}
REPLACE_LABEL = {
    "v_len": "$|v|$ (mm/s)",
    "v_dir": "$\\varphi_v$ (°)",
    "speed": "$|v|$ (mm/s)",
    "theta": "$\\theta$ (°)",
    "abs_theta": "$|\\theta|$ (°)",
    "wing_m": "$\\beta_{max}$ (°)",
    "acc": "$A$ (mm/$s^2$)",
    "abs_acc": "$|A|$ (mm/$s^2$)",
    "abs_av": "$|\\omega|$ (°)",
    "rel_polar:r": "$D^{McFc}$ (mm)",
    "rel_polar:t": "$\\alpha$ (°)",
    "abs_rel_polar:t": "$|\\alpha|$ (°)",
    "abs_vs": "$|v_s|$ (mm/s)",
    "vf": "$v_f$ (mm/s)",
    "we_ipsi": "WCI",
}
REPLACE_TITLE = {
    "rel_polar:r": "Distance",
    "abs_rel_polar:t": "$\\alpha$",
    "abs_90_rel_polar:t": "$\\alpha$",
    "speed": "Velocity (|v|)",
    "vf": "Forward velocity (vf)",
    "abs_vs": "Side velocity (|vs|)",
    "abs_av": "Angular velocity (|av|)",
    "acc": "Linear acceleration (accl)",
    "abs_acc": "Absolute linear acceleration (|accl|)",
    "acc_len": "Acceleration (|acc|)",
    "acc_dir": "Acceleration direction (acc_dir)",
    "v_dir": "Velocity direction (v_dir)",
    "theta": "Sidewalk angle (θ)",
    "abs_theta": "Absolute sidewalk angle (|θ|)",
    "wing_m": "Larger wing angle (we)",
}
def k_with_dfk(x, df):
    x = STAT_ALIAS.get(x, x)
    if x.startswith("abs_"):
        x = x[4:]
        if x.startswith("90_"):
            x = x[3:]
            dfx = abs(df[x] - REL_POLAR_T_OFFSET)
        else:
            dfx = abs(df[x])
    elif x.startswith("lim20_"):
        x = x[6:]
        dfx = np.clip(df[x], 0, 20)
    elif x.startswith("90_"):
        x = x[3:]
        dfx = df[x] - REL_POLAR_T_OFFSET
    elif x == "1":
        dfx = np.ones((len(df),))
    # elif x.find("Y") > 0:
    #     x1, x2 = x.split("Y")
    #     dfx = df[x1] / df[x2]
    elif x.startswith("sd_"):
        dfx = None
    else:
        dfx = df[x]
    return x, dfx

def plot_angles_bar(bin_l, hist_f, bins):
    angles = np.deg2rad(np.array(bin_l) + 90)
    # plt.polar(angles, hist_f, "b")
    plt.bar(angles[:-1], hist_f[:-1], color="b", width=np.pi * 2 / bins, alpha=0.5)
    labels = np.array([0, 90, 180, 270])
    plt.thetagrids(labels, labels - 90)

def to_list(info):
    if not info:
        return info
    for i, r in enumerate(info):
        if isinstance(r, list):
            info[i] = to_list(r)
        elif isinstance(r, pd.Series):
            info[i] = r.tolist()
        elif isinstance(r, np.ndarray):
            info[i] = r.tolist()
    return info

def lines_to_err_band(py_l):
    py = np.nanmean(py_l, axis=0)
    err_l = np.nanstd(py_l, axis=0)
    if USE_SEM:
        err_l = err_l / np.sqrt(np.count_nonzero(~np.isnan(py_l), axis=0))
    return py, err_l

# weight_heat-male-v_len
def plot_summary_by_name(names, dfs, filename, bouts=[], w_sc=1.2, h_sc=1, bottom=0.06, col=4, space=0.3, left=0.06, right=0.96, cb=None, n=1, save_pickle=False, save_svg=True, need_title=False, info_dir=None, sharex=False, count=None):
    fig_count = len(names)
    if not col:
        col = fig_count
    row = int(fig_count / col)
    if row * col < fig_count:
        row += 1
    global g_fig
    g_fig = plt.figure(filename, figsize=(col * 3.6 * w_sc, row * 3.6 * h_sc))
    plt.clf()
    plt.subplots_adjust(left=left, right=right, top=1 - bottom, bottom=bottom, hspace=space, wspace=space)
    dir_name, f_name = os.path.split(filename)
    for idx, name in enumerate(names):
        if not name:
            continue
        name = name.replace(",", ":")
        print("plot summary: ", name)
        fname_t = name.split("-")
        if name.startswith("polar"):
            ax = plt.subplot(row, col, idx + 1, projection="polar")
        else:
            ax = plt.subplot(row, col, idx + 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        l = len(fname_t)
        fly = FLY_ID[fname_t[1]] if l > 1 else 1
        x = fname_t[2] if l > 2 else None
        y = fname_t[3] if l > 3 else None
        if cb:
            cb(fname_t[1], ax, dfs, x, y)
        else:
            ret = plot_figure(ax, dfs, fly, fname_t[0], x, y, bouts, count=count)
            if need_title:
                if fname_t[0].startswith("fc_"):
                    name_post = fname_t[0][3:]
                else:
                    name_post = fname_t[-1]
                ax.set_title(REPLACE_TITLE.get(name_post, name), fontsize=FONT_SIZE)
            info_dir = info_dir or os.path.dirname(filename)
            if save_pickle:
                pf = open(os.path.join(info_dir, "%s_%s.pickle" % (f_name, name.replace(":", ","))), "wb")
                pickle.dump(to_list(ret), pf)
                pf.close()
            # save_dict("%s/info/%s_%s.txt" % (dir_name, f_name, name), to_list(ret))
        if sharex and (idx+1) % row != 0:
            ax.set_xticklabels([])
            ax.set_xlabel("")
    if need_title:
        plt.suptitle(filename + " n=%d cir_n=%d" % (n, len(bouts)), y=0.99)
    # if space < 0.5:
    #     plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_and_open(filename, save_svg)

def pass_angle(a, angle):
    angle_l = np.abs(lim_dir_a(a - angle))
    head_frame = np.argmin(angle_l)
    if angle_l[head_frame] > 10:# or (head_frame < 5 or head_frame > total_frame-5):
        return None
    return head_frame

def in_range(a, r1, r2):
    return r1 < a < r2

def valid_circle(a_l, r_l):  # NOTE: filter female jump
    aa = np.array(a_l)
    if np.max(np.abs(lim_dir_a(aa[2:] - aa[:-2]))) > 120:
        return False
    rr = np.array(r_l)
    if np.max(np.abs(rr[2:] - rr[:-2])) > 3:
        return False
    if np.min(rr) > 5.5:
        return False
    return True

def calc_track_pass_angle(dfr, dfa, bouts, a_range):
    r_l, a_l = [], []
    for s, e in bouts:
        total_frame = e - s
        rs = np.array(dfr[s:e])
        alpha = correct_angle(dfa[s:e].tolist())
        if not valid_circle(alpha, rs):
            continue
        alpha = lim_dir_a(np.array(alpha) - REL_POLAR_T_OFFSET)
        f0 = pass_angle(alpha, a_range[0])
        if f0 is None:
            continue
        f1 = pass_angle(alpha, (a_range[0] + a_range[1]) / 2)
        if f1 is None:
            continue
        f2 = pass_angle(alpha, a_range[1])
        if f2 is None:
            continue
        ss, ee = f1, f1
        for f in np.arange(f1, -1, -1):
            if not in_range(alpha[f], a_range[0], a_range[1]):
                break
            ss = f
        for f in np.arange(f1, total_frame):
            if not in_range(alpha[f], a_range[0], a_range[1]):
                break
            ee = f
        if ee - ss > 3:
            r_l.append(rs[ss:ee])
            a_l.append(alpha[ss:ee])
    return r_l, a_l

def calc_track_in_angle_range(dfr, dfa, bouts, a_range_l):
    r_ll, a_ll = [], []
    for a_range in a_range_l:
        r_ll.append([])
        a_ll.append([])
    for s, e in bouts:
        rs = np.array(dfr[s:e])
        alpha = correct_angle(dfa[s:e].tolist())
        if not valid_circle(alpha, rs):
            continue
        angle_range = np.max(alpha) - np.min(alpha)
        for i, a_range in enumerate(a_range_l):
            if in_range(angle_range, a_range[0], a_range[1]):
                r_ll[i].append(rs)
                a_ll[i].append(alpha)
    return r_ll, a_ll

def dfs_filter_track_len(dfs, bouts, min_len):
    dfr, dfa = dfs[CENTER_FLY]["rel_polar:r"], dfs[CENTER_FLY]["rel_polar:t"]
    idx = np.zeros(len(dfr), dtype=bool)
    for s, e in bouts:
        rs = np.array(dfr[s:e])
        alpha = correct_angle(dfa[s:e].tolist())
        if not valid_circle(alpha, rs):
            continue
        angle_range = np.max(alpha) - np.min(alpha)
        idx[s:e] = angle_range > min_len
    return dfs[0][idx], dfs[1][idx], dfs[2][idx]

def calc_track_stop_angle(dfr, dfa, bouts, sections, count=1e10, is_start=False):
    ret_bouts = []
    for s, e in bouts:
        rs = np.array(dfr[s:e])
        alpha = correct_angle(dfa[s:e].tolist())
        if not valid_circle(alpha, rs):
            continue
        if is_start:
            stop_angle = alpha[0]
        else:
            stop_angle = alpha[-1]
        if int(abs(stop_angle) / 45) in sections:
            ret_bouts.append((s, e))
            if len(ret_bouts) > count:
                break
    return ret_bouts

def stop_trigger_avg(dfx, dfs, bouts, frames=66, bins=8):
    dfa = np.array(dfs[CENTER_FLY]["rel_polar:t"])
    dfp = np.array(dfs[0]["pair"])
    hb = int(bins/2)
    bin_step = 360/bins
    pair_d = {}
    for s, e in bouts:
        fps = (e-s) / (dfs[0]["time"][e-1] - dfs[0]["time"][s])
        t = e - s
        if fps > 40:
            if t > frames:
                idx = np.arange(e - frames, e)
            else:
                idx = np.concatenate([np.zeros((frames - t)), np.arange(s, e)])
        else:
            continue
            f2 = int(frames/2)
            if t > f2:
                idx = np.arange(e - f2, e)
            else:
                idx = np.concatenate([np.zeros((f2 - t)), np.arange(s, e)])
            idx = np.ravel([[i, i] for i in idx])

        alpha = dfa[e-1] - REL_POLAR_T_OFFSET
        # section = int((alpha+180) / bin_step)
        section = int(abs(alpha) / bin_step)
        pair = dfp[s]
        pair_d.setdefault(pair, [[] for i in range(hb)])
        pair_d[pair][section].append(idx)
    ret = []
    xa = np.array(dfx)
    py_ll = [[] for i in range(hb)]
    for pair, iz in pair_d.items():
        for i in range(hb):
            if len(iz[i]):
                iz1 = np.array(iz[i], dtype=int)
                z = np.array([xa[zz] for zz in iz1])
                z[iz1 == 0] = np.nan
                py_ll[i].append(np.nanmean(z, axis=0))
    # NOTE: merge 1 and 2
    py_ll[1].extend(py_ll[2])
    py_ll[2] = py_ll[3]

    for py_l in py_ll[:-1]:
        py_l = np.array(py_l)
        py = np.nanmean(py_l, axis=0)
        err_l = np.nanstd(py_l, axis=0)
        if USE_SEM:
            err_l = err_l / np.sqrt(np.count_nonzero(~np.isnan(py_l), axis=0))
        ret.append([py, err_l])
    return ret

def clear_align_cache():
    global g_time_align_cache, g_angle_align_cache
    g_time_align_cache = {}
    g_angle_align_cache = {}
clear_align_cache()

def calc_time_align_pcolor(dfs, bouts, align="head", t_range=1, bins=30, random=False, sort=False):
    iz = g_time_align_cache.get(align)
    if iz is not None:
        return iz
    to_head = None
    if align == "sideu":
        to_head = True
    elif align == "sided":
        to_head = False
    dfa = dfs[CENTER_FLY]["rel_polar:t"]
    dftheta = dfs[1]["theta"]
    dft = dfs[0]["time"]
    dfp = dfs[0]["pair"]
    bin_step = t_range/bins
    iz = []
    bouts1 = bouts
    if sort:
        bouts1 = sorted(bouts, key=lambda a: a[1] - a[0], reverse=True)
    for s, e in bouts1:
        # align to head, +-10deg
        total_frame = e - s
        ts = np.array(dft[s:e])
        alpha = np.array(dfa[s:e]) - REL_POLAR_T_OFFSET
        if align == "head":
            angle_l = np.abs(lim_dir_a(alpha))
        elif align == "tail":
            angle_l = np.abs(lim_dir_a(alpha + 180))
        else:
            angle_l = np.abs(lim_dir_a(alpha + 90))
            angle_l[angle_l > 90] = 180 - angle_l[angle_l > 90]
        if random:
            head_frame = np.random.randint(total_frame)
        else:
            head_frame = np.argmin(angle_l)
            if angle_l[head_frame] > 10: #or (head_frame < 5 or head_frame > total_frame-5):
                continue
        if to_head is not None:
            rx, theta = alpha[head_frame], dftheta[s + head_frame]
            is_to_tail = ((rx < 0) and (theta > 0)) or ((rx > 0) and (theta < 0))
            if is_to_tail == to_head:
                continue
        # v = np.full([bins*2], np.nan)
        v = np.zeros((bins*2, ), dtype=int)
        t0 = ts[head_frame]
        for i in range(0, total_frame):
            t = ts[i] - t0
            if t > t_range:
                break
            if t < -t_range:
                continue
            bin_n = int(t / bin_step + .5)
            v[bins + bin_n - 1] = s + i
        first_nnan = -1
        for i in range(0, len(v)):
            if v[i] != 0:
                first_nnan = i
                break
        iz.append([first_nnan, v, dfp[s]])
        # print(dfp[s], dfs[0]["frame"][s], total_frame, head_frame)
    if sort:
        iz.sort(key=lambda zz: zz[0])
    # print([zz[1] for zz in iz[:10]])
    # iz = np.array([zz[1] for zz in iz])
    g_time_align_cache[align] = iz
    return iz

def bouts_aligned_group_by_pair(iz):
    # NOTE: iz[i]=[first_nnan, bouts_aligned, pair_l]
    ret = {}
    for z in iz:
        pair = z[2]
        ret.setdefault(pair, [])
        ret[pair].append(z[1])
    return list(ret.values())

def calc_angle_align_pcolor(dfs, bouts, align="head", a_range=180, bins=30, random=False):
    # NOTE: 最大旋转180度截断; switching时截断(敏感); 横轴为相对对齐点的角度; 空值用左右两边均匀插值
    # NOTE: 保证从左到右时间顺序; side对齐到右边时镜像到左边; head/side/tail合并CW和CCW; side对齐CW去头CCW去尾
    # NOTE: CW/CCW用+-2帧的alpha差值判断, 和以theta判断结果差不多
    cache = g_angle_align_cache.get(align)
    if cache is not None:
        return cache
    to_head = None
    if align == "sideu":
        to_head = True
    elif align == "sided":
        to_head = False
    is_side = align.startswith("side")
    dfa = dfs[CENTER_FLY]["rel_polar:t"]
    dftheta = dfs[1]["theta"]
    dfp = dfs[0]["pair"]
    bin_step = a_range/bins
    iz_cw, iz_ccw = [], []
    range_offset = {"head": 0, "tail": 180, "left": 90, "right": -90}
    o = range_offset.get(align, 90)
    a_start, a_end = -a_range + o, a_range + o
    for s, e in sorted(bouts, key=lambda a: a[1]-a[0], reverse=True):
        # align to head, +-10deg
        total_frame = e - s
        alpha = np.array(dfa[s:e]) - REL_POLAR_T_OFFSET
        la = lim_dir_a(alpha - o)
        angle_l = np.abs(la)
        if is_side:
            angle_l[angle_l > 90] = 180 - angle_l[angle_l > 90]
        if random:
            head_frame = np.random.randint(total_frame)
        else:
            head_frame = np.argmin(angle_l)
            if angle_l[head_frame] > 10: #or (head_frame < 5 or head_frame > total_frame-5):
                continue
        if is_side:
            if abs(la[head_frame]) > 10:
                la = lim_dir_a(180 - la)  # NOTE: mirror to left
        # is_cw = dftheta[s + head_frame] > 0
        st = 2
        cst = head_frame
        if cst + st >= total_frame:
            cst = total_frame - st - 1
        elif cst - st < 0:
            cst = st
        is_cw = lim_dir(la[cst + st] - la[cst - st]) < 0
        v = np.zeros((bins*2, ), dtype=int)

        tor = 5
        for i in range(head_frame, total_frame):
            bin_n = int(la[i] / bin_step + .5)
            if abs(bin_n) > bins:
                break
            if i > 0:
                if abs(la[i] - la[i - 1]) > 180:  # rewind
                    break
                ddir = lim_dir(la[i] - la[i-1])
                if is_cw and ddir > tor:
                    break
                if not is_cw and ddir < -tor:
                    break
            v[bins + abs(bin_n) - 1] = s + i #i-head_frame#
        for i in range(head_frame - 1, 0, -1):
            bin_n = int(la[i] / bin_step + .5)
            if abs(bin_n) > bins:
                break
            if i > 0:
                if abs(la[i] - la[i - 1]) > 180:
                    break
                ddir = lim_dir(la[i] - la[i-1])
                if is_cw and ddir > tor:
                    break
                if not is_cw and ddir < -tor:
                    break
            v[bins - abs(bin_n) - 1] = s + i
        first_nnan = -1
        ns = -1
        for i in range(0, len(v)):
            if v[i] != 0:
                if first_nnan < 0:
                    first_nnan = i
                if ns >= 0 and i - ns > 1:  # NOTE: interpolate empty bins
                    # inter = (v[ns] + v[i]) / 2
                    step = (v[i] - v[ns]) / (i - ns)
                    for j in range(ns + 1, i):
                        v[j] = v[j - 1] + step
                ns = i
        if first_nnan >= 0:
            if is_cw:
                iz_cw.append([first_nnan, v, dfp[s]])
            else:
                iz_ccw.append([first_nnan, v, dfp[s]])
    if to_head is not None:
        ret = iz_cw if to_head else iz_ccw
    else:
        iz_cw.extend(iz_ccw)
        ret = iz_cw
    ret.sort(key=lambda x: x[0])
    g_angle_align_cache[align] = ret
    return ret

WRONG_ANGLE_CHANGE_MIN = 50
def correct_angle(v):
    # return np.array(v) - v[0]
    ret = []
    if not v:
        return ret
    li = v[0]
    offset = 0
    for i in v:
        i += offset
        d = i - li
        if d > 150:
            offset -= 360
            i -= 360
        elif d < -150:
            offset += 360
            i += 360
        ret.append(i)
        li = i
    for j in range(1, len(ret) - 1):
        i1, i2, i3 = ret[j - 1], ret[j], ret[j + 1]
        if abs(i1 - i3) < WRONG_ANGLE_CHANGE_MIN < abs(i2 - i3) and abs(i2 - i1) > WRONG_ANGLE_CHANGE_MIN:
            ret[j] = (i1 + i3) / 2
    return ret

def plot_hist_by_info(ax, fig_info, need_norm=True, label=None):
    xs, range_x, bins, is_polar, color, x, line = fig_info
    if isinstance(xs, list):
        xs = pd.Series(xs)
    hist_f, hist_b = np.histogram(xs, bins=bins, range=range_x)
    # vs_l, bins_l = bin_data_polar(xs, xs, bins=bins)
    # hist_f2 = np.array([len(v) for v in vs_l[:-1]])
    if need_norm:
        hist_f = hist_f / np.sum(hist_f)
        # hist_f2 = hist_f2 / np.sum(hist_f2)
    if is_polar:
        hist_b += (hist_b[1] - hist_b[0]) / 2
        hist_f = np.append(hist_f, hist_f[0])
    else:
        hist_b = hist_b[:-1]
    if is_polar:
        hist_b[-1] = hist_b[0]
        angles = np.deg2rad(hist_b + 90)
        im = ax.bar(angles[:-1], hist_f[:-1], color=color, width=np.pi * 2 / bins, alpha=0.5)
    else:
        if line:
            im = ax.plot(hist_b, hist_f, line, color=color, label=label)
            # im = ax.plot(hist_b, hist_f2, line, color="gray", label=label)
        else:
            im = ax.bar(hist_b, hist_f, width=hist_b[1] - hist_b[0], color=color, alpha=0.5)
    if HEAT_CONFIG["plot_mean"]:
        x, y = get_mean_or_median(xs, x), np.max(hist_f) * 1.1
        x_mean = xs.mean()
        im = ax.scatter([x], [y], c="g", marker="o")
        ax.text(x, y, " %.2f" % x)
        # ax.scatter([x_mean], [y], c="r", marker="+")
        # ax.text(x_mean, y, " (%.2f)" % x_mean, alpha=0.5)
    if range_x[0] == -180 and range_x[1] == 180:
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
        ax.set_xlabel("$\\theta$", fontsize=FONT_SIZE)
    elif range_x[0] == 0 and range_x[1] == 180:
        ax.set_xticks([0, 90, 180])
        ax.set_xticklabels([0, "$\pi/2$", "$\pi$"])
        ax.set_xlabel("$|\\alpha|$", fontsize=FONT_SIZE)
    else:
        ax.set_xlabel("x", fontsize=FONT_SIZE)
    # if need_norm:
    #     ax.set_yticklabels([("%.1f%%" % (t*100)) for t in ax.get_yticks()])
    #     ax.set_ylabel("Probability", fontsize=FONT_SIZE)
    return im

def check_is_angle(x):
    return x == "dir" or x == "theta" or x.endswith(":t")

def get_mean_or_median(xs, x):
    if check_is_angle(x):
        return mean_angle(xs)
    return xs.median()

def mean_weights(xs, weights):
    if weights is None:
        return mean_value(xs)
    return np.sum(xs * weights) / len(weights)

def mean_value(xs):
    if not len(xs):
        return np.nan
    return np.nanmean(xs)

def mean_angle(angles, weights=None):
    rads = np.deg2rad(angles)
    x, y = np.cos(rads), np.sin(rads)
    return np.rad2deg(np.arctan2(mean_weights(y, weights), mean_weights(x, weights)))

def std_angle(angles, means):
    d = angle_diff_a(angles, means)
    return np.sqrt(np.nanmean(d**2))

WRONG_POS_MOVE_MIN = 2 #mm
def filter_wrong_pos(xs, ys, ds):
    # 2 tracks, use the longest one
    rxs, rys, rds = [], [], []
    xs = xs.tolist()
    ys = ys.tolist()
    lx, ly = xs[0], ys[0]
    rxs2, rys2, rds2 = [], [], []
    lx2, ly2 = 0, 0
    s_d_max = WRONG_POS_MOVE_MIN ** 2
    for x, y, d in zip(xs, ys, ds):
        if (x-lx)**2+(y-ly)**2 < s_d_max:
            rxs.append(x)
            rys.append(y)
            rds.append(d)
            lx, ly = x, y
        else:
            if rxs2:
                if (x-lx2)**2+(y-ly2)**2 < s_d_max:
                    rxs2.append(x)
                    rys2.append(y)
                    rds2.append(d)
                    lx2, ly2 = x, y
            else:
                rxs2.append(x)
                rys2.append(y)
                rds2.append(d)
                lx2, ly2 = x, y
    if len(rxs2) > len(rxs):
        rxs = rxs2
        rys = rys2
        rds = rds2
    return rxs, rys, rds

def plot_overlap(ax, xs, ys, dirs, idx):
    xs, ys, dirs = filter_wrong_pos(xs, ys, dirs)
    if HEAT_CONFIG["only_plot_arrow"]:
        j = 0
        for x, y, d in zip(xs, ys, dirs):
            if j % 10 == 0:
                rect = plt.Polygon(triangle_for_angle(d, x, y, 0.5), color="C%d" % (idx % 10), alpha=0.3, linewidth=0)
                ax.add_patch(rect)
            j += 1
    else:
        ax.plot(xs, ys, linewidth=1, color="C%d" % (idx % 10))

def plot_overlap_time(ax, xs, ys, dirs, fly, inter=10):
    xs, ys, dirs = filter_wrong_pos(xs, ys, dirs)
    a_l = np.arange(1, 0.001, -1/len(xs)) * 0.9
    color = COLOR_FOR_FLY[fly]
    if fly == 1:
        color_l = [(a, a, 1) for a in a_l]
    else:
        color_l = [(1, a, a) for a in a_l]
    alpha_l = 1 - a_l
    ax.scatter(xs, ys, linewidth=1, color=color_l, marker=".")
    for j in range(1, len(xs)):
        line = plt.Line2D(xs[j-1:j+1], ys[j-1:j+1], color=color_l[j])
        ax.add_line(line)
    j = 0
    for x, y in zip(xs, ys):
        if j % inter == 0:
            d = dirs[j] # iloc
            # rect = plt.Polygon(triangle_for_angle(d, x, y, 0.5), color=color_l[j])
            rect = plt.Polygon(triangle_for_angle(d, x, y, 0.5), color=color, alpha=alpha_l[j], linewidth=0)
            ax.add_patch(rect)
        j += 1

def points_to_line_collection(xs, ys, cmap, linewidth):
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    t = np.arange(0, len(segments))
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(t[::-1])  # t
    lc.set_linewidth(linewidth)
    return lc

def plot_overlap_time2(ax, xs, ys, dirs, fly, inter=3, need_colorbar=False, need_filter=True, cmap=None):
    if need_filter:
        xs, ys, dirs = filter_wrong_pos(xs, ys, dirs)
    else:
        xs, ys, dirs = xs.tolist(), ys.tolist(), dirs.tolist()
    if not cmap:
        cmap = "viridis" if fly == 2 else "autumn"
    ax.add_collection(points_to_line_collection(xs, ys, cmap, 2))

    if len(dirs) == 0:
        return
    verts = []
    j = 0
    for x, y in zip(xs, ys):
        if j % inter == 0:
            d = dirs[j]
            verts.append(triangle_for_angle(d, x, y, 0.5))
        j += 1
    t = np.arange(0, len(verts))
    norm = plt.Normalize(t.min(), t.max())
    pc = PolyCollection(verts, cmap=cmap, norm=norm, alpha=1)
    pc.set_array(t[::-1])
    ax.add_collection(pc)
    need_colorbar and plot_colorbar(ax, im=pc)

def plot_scatter_fly_body(scatter_path, figsize = None):
    # NOTE: se_pos, switch_pos_mv, ccw_, wc_
    plt.figure(figsize=figsize or (2.6, 2.6), constrained_layout=True, dpi=300)
    ax = plt.gca()
    ax.set_position([0, 0, 1, 1], which="both")
    ax.axis("off")
    # import cv2
    # img = cv2.imread(scatter_path)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # ax.imshow(img)
    f = open(scatter_path.replace(".png", ".pickle"), "rb")
    fly, range_xy, body_size, line_color = pickle.load(f)
    ax.set_xlim(range_xy[0])
    ax.set_ylim(range_xy[1])
    plot_fly_body(ax, fly, range_xy, body_size, line_color)
    save_and_open(scatter_path.replace(".png", ".pdf"))

def plot_scatter_by_info(ax, fig_info, body_size=None, cmap=None, line_color=None, alpha=0.5):
    xs, ys, range_x, range_y, s, fly, color = fig_info
    if body_size:
        plt.axis("equal")
        if HEAT_CONFIG.get("no_fly_body"):
            print("plot_fly_body(ax, %d, [%s, %s], %s, %s)" % (fly, range_x, range_y, body_size, line_color))
        else:
            plot_fly_body(ax, fly, [range_x, range_y], body_size, line_color)
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    if HEAT_CONFIG.get("plot_scatter_axis"):
        # ax.scatter(xs, ys, s=s, c=color)
        # ax.scatter(range_x, range_y, s=s, c=color, cmap=cmap)
        if range_x[0] == -180 and range_x[1] == 180:
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
    else:
        if HEAT_CONFIG["axis_off"]:
            ax.axis("off")
        if cmap:
            im = ax.scatter(xs, ys, s=s, c=color, cmap=cmap)
            # plot_colorbar(ax, im=im)
        else:
            ax.scatter(xs, ys, s=s, color=color, alpha=alpha, linewidths=0)  # no circles in pdf

def plot_scatter_sd(ax, fig_info, need_scatter=False, only_x=False, only_y=False):
    sdx, sdy, color = fig_info
    if only_x:
        plot_hist_by_info(ax, [sdx, (0, 4), 30, False, color, "sdx", "-"])
    elif only_y:
        plot_hist_by_info(ax, [sdy, (0, 4), 30, False, color, "sdy", "-"])
    else:
        if need_scatter:
            ax.scatter(sdx, sdy, s=0.5, color=color, alpha=0.4)
        sn = np.sqrt(len(sdx))
        sdx_m, sdx_e = np.mean(sdx), np.std(sdx) / sn
        sdy_m, sdy_e = np.mean(sdy), np.std(sdy) / sn
        ax.errorbar([sdx_m], [sdy_m], xerr=[sdx_e], yerr=[sdy_e], color=color)

def vec_angle(v):
    theta = np.arctan2(v[1], v[0])
    return np.rad2deg(theta)

def triangle_for_vector(x, y, px, py, r=4):
    return ((px+y/r)-x, (py-x/r)-y), ((px-y/r)-x, (py+x/r)-y), ((px+x), (py+y))

def triangle_for_angle(a, px, py, l):
    return triangle_for_vector(np.cos(np.deg2rad(a)) * l, np.sin(np.deg2rad(a)) * l, px, py)

def bin_data_polar(vs, angles, range_a=(-180, 180), bins=30):
    step = (range_a[1] - range_a[0]) / bins
    bin_l = np.arange(range_a[0], range_a[1] + 1, step)
    ret = []
    l = len(bin_l)
    for i in range(l):
        ret.append([])
    for v, angle in zip(vs, angles):
        bin_idx = (lim_dir(angle) - range_a[0]) / step
        if not np.isnan(bin_idx):
            bin_idx = int(bin_idx + 0.5)
            if 0 <= bin_idx < l:
                ret[bin_idx].append(v)
    ret[0].extend(ret[-1])
    ret[-1] = ret[0]
    return ret, bin_l

def bin_data(vs, ts, range_a=(-180, 180), bins=30):
    step = (range_a[1] - range_a[0]) / bins
    bin_l = np.arange(range_a[0], range_a[1] + 1, step)
    ret = []
    l = len(bin_l)
    for i in range(l):
        ret.append([])
    for v, t in zip(vs, ts):
        bin_idx = (t - range_a[0]) / step
        if not np.isnan(bin_idx):
            bin_idx = int(bin_idx)  # + 0.5 bug!
            if 0 <= bin_idx < l:
                ret[bin_idx].append(v)
    return ret[:-1], bin_l[:-1]  # for line

def im_gaussian(X):
    x = np.array(X)
    s = x.shape
    import cv2
    x2 = cv2.resize(x, (s[1]*2, s[0]*2), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(x2, (3, 3), 0)

def plot_heat_by_info(ax, X, extent, lim_heat, fly, range_xy, body_size, no_color_bar=False, cmap="jet"):
    if HEAT_CONFIG["axis_off"]:
        ax.axis("off")
        if HEAT_CONFIG["fill_fig"]:
            ax.set_position([0, 0, 1, 1], which="both")
    else:
        ax.tick_params(labelsize=FONT_SIZE)
    # ax.axis("equal")
    if HEAT_CONFIG.get("gaussian"):
        X = im_gaussian(X)
    interpolation = "nearest"#"gaussian" #
    im = ax.imshow(X, extent=extent, interpolation=interpolation, cmap=cmap, vmin=lim_heat[0], vmax=lim_heat[1], aspect="equal" if (PLOT_FLY_BODY and fly) else "auto")  # swap xy
    if not HEAT_CONFIG["axis_off"]:
        if range_xy == CENTER_RANGE:
            ax.set_xticks([-5, 0, 5])
            ax.set_yticks([-5, 0, 5])
        elif range_xy == ARENA_RANGE:
            ticks = np.arange(0, ARENA_RANGE[0][1] + 1, 5)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
    ax.set_xlim(range_xy[0])
    ax.set_ylim(range_xy[1])
    if not no_color_bar and HEAT_CONFIG["color_bar"]:
        plot_colorbar(ax, lim_heat[0], lim_heat[1], im=im)
    if range_xy == ARENA_RANGE:
        ax.invert_yaxis()
    elif range_xy == POLAR_RANGE:
        pass
    else:
        plot_fly_body(ax, fly, range_xy, body_size)

def plot_heat_diff(ax, info1, info2, fix_lim_heat=None):
    X1, extent, lim_heat, fly, range_xy, body_size = info1
    if fix_lim_heat:
        diff_heat = fix_lim_heat
    else:
        diff_heat = (lim_heat[1] - lim_heat[0]) / 5
    X2 = info2[0]
    plot_heat_by_info(ax, np.array(X1) - np.array(X2), extent, [-diff_heat, diff_heat], fly, range_xy, body_size)

def plot_heat_diff_p_value(ax, info1, info2):
    X1, extent, lim_heat, fly, range_xy, body_size, bin_data1 = info1
    bin_data2 = info2[-1]
    bins = len(bin_data1)
    p_m = []
    for i in range(bins):
        row = []
        for j in range(bins):
            d1 = bin_data1[i][j]
            d2 = bin_data2[i][j]
            if len(d1) < HEAT_CONFIG["min_bin"] or len(d2) < HEAT_CONFIG["min_bin"]:
                p = np.nan
            else:
                try:
                    s, p = mannwhitneyu(d1, d2)
                    # if p < 0.05:
                    #     p = 0#np.nan
                    # elif p < 0.1:
                    #     p = 0.5
                    # else:
                    #     p = 1
                    p = -np.log10(p)
                except:
                    p = np.nan
            row.append(p)
        p_m.append(row)
    plot_heat_by_info(ax, p_m, extent, [1, 4], fly, range_xy, body_size, cmap="Reds")

def bin_data_2d(xs, ys, weights, range_xy, bins):
    binx0, binx1 = range_xy[0]
    biny0, biny1 = range_xy[1]
    binxa = (binx1 - binx0) / bins
    binya = (biny1 - biny0) / bins
    bin_data = []  # bins*bins
    for i in range(bins):
        row = []
        for j in range(bins):
            row.append([])
        bin_data.append(row)
    if not isinstance(xs, list):
        xs = xs.tolist()
        ys = ys.tolist()
    for i, w in enumerate(weights):
        x = xs[i]
        y = ys[i]
        if binx1 > x > binx0 and biny1 > y > biny0:
            binx = int((x - binx0) / binxa)
            biny = int((y - biny0) / binya)
        # if bins > binx >= 0 and bins > biny >= 0:
            bin_data[biny][binx].append(w)
    return bin_data, np.linspace(binx0, binx1, bins), np.linspace(biny0, biny1, bins)  # flip y

def plot_stream(ax, bin_2d, binx, biny, bin_2d_len=None, lim_heat=None):
    # X, Y = np.meshgrid(binx, biny)
    # mean_bin = np.array([[mean_angle(v) for v in row] for row in bin_2d])
    len_bin = np.array([[len(v) for v in row] for row in bin_2d])
    max_len = np.max(len_bin)
    # U = np.cos(np.deg2rad(mean_bin))
    # V = np.sin(np.deg2rad(mean_bin))
    # ax.streamplot(X, Y, U, V, color=len_bin, cmap="autumn")
    ox, oy = (binx[1] - binx[0])/2, (biny[1] - biny[0])/2
    px, py, c = [], [], []
    cm = plt.get_cmap("jet")
    sm = ScalarMappable(norm=NoNorm(), cmap=cm)
    for i, row in enumerate(bin_2d):
        if i == 0 or i == len(bin_2d)-1:
            continue
        for j, v in enumerate(row):
            m, l = mean_angle(v) + 90, len(v)
            if l > HEAT_CONFIG["min_bin"]:
                x, y = binx[j], biny[i]
                xa, ya = np.cos(np.deg2rad(m)), np.sin(np.deg2rad(m))
                x1, y1 = x + xa * ox, y + ya * oy
                if bin_2d_len is None:
                    # color = l/max_len # NOTE: probability
                    len_m = np.clip(std_angle(v, m - 90), lim_heat[0], lim_heat[1])  # NOTE: vairiation
                    color = (len_m - lim_heat[0]) / (lim_heat[1] - lim_heat[0])
                else:
                    len_m = np.clip(np.median(bin_2d_len[i][j]), lim_heat[0], lim_heat[1])
                    color = (len_m - lim_heat[0]) / (lim_heat[1] - lim_heat[0])
                # line = plt.Line2D([x, x1], [y, y1], color=sm.to_rgba(color), linewidth=0.5)
                # ax.add_line(line)
                rect = plt.Polygon(triangle_for_angle(m, x, y, ox), color=sm.to_rgba(color), alpha=1, linewidth=0)
                ax.add_patch(rect)
                px.append(x)
                py.append(y)
                c.append(color)
    # ax.scatter(px, py, c=c, s=0.5, cmap="jet")
    return []

def align_heat_mean(xs, ys, range_xy):
    vs_l, bin_l = bin_data_polar(ys, xs, bins=HEAT_BINS)
    median_f = [np.median(vs) for vs in vs_l]
    step = bin_l[1] - bin_l[0]
    l = len(bin_l)
    nxs, nys = [], []
    for xi, x in enumerate(xs):
        bin_idx = (x + 180) / step
        bin_idx = int(bin_idx)
        if 0 <= bin_idx < l:
            # ys[xi] -= median_f[bin_idx]
            nxs.append(x)
            nys.append(ys[xi] - median_f[bin_idx])
    me = np.mean(median_f)
    nr = range_xy[0], [range_xy[1][0] - me, range_xy[1][1] - me]
    return nxs, nys, nr

# TS = np.arange(-180, 180, 360 / 50)
# MEDIAN = [2.6523, 2.6301, 2.6022, 2.6003, 2.6026, 2.5462, 2.5288, 2.5020, 2.5253, 2.5492, 2.5986, 2.6842, 2.7613, 2.8610, 2.9664, 3.0640, 3.2130, 3.3378, 3.4512, 3.5021, 3.6292, 3.7745, 3.8435, 3.8705, 3.8349, 3.9407, 3.8977, 3.8612, 3.8024, 3.7513, 3.6817, 3.5443, 3.3820, 3.2652, 3.1431, 3.0735, 2.9855, 2.8556, 2.7641, 2.6803, 2.6425, 2.6152, 2.5295, 2.5377, 2.5164, 2.5055, 2.5469, 2.5893, 2.6267, 2.6458]
# HALF_PEAK_R_LEFT = [2.1,2.1,2.1,2.1,1.95,1.95,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.95,2.25,2.25,2.4,2.4,2.55,2.55,2.7,2.85,3.0,2.85,3.0,3.0,3.0,3.0,2.85,2.7,2.7,2.4,2.4,2.25,2.1,2.1,1.95,1.95,1.8,1.8,1.8,1.65,1.8,1.8,1.8,1.95,1.95,2.1,2.1,2.1]
# HALF_PEAK_R_RIGHT = [2.85,2.85,2.7,3.0,2.85,2.85,3.0,2.85,3.0,2.85,3.15,3.45,3.45,3.45,3.45,3.9,3.9,4.2,4.35,4.2,4.5,4.5,4.65,4.65,4.95,4.5,4.65,4.5,4.65,4.35,4.2,4.2,3.75,4.05,3.9,3.75,3.45,3.3,3.3,3.15,3.3,3.0,2.85,2.85,2.85,2.7,2.85,2.85,2.85,3.0]

def plot_heat(ax, xs, ys, fly, range_xy, lim_heat=None, body_size=None, count=None, is_3d=False, show_range=None):
    count = count or len(xs)
    if not count:
        return None
    if ALIGN_HEAT_MEAN:
        xs, ys, range_xy = align_heat_mean(xs, ys, range_xy)
    weights = np.full([len(xs), ], 1/count)
    hist_r, hist_bx, hist_by = np.histogram2d(xs, ys, bins=HEAT_BINS, range=range_xy, weights=weights)
    if lim_heat:
        hist_r = np.clip(hist_r, lim_heat[0], lim_heat[1])
    else:
        lim_heat = [None, None]
    hist_r[hist_r <= HEAT_CONFIG["min_bin"]/count] = np.nan if HEAT_BLANK else 0
    extent = [hist_bx[0], hist_bx[-1], hist_by[-1], hist_by[0]]
    # extent = [hist_bx[0], hist_by[0], hist_bx[-1], hist_by[-1]]
    X = hist_r.T.tolist()
    hist_f = None
    if not ONLY_CALC_PLOT:
        if is_3d:
            mx, my = np.meshgrid(hist_bx[:-1], hist_by[:-1])
            ax.plot_surface(mx, my, np.array(X), cmap="jet", linewidth=0, vmax=lim_heat[1] * 0.46)
            # ax.set_xlabel("x", fontsize=FONT_SIZE)
            # ax.set_ylabel("y", fontsize=FONT_SIZE)
            # ax.set_zlabel("Probability", fontsize=FONT_SIZE)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.set_xticks([-5, 0, 5])
            ax.set_yticks([-5, 0, 5])
            ax.set_zticks([0, 0.002, 0.004])
            # ax.set_zticklabels(["0%", "0.2%", "0.4%"])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.view_init(elev=30, azim=155)
        else:
            plot_heat_by_info(ax, X, extent, lim_heat, fly, show_range or range_xy, body_size)
            if PLOT_HEAT_MEAN:
                vs_l, bin_l = bin_data_polar(ys, xs, bins=HEAT_BINS)
                hist_f = [np.median(vs) for vs in vs_l]
                ax.plot(bin_l, hist_f, color="b", linewidth=2)
            if PLOT_HEAT_HALF:
                median_info = get_median_info()
                if median_info:
                    bin_l = median_info["alpha"]
                    ax.plot(bin_l, median_info["median"], color="b", linewidth=2)
                    ax.plot(bin_l, median_info["half_peak_left"], color="gray", linewidth=2)
                    ax.plot(bin_l, median_info["half_peak_right"], color="gray", linewidth=2)
            if PLOT_HEAT_ACCUM:
                vs_l, bin_l = bin_data_polar(ys, xs, bins=HEAT_BINS)
                hist_f1 = np.array([len(vs) for vs in vs_l])
                hist_f1 = hist_f1 / np.sum(hist_f1) * 100
                ax.plot(bin_l, hist_f1, color="r", linewidth=2)

    return X, extent, lim_heat, fly, range_xy, body_size, hist_f, xs, ys

def plot_weight_heat(ax, xs, ys, fly, range_xy, lim_heat=None, body_size=None, weights=None):
    # body_size = [BODY_WID[fly], BODY_LEN[fly]]
    if weights is not None:
        valid_idx = ~np.isnan(weights)
        xs = xs[valid_idx]
        ys = ys[valid_idx]
        weights = weights[valid_idx]
    if ALIGN_HEAT_MEAN:
        xs, ys, range_xy = align_heat_mean(xs, ys, range_xy)
    hist_f, hist_bx, hist_by = np.histogram2d(xs, ys, bins=HEAT_BINS, range=range_xy, weights=weights)
    hist_f0, hist_bx0, hist_by0 = np.histogram2d(xs, ys, bins=HEAT_BINS, range=range_xy)
    hist_f0[hist_f0 <= HEAT_CONFIG["min_bin"]] = np.nan  # empty bins
    if HEAT_CONFIG.get("fc_contact"):
        # H = hist_f.T[::-1,:] # for fc_contact_value
        # row, col = H.shape
        # ax.imshow(H, aspect="auto")
        # for i in range(row):
        #     for j in range(col):
        #         ax.text(j, i, str(int(H[i][j])), ha="center", va="center", color="w")
        # ax.set_position([0, 0, 1, 1], which="both")
        # return None
        hist_f0[hist_f <= 2] = np.nan  # for fc_contact !!!
    hist_r = hist_f / hist_f0
    if lim_heat:
        hist_r = np.clip(hist_r, lim_heat[0], lim_heat[1])
    else:
        lim_heat = [None, None]
    hist_r[hist_r == 0] = np.nan
    extent = [hist_bx[0], hist_bx[-1], hist_by[-1], hist_by[0]]
    X = hist_r.T.tolist()
    if not ONLY_CALC_PLOT:
        plot_heat_by_info(ax, X, extent, lim_heat, fly, range_xy, body_size)
        if PLOT_HEAT_MEAN:
            vs_l, bin_l = bin_data_polar(ys, xs, bins=HEAT_BINS)
            hist_f = [np.median(vs) for vs in vs_l]
            ax.plot(bin_l, hist_f, color="b", linewidth=2)
        if PLOT_HEAT_HALF:
            median_info = get_median_info()
            if median_info:
                bin_l = median_info["alpha"]
                ax.plot(bin_l, median_info["median"], color="b", linewidth=2)
                ax.plot(bin_l, median_info["half_peak_left"], color="gray", linewidth=2)
                ax.plot(bin_l, median_info["half_peak_right"], color="gray", linewidth=2)
    return X, extent, lim_heat, fly, range_xy, body_size, xs, ys, weights #bin_data_2d(xs.tolist(), ys.tolist(), weights.tolist(), range_xy, HEAT_BINS)[0]

g_colorbar = {}
def plot_colorbar(ax, vmin=None, vmax=None, im=None):
    global g_colorbar
    if g_colorbar.get(ax):
        g_colorbar[ax].remove()
    g_colorbar[ax] = plt.colorbar(im, ax=ax, shrink=0.3, aspect=10)
    cax = g_colorbar[ax].ax
    yax = cax.yaxis
    if vmin is not None:
        yax.set_ticks([vmin, vmax])
        if 0 <= vmin and vmax <= 1:
            yax.set_ticklabels(["%.1f%%" % (vmin*100), "%.1f%%" % (vmax*100)])
        elif vmax <= 30:
            yax.set_ticklabels([str(vmin), str(vmax)])
            g_colorbar[ax].set_label("mm/s", fontsize=FONT_SIZE)
        else:
            yax.set_ticklabels([str(vmin) + "°", str(vmax) + "°"])
    for l in yax.get_ticklabels():
        l.set_fontsize(FONT_SIZE)

def plot_fly_body(ax, fly, range_xy, body_size, line_color="w", lw=2, zo=None):
    # if HEAT_CONFIG["body_line"] <= 0:
    #     return
    # if HEAT_CONFIG["body_line"] > 0:
    #     HEAT_CONFIG["body_line"] -= 1
    zorder = zo#10 if line_color == "k" else None
    #range_xy[0][0]/4
    ax.add_line(plt.Line2D((range_xy[0][0], range_xy[0][1]), (0, 0), linewidth=2, color=line_color, linestyle="--", zorder=zorder))
    ax.add_line(plt.Line2D((0, 0), (range_xy[1][0], range_xy[1][1]), linewidth=2, color=line_color, linestyle="--", zorder=zorder))
    if not fly or not PLOT_FLY_BODY:
        return
    if HEAT_CONFIG["fly_body_ellipse"]:
        body_len, body_sh = body_size
        ell1 = patches.Ellipse(xy=(0, 0), width=body_sh, height=body_len, facecolor=COLOR_FOR_FLY[fly], alpha=HEAT_CONFIG["fly_body_alpha"], edgecolor="w", linewidth=lw)
        ax.add_patch(ell1)
    else:
        body_len = body_size[0]
        body_sh = 0.4 * body_len #body_size[1]
        ell1 = patches.Ellipse(xy=(0, 0), width=body_sh, height=body_len, facecolor=COLOR_FOR_FLY[fly], alpha=HEAT_CONFIG["fly_body_alpha"], edgecolor="w", linewidth=lw)
        ell2 = patches.Ellipse(xy=(0, body_len / 2 - body_sh * 0.3), width=body_sh, height=body_sh * 0.8, facecolor=COLOR_FOR_FLY[fly], alpha=HEAT_CONFIG["fly_body_alpha"], edgecolor="w", linewidth=lw)
        ax.add_patch(ell1)
        ax.add_patch(ell2)

def plot_inner_limit(ax, offset=0):
    a = (DIST_TO_FLY_INNER_H + offset) * 2
    ax.add_patch(patches.Arc((0, 0), a, a, 0, 45, 135, linewidth=0.5, facecolor=None, edgecolor="k", linestyle="--"))
    a = (DIST_TO_FLY_INNER_T + offset) * 2
    ax.add_patch(patches.Arc((0, 0), a, a, 0, 135, 405, linewidth=0.5, facecolor=None, edgecolor="k", linestyle="--"))

def parse_name(name):
    fname_t = name.split("-")
    l = len(fname_t)
    figname = fname_t[0]
    fly = FLY_ID[fname_t[1]] if l > 1 else 1
    x = fname_t[2] if l > 2 else None
    y = fname_t[3] if l > 3 else None
    return figname, fly, x, y

# plot_by_name(dfs, "img/R76XShi-weight_heat-male-v_len")
# plot_by_name(dfs, "img/R76XShi-relation-male-theta-v_len")
def plot_by_name(dfs, path, polar=False, figsize=None, bouts=[], save_pickle=False, ax=None, color=None, is_3d=False, save_svg=True, count=None, cir_meta=None, lim_heat=None):
    fname = os.path.basename(path).replace(",", ":")
    fname_t = fname.split("-")
    save = False
    if not ax:
        fig = plt.figure(fname, figsize=figsize or (2.6, 2.6), constrained_layout=True, dpi=300)
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.gca(projection="3d")
        else:
            ax = plt.subplot(polar=polar)
        save = True
    l = len(fname_t)
    fly = FLY_ID[fname_t[2]] if l > 2 else 1
    x = fname_t[3] if l > 3 else None
    y = fname_t[4] if l > 4 else None
    fig_info = plot_figure(ax, dfs, fly, fname_t[1], x=x, y=y, bouts=bouts, color=color, count=count, cir_meta=cir_meta, lim_heat=lim_heat)
    if not polar:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if save:
        save_and_open(path, save_svg)
        if save_pickle:
            pf = open(path + ".pickle", "wb")
            pickle.dump(fig_info, pf)
            pf.close()

def plot_multi_info(fm, prefix="", out_path="multi.png", col=3):
    if isinstance(fm[0], list):
        col = len(fm[0])
        fm = np.concatenate(fm)
    count = len(fm)
    col = min(col, count)
    row = int(count / col)
    if row * col < count:
        row += 1
    FIGURE_W = 12
    fz = (col * FIGURE_W, row * FIGURE_W)
    base = os.path.basename(out_path)
    if base.startswith("polar_") or base.startswith("we-polar_"):
        fig, axes = plt.subplots(row, col, figsize=fz, subplot_kw={"projection": "polar"})
    else:
        fig, axes = plt.subplots(row, col, figsize=fz)
    axes = axes.flatten()
    # plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0, wspace=0)
    plt.subplots_adjust(left=0.0, right=0.99, top=0.95, bottom=0.0, hspace=-0.1, wspace=-0.23)
    ret = False
    for i, fi in enumerate(fm):
        title = fi.split("-")[0]
        title = title.split("/")[-1]
        bc = "w"
        ax = axes[i]
        info = load_fig_info(os.path.join(prefix, fi))
        if info:
            try:
                ret = plot_fig_info(info, ax=ax)
            except:
                ret = None
            if not ret:
                print("error")
                break
            if isinstance(info[1], int):
                title += " (%d)" % info[1]
                if info[1] > 600:
                    bc = "y"
        ax.set_title(title, fontsize=FONT_SIZE*2, color="k", backgroundcolor=bc)
    # plt.tight_layout()
    try:
        ret and save_and_open(out_path, False)
    except:
        pass
    plt.close("all")
    print("saved: " + str(fm), out_path)

def load_fig_info(f):
    if not os.path.exists(f):
        return None
    if f.endswith(".txt"):
        return load_dict(f)
    pf = open(f, "rb")
    ret = pickle.load(pf)
    pf.close()
    return ret

def plot_by_info(f):
    plt.figure(figsize=(8, 8), constrained_layout=True)
    plot_fig_info(load_fig_info(f), plt.gca())
    plt.savefig(f + ".png")

def plot_fig_info(fig_info_l, ax, no_color_bar=True, legend=None):
    if not fig_info_l:
        return False
    if ax.spines.get('top'):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE)
    if isinstance(fig_info_l[0], str):
        fig_info_l = [fig_info_l]
    for i, fig_info in enumerate(fig_info_l):
        fig_type = fig_info[0]
        fig_data = fig_info[-1]
        if fig_type == "scatter":
            plot_scatter_by_info(ax, fig_data)
        elif fig_type == "scatter_sd":
            plot_scatter_sd(ax, fig_data)
        elif fig_type == "polar_adist" or fig_type == "polar_adist_h" or fig_type == "adist":
            #[bin_l, hist_f, line, color, lim_heat, sd_f]
            # plot_adist_by_info(ax, fig_data)
            plot_adist(*fig_data, ax=ax, is_polar=fig_type.startswith("polar"))
        elif fig_type == "adist_diff":
            pval_c = pval_to_color(fig_info[-2])
            bin_l, hist_f, line, color, lim_heat, sd_f = fig_data
            ax.scatter(bin_l, hist_f, c=pval_c, cmap="Reds", marker="o", s=30)
            plot_adist(bin_l, hist_f, line, color, [0, 0.8], ax=ax)
        elif fig_type == "heat" or fig_type == "weight_heat":
            plot_heat_by_info(ax, *fig_data[:6], no_color_bar=no_color_bar)
            #test
            # X, extent, lim_heat, fly, range_xy, body_size, xs, ys, weights = fig_data
            # plot_weight_heat(ax, xs, ys, fly, range_xy, [-50, 50], body_size, weights)
        elif fig_type == "hist" or fig_type == "histo":
            plot_hist_by_info(ax, fig_data)
        elif fig_type == "line":
            if legend:
                plot_line_by_info(ax, fig_data, label=legend[i])
            else:
                plot_line_by_info(ax, fig_data)
        elif fig_type == "line_cmap":
            plot_line_with_cmap(ax, fig_data)
        else:
            print("unknown type: " + fig_type)
            return False
    return True

def save_and_open(path, save_svg=True):
    if SHOW:
        return plt.show()

    # os.path.exists(path) and os.remove(path)
    if not path.endswith(FIGURE_FILE_EXTENSION):
        path = path + FIGURE_FILE_EXTENSION
    if not save_svg and FIGURE_FILE_EXTENSION == ".pdf":
        path = path.replace(".pdf", ".png")
    plt.savefig(path)
    if SAVE_SVG and save_svg and not path.endswith(FIGURE_VEC_EXTENSION):
        plt.savefig(path.replace(FIGURE_FILE_EXTENSION, FIGURE_VEC_EXTENSION))
    print("save_and_open", path)
    plt.close()
    # os.startfile(os.path.abspath(path))

ETHOGRAM_TIME_RANGE = 3600
def plot_ethogram_by_data(names, cir_b, cop_b=None, xlim=ETHOGRAM_TIME_RANGE, ax=None, hist_bins=60): # TODO: ax.eventplot
    count = len(names)
    h = 4 if count == 0 else (count * ((4/count) if count < 15 else 0.23))
    need_adjust = False
    if not ax:
        plt.figure("ethogram_%d" % count, figsize=(32, h))
        plt.clf()
        ax = plt.gca()
        need_adjust = True
    # ax.tick_params(labelsize=FONT_SIZE)
    ax.set_ylim(0, count)
    ax.set_xlim(0, xlim)
    bin_step = xlim/hist_bins
    cir_count = 0
    hist_l = []
    for y, data in enumerate(cir_b):
        hist = [0] * hist_bins
        cir_dur = 0
        for d in data:
            r0, r1 = d  #int(d[0]) / fps, int(d[1]) / fps
            cir_dur += r1 - r0
            rect = patches.Rectangle((r0, y), max(r1 - r0, 0.1), 1, color="b", linewidth=0)
            ax.add_patch(rect)
            bin0, bin1 = int(r0/bin_step), int(r1/bin_step)
            # for b in range(bin0, bin1+1):
            #     hist[b] += 1
            if bin0 < hist_bins:
                hist[bin0] += 1
        hist_l.append(hist)
        # names[y] = pair_name_str(names[y])
        cir_count += len(data)
    if cop_b:
        for y, data in enumerate(cop_b):
            cir_dur = 0
            for d in data:
                r0, r1 = d
                cir_dur += r1 - r0
                rect = patches.Rectangle((r0, y), max(r1 - r0, 0.1), 1, color="r", linewidth=0)
                ax.add_patch(rect)
    ax.set_yticks(np.arange(len(names)) + 0.5)
    ax.set_yticklabels(names)
    if need_adjust:
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
        ax.set_ylabel("Pairs (n=%d cir=%d)" % (count, cir_count), fontsize=FONT_SIZE)
        # plt.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.2)
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        except:
            pass
    return np.arange(0, xlim, bin_step), hist_l

def plot_shade(ax, x_lim, alpha=0.1):
    yticks = ax.get_yticks()
    max_y = yticks[-1]
    ax.add_patch(plt.Rectangle((x_lim[0], 0), x_lim[1] - x_lim[0], max_y, alpha=alpha, color="k", linewidth=0))

def plot_adist(bin_l, hist_f, line, color, lim_heat, sd_f=None, ax=None, is_polar=False):
    bin_l1 = bin_l
    if is_polar:
        bin_l1 = np.deg2rad(np.array(bin_l) + (90 - REL_POLAR_T_OFFSET))
    ax.plot(bin_l1, hist_f, line, c=color)
    if sd_f is not None:
        # ax.errorbar(bin_l1, hist_f, sd_f, ecolor=color, color=color)
        am = np.array(hist_f)
        asd = np.array(sd_f)
        am_asd = am - asd
        if is_polar:
            am_asd[am_asd < 0] = 0
        ax.fill_between(bin_l1, am_asd, am + asd, facecolor=color, alpha=.3)
    ax.set_ylim(lim_heat)
    if is_polar:
        grids = [0, 90, 180, 270]
        labels = ["$-\pi/2$", 0, "$\pi/2$", "$\pi$"]
        plt.thetagrids(grids, labels)
    else:
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
        ax.set_xlabel("$\\alpha$", fontsize=FONT_SIZE)
        # ax.grid()
    return [bin_l, hist_f, line, color, lim_heat, sd_f]

def plot_adist_band(vs_l, bin_l, color, lim_heat, ax):
    xsh, ysh = [], []
    for b, vs in zip(bin_l, vs_l):
        for v in vs:
            xsh.append(b)
            ysh.append(v)
    sns.lineplot(xsh, ysh, estimator="median", ci="sd", color=color)
    ax.set_ylim(lim_heat)
    ax.set_xlim([-180, 180])
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])

def plot_adist_by_info(ax, fig_info):
    xs, ys, bins, is_polar, color, lim_heat = fig_info
    vs_l, bin_l = bin_data_polar(ys, xs, bins=bins)
    # plot_adist_band(vs_l, bin_l, color, lim_heat, ax)
    hist_f = [np.nanmedian(vs) for vs in vs_l]
    if USE_SEM:
        sd_f = [(np.nanstd(vs) / np.sqrt(len(vs))) for vs in vs_l]
    else:
        sd_f = [np.nanstd(vs) for vs in vs_l]
    return plot_adist(bin_l, hist_f, "-", color, lim_heat, sd_f, ax, is_polar=is_polar)
    # return fig_info

def plot_adist_by_info_n(ax, fig_info):
    xs_l, ys_l, bins, is_polar, color, lim_heat = fig_info
    hist_f_l = []
    estimator = np.nanmedian
    # if lim_heat[1] == 0.1: #??
    #     estimator = np.nanmean
    for xs, ys in zip(xs_l, ys_l):
        vs_l, bin_l = bin_data_polar(ys, xs, bins=bins)
        hist_f_l.append([estimator(vs) for vs in vs_l])
    hist_f = estimator(hist_f_l, axis=0)
    sd_f = np.nanstd(hist_f_l, axis=0)
    if USE_SEM:
        sd_f = sd_f / np.sqrt(np.count_nonzero(~np.isnan(hist_f_l), axis=0))
    return plot_adist(bin_l, hist_f, "-", color, lim_heat, sd_f, ax, is_polar=is_polar)

def print_heat_xy_video(xs, ys, dfs, xy, xy_range=CENTER_RANGE, bins=HEAT_BINS, weights=None, postfix=""):
    x_range = (xy_range[0][1] - xy_range[0][0]) / bins
    y_range = (xy_range[1][1] - xy_range[1][0]) / bins
    idx = (np.fabs(xs - xy[0]) < x_range) & (np.fabs(ys - xy[1]) < y_range)
    if weights is not None:
        dfs[0]["weights"] = weights
        ret = dfs[0][idx][["pair", "frame", "weights"]]
    else:
        ret = dfs[0][idx][["pair", "frame"]]
    print("heat_xy%s: %d" % (postfix, len(ret)))
    ret.to_csv("img/heat_xy%s.csv" % postfix, index=False)

def split_pair_d(pair_l, xs):
    xs_d = {}
    lp = pair_l.iloc[0]
    s = 0
    for i, pair in enumerate(pair_l):
        if pair != lp:
            xs_d[lp] = xs[s:i]
            s = i
        lp = pair
    xs_d[lp] = xs[s:]
    return xs_d

def split_pair1(pair_l, xs):
    xs_l = []
    lp = pair_l.iloc[0]
    s = 0
    for i, pair in enumerate(pair_l):
        if pair != lp:
            xs_l.append(xs[s:i])
            s = i
        lp = pair
    xs_l.append(xs[s:])
    return xs_l

def split_pair2(pair_l, xs, ys):
    xs_l, ys_l = [], []
    lp = pair_l.iloc[0]
    s = 0
    for i, pair in enumerate(pair_l):
        if pair != lp:
            xs_l.append(xs[s:i])
            ys_l.append(ys[s:i])
            s = i
        lp = pair
    xs_l.append(xs[s:])
    ys_l.append(ys[s:])
    return xs_l, ys_l

def split_pair_dfs(dfs, bouts):
    pair_l = dfs[0]["pair"]
    dfs_bouts_l = []
    lp = pair_l.iloc[0]
    s = 0
    for i, pair in enumerate(pair_l):
        if pair != lp:
            dfs_bouts_l.append([[dfs[0][s:i], dfs[1][s:i], dfs[2][s:i]],
                               [b for b in bouts if (s <= b[0] < i)]])
            s = i
        lp = pair
    dfs_bouts_l.append([[dfs[0][s:], dfs[1][s:], dfs[2][s:]],
                       [b for b in bouts if (s <= b[0])]])
    dfs_bouts_l.sort(key=lambda d: -len(d[0][0]))
    dfs_l = [d[0] for d in dfs_bouts_l]
    bouts_l = [squeeze_cir_bouts(d[1]) for d in dfs_bouts_l]
    return dfs_l, bouts_l

def get_m5_pairs(pair_swarm_csv):
    df = pd.read_csv(pair_swarm_csv)
    return df[df["cir_count"] <= 5]["pair"]

def get_body_size(dfs, center_fly):
    if HEAT_CONFIG["fly_body_ellipse"]:
        return dfs[center_fly]["e_maj"].mean(), dfs[center_fly]["e_min"].mean()
    return BODY_SIZE[center_fly]

# plot_figure(ax, dfs, 1, "c_track", "pos:x", "pos:y")
def plot_figure(ax, dfs, fly, name, x=None, y=None, bouts=[], color=None, lim_heat=None, count=None, limx=None, limy=None, bins=None, cir_meta=None):
    if not len(dfs[1]):
        return
    color = COLOR_FOR_FLY.get(fly, "k") if not color else color
    df = dfs[fly]
    fig_info = None
    dfx, dfy = None, None
    xo, yo = x, y
    if fly > 0:
        if x is not None:
            ax.set_xlabel(REPLACE_LABEL.get(x, x), fontsize=FONT_SIZE)
            x, dfx = k_with_dfk(x, df)
        if y is not None:
            ax.set_ylabel(REPLACE_LABEL.get(y, y), fontsize=FONT_SIZE)
            y, dfy = k_with_dfk(y, df)
    else:
        if x is not None:
            dfx = df[x]
        if y is not None:
            dfy = df[y]

    ax.tick_params(labelsize=FONT_SIZE)
    if name == "heat" or name == "heat_3d":
        # (ax, dfs, 1, heat, "pos:x", "pos:y")
        lim_heat = lim_heat or HEAT_LIMIT[fly].get(x)
        if limx and limy:
            show_range = [limx, limy]
        else:
            show_range = None
        dfx2 = dfx
        if x.startswith("rel_pos"):
            xy_range = CENTER_RANGE
        elif x.startswith("rel_polar"):
            dfx2 = dfx - REL_POLAR_T_OFFSET
            xy_range = POLAR_RANGE
        else:
            xy_range = ARENA_RANGE
        if PRINT_HEAT_XY_VIDEO:
            print_heat_xy_video(dfx2, dfy, dfs, PRINT_HEAT_XY_VIDEO, xy_range, HEAT_BINS, weights=dfs[1]["theta"])
        fig_info = plot_heat(ax, dfx2, dfy, fly, xy_range, lim_heat, get_body_size(dfs, fly), count=count, is_3d=(name=="heat_3d"), show_range=show_range)
    elif name == "weight_heat":
        if HEAT_CONFIG["female_fast"] == 1:
            ids = dfs[2]["v_len"] >= 2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]
            dfx = dfx[ids]
        elif HEAT_CONFIG["female_fast"] == 0:
            ids = dfs[2]["v_len"] < 2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]
            dfx = dfx[ids]

        xs, ys = dfs[CENTER_FLY][WEIGHT_HEAT_X], dfs[CENTER_FLY][WEIGHT_HEAT_Y]
        # xs, ys = dfs[CENTER_FLY]["rel_pos:x"], dfs[CENTER_FLY]["rel_pos:y"]
        weights = dfx
        lim_heat = lim_heat or HEAT_LIMIT[fly].get(x)
        if PRINT_HEAT_XY_VIDEO:
            print_heat_xy_video(xs, ys, dfs, PRINT_HEAT_XY_VIDEO, CENTER_RANGE, HEAT_BINS, weights=weights)
        if WEIGHT_HEAT_X.startswith("rel_polar"):
            fig_info = plot_weight_heat(ax, xs - REL_POLAR_T_OFFSET, ys, CENTER_FLY, POLAR_RANGE, lim_heat, None, weights=weights)
        else:
            fig_info = plot_weight_heat(ax, xs, ys, CENTER_FLY, CENTER_RANGE, lim_heat, get_body_size(dfs, CENTER_FLY), weights=weights)
    elif name.startswith("polar_adist") or name == "adist":
        # (ax, dfs, 1, polar_adist, "v_len")
        ts = dfs[2]["rel_polar:t"]
        weights = dfx
        if name == "adist":
            ts = np.array(ts - REL_POLAR_T_OFFSET)
            ax.set_ylabel(REPLACE_LABEL.get(xo, xo), fontsize=FONT_SIZE)
        else:
            if name == "polar_adist_h":
                ts = dfs[2]["rel_polar_h:t"]
            ts = np.array(ts)
        lim_heat = lim_heat or HEAT_LIMIT[fly].get(x)
        if USE_MEAN_PAIR:
            ts, weights = split_pair2(dfs[0]["pair"], ts, weights)
            fig_info = plot_adist_by_info_n(ax, [ts, weights, 60, name.startswith("polar_adist"), color, lim_heat])
        else:
            fig_info = plot_adist_by_info(ax, [ts, weights, 60, name.startswith("polar_adist"), color, lim_heat])
    elif name == "track_overlap":
        # (ax, dfs, 1, track_overlap, "pos:x", "pos:y", bouts)
        # (ax, dfs, 1, track_overlap, "rel_pos:x", "rel_pos:y", bouts)
        ax.axis("off")
        # ax.axis("equal")
        x1, x2 = CENTER_OVERLAP_RANGE_X if x.startswith("rel") else ARENA_RANGE_X
        ax.set_xlim(x1, x2)
        ax.set_xticks(np.arange(x1, x2 + 1, 5))
        ax.set_ylim(x1, x2)
        ax.set_yticks(np.arange(x1, x2 + 1, 5))
        if len(bouts) == 1 and not PLOT_VIDEO:
            s, e = bouts[0]
            bout = df.loc[s:e-1]
            bout2 = dfs[3-fly].loc[s:e-1]
            if x.startswith("rel"):
                rect = plt.Polygon(triangle_for_angle(90, 0, 0, 0.5), color=COLOR_FOR_FLY[fly], alpha=1, linewidth=0)
                ax.add_patch(rect)
                plot_overlap_time(ax, bout[x], bout[y], bout2["dir"] - bout["dir"] + 90, 3-fly)
            else:
                plot_overlap_time(ax, bout[x], bout[y], bout["dir"], fly)
                plot_overlap_time(ax, bout2[x], bout2[y], bout2["dir"], 3-fly)
        else:
            if x.startswith("rel"):
                rect = plt.Polygon(triangle_for_angle(90, 0, 0, 0.5), color=COLOR_FOR_FLY[fly], alpha=1, linewidth=0)
                ax.add_patch(rect)
                i = 0
                for s, e in bouts:
                    if e > s:
                        bout = df.loc[s:e-1]
                        bout2 = dfs[3-fly].loc[s:e-1]
                        plot_overlap(ax, bout[x], bout[y], bout2["dir"] - bout["dir"] + 90, s)
                        i += 1
            else:
                i = 0
                for s, e in bouts:
                    if e > s:
                        bout = df.loc[s:e-1]
                        plot_overlap(ax, bout[x], bout[y], bout["dir"], s)
                        i += 1
    elif name == "scatter":
        # start_end (ax, dfs, 1, scatter, rel_pos:x, rel_pos:y) TODO: plot2
        # relation (ax, dfs, 1, scatter, v_len, theta)
        # we_choice: dfs = dfs_only_we(dfs, "we_l")
        # we_choice_head: (ax, dfs, 1, scatter, rel_pos_h:x, rel_pos_h:y)
        # we_choice_tail: (ax, dfs, 1, scatter, rel_pos_t:x, rel_pos_t:y)
        # we_choice_center: (ax, dfs, 1, scatter, rel_pos:x, rel_pos:y)
        # co = np.count_non_zero(df["we_ipsi"]==-1)
        # ip = np.count_non_zero(df["we_ipsi"]==1)
        # wec = (ip-co)/(ip+co)
        fig_info1 = [dfx, dfy, limx or STAT_LIMIT[fly].get(x), limy or STAT_LIMIT[fly].get(y), STAT_POINT_SIZE.get(x, 0.1), fly, color]
        plot_scatter_by_info(ax, fig_info1, alpha=STAT_POINT_SIZE.get(x, 0.5))
    elif name == "hist" or name == "histo" or name == "polar_hist":
        # dist TODO: plot2
        # polar_dist
        # polar_start_end_dist
        is_polar = name.startswith("polar")
        range_x = limx or STAT_LIMIT[fly].get(x)
        if not bins:
            bins = 30 if is_polar else 90
        range_x and ax.set_xlim(range_x)
        line = "-" if name == "histo" else None
        fig_info = [dfx, range_x, bins, is_polar, color, x, line]
        plot_hist_by_info(ax, fig_info)
    elif name == "time":
        if check_is_angle(x):
            xs = correct_angle(dfx)
            m1, m2 = np.min(xs), np.max(dfx)
            ax.set_ylim(m1 - 5, max(m2, m1 + 180))
        ax.plot(dfx, "o-", c=color)
    elif name == "time_bin":
        ts = dfs[0]["time"]
        video_len = ts.max()#3600

        ts1, dfx1 = split_pair2(dfs[0]["pair"], ts, dfx)
        py_l = []
        for ts2, dfx2 in zip(ts1, dfx1):
            vs_l, bin_l = bin_data(dfx2, ts2, [0, video_len], 30)
            if xo == "1":
                py = [len(vs) for vs in vs_l]
            else:
                py = [np.nanmean(vs) for vs in vs_l]
            py_l.append(py)
        py1, err_l1 = lines_to_err_band(py_l)
        name, fig_info = "line", [bin_l, py1, err_l1, [0, video_len], lim_heat or HEAT_LIMIT[fly].get(x), "k", "-", None]
        plot_line_by_info(ax, fig_info)

        vs_l, bin_l = bin_data(dfx, ts, [0, video_len], 30)
        if xo == "1":
            py = [len(vs)/len(ts1) for vs in vs_l]
        else:
            py = [np.mean(vs) for vs in vs_l]
        err_l = None
        name1, fig_info1 = "line", [bin_l, py, err_l, [0, video_len], lim_heat or HEAT_LIMIT[fly].get(x), "r", "-", None]
        plot_line_by_info(ax, fig_info1)

        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
        ax.set_ylabel(REPLACE_LABEL.get(xo, xo), fontsize=FONT_SIZE)
    elif name == "switch_mv":
        step_time = 0.2
        for i, mv in enumerate([0, 2, 4, 6, 8]):
            all_switch_alpha, txs, tys, tvs, ts = get_switch_info(dfs, bouts, step_time, mv)
            a = (i + 1) / 5
            plot_hist_by_info(ax, [lim_dir_a(np.array(all_switch_alpha)), (-180, 180), 18, False, (0, 0, 1, a), x, "o-"],
                              False, label="v > %dmm/s" % mv)
        ax.set_ylabel("Frequency", fontsize=FONT_SIZE)
        ax.set_xlabel("$\\alpha$", fontsize=FONT_SIZE)
        # ax.legend(loc="upper right", fontsize=FONT_SIZE, frameon=False)
    elif name == "switch_step":
        mv = 2
        for i, step_time in enumerate([0.08, 0.14, 0.20, 0.26, 0.32]):
            all_switch_alpha, txs, tys, tvs, ts = get_switch_info(dfs, bouts, step_time, mv)
            a = (i + 1) / 5
            plot_hist_by_info(ax, [lim_dir_a(np.array(all_switch_alpha)), (-180, 180), 18, False, (0, 0, 1, a), x, "o-"],
                              False, label="step = %.2fs" % step_time)
        ax.set_ylabel("Frequency",  fontsize=FONT_SIZE)
        ax.set_xlabel("$\\alpha$", fontsize=FONT_SIZE)
        # ax.legend(loc="upper right", fontsize=FONT_SIZE, frameon=False)
    elif name.startswith("d_switch"):
        step_time, mv = 0.2, 2
        all_switch_alpha, txs, tys, tvs, ts = get_switch_info(dfs, bouts, step_time, mv)
        alpha = lim_dir_a(np.array(all_switch_alpha))
        x0 = -180
        if name.endswith("abs"):
            alpha = np.abs(alpha)
            x0 = 0
        name, fig_info = "hist", [alpha, (x0, 180), 18, False, "b", x, "-"]
        # plot_hist_by_info(ax, fig_info, True)
        # ax.set_ylabel("Probability", fontsize=FONT_SIZE)
        plot_hist_by_info(ax, fig_info, False)
        ax.set_ylabel("Switch number", fontsize=FONT_SIZE)
        ax.set_xlabel("$\\alpha$", fontsize=FONT_SIZE)
    elif name == "switch_time":
        step_time, mv = 0.2, 2
        all_switch_alpha, txs, tys, tvs, ts = get_switch_info(dfs, bouts, step_time, mv)
        name, fig_info = "hist", [ts, (0, ETHOGRAM_TIME_RANGE), 60, False, "k", x, None]
        plot_hist_by_info(ax, fig_info, False)
    elif name == "switch_pos":
        if HEAT_CONFIG["fill_fig"]:
            ax.set_position([0, 0, 1, 1], which="both")
        all_switch_alpha, txs, tys, tvs, ts = get_switch_info(dfs, bouts)
        name, fig_info = "scatter", [txs, tys, CENTER_RANGE_X, CENTER_RANGE_X, .5, 2, color]
        plot_scatter_by_info(ax, fig_info, get_body_size(dfs, 2), line_color="k")
        fig_info[-3] = 5
    elif name == "switch_pos_mv":
        if HEAT_CONFIG["fill_fig"]:
            ax.set_position([0, 0, 1, 1], which="both")
        all_switch_alpha, txs, tys, tvs, ts = get_switch_info(dfs, bouts, mv=0)
        # color = (np.clip(tvs, 1, 4) - 1) / 3
        color = np.full((len(tvs),), COLOR_SLOW)
        color[np.array(tvs) > 4] = COLOR_FAST
        idx_fs = np.concatenate([np.nonzero(np.array(tvs) < 4)[0], np.nonzero(np.array(tvs) > 4)[0]])
        txs = np.array(txs)[idx_fs]
        tys = np.array(tys)[idx_fs]
        color = np.array(color)[idx_fs]
        name, fig_info = "scatter", [txs, tys, CENTER_RANGE_X, CENTER_RANGE_X, 3, 2, color] #15
        # plot_scatter_by_info(ax, fig_info, (BODY_LEN[2], BODY_WID[2]), cmap="winter", line_color="k")
        plot_scatter_by_info(ax, fig_info, get_body_size(dfs, 2), line_color="k")
        fig_info[-3] = 5
        return [2, (CENTER_RANGE_X, CENTER_RANGE_X), get_body_size(dfs, 2), "k"]
    elif name == "split_time":
        is_angle = check_is_angle(x)
        for s, e in bouts:
            xs = df[s:e][x]
            if is_angle:
                xs = correct_angle(xs.tolist())
                ax.set_ylim([-600, 600])
            ax.plot(xs, c="b", linewidth=0.1)
        # fig_info.append(["plot", "split_time", v, None, [[0, 150], [-600, 600]], fig_bins])
        ax.set_xlim([0, 150])
        range_y = STAT_LIMIT[fly].get(x)
        range_y and ax.set_ylim(range_y)
    elif name == "track_pass_angle":
        ax.grid(False)
        # for a_range in [[-180, 180, "y"]]:
        for a_range in [[-45, 0, "b"], [-90, -45, "b"], [-135, -90, "b"], [-180, -135, "b"], [0, 90, "g"], [0, 180, "r"]]:
        # for a_range in [[-45, 0, "b"], [0, 45, "b"], [45, 90, "b"], [90, 135, "b"], [135, 180, "b"], [-180, -135, "b"], [-135, -90, "b"], [-90, -45, "b"],
        #                 [0, 90, "g"], [90, 180, "g"], [-180, -90, "g"], [-90, 0, "g"],
        #                 [0, 180, "r"], [-180, 0, "r"], [-180, 180, "y"]]:
            r_l, a_l = calc_track_pass_angle(dfs[CENTER_FLY]["rel_polar:r"], dfs[CENTER_FLY]["rel_polar:t"], bouts, a_range)
            color = a_range[2]
            # for r, a in zip(r_l, a_l):
            for r, a in zip(r_l[:10], a_l[:10]):
                ax.plot(np.deg2rad(a+90), r, lw=0.1, color=color, alpha=0.5)
            r_l = np.concatenate(r_l)
            a_l = np.concatenate(a_l)
            vs_l, bins_l = bin_data_polar(r_l, a_l, a_range, bins=int((a_range[1] - a_range[0])/3))
            bins_l = np.deg2rad(bins_l+90)
            m = np.array([np.mean(vs) for vs in vs_l])
            # sd = np.array([np.std(vs) for vs in vs_l])
            ax.plot(bins_l, m, "-", lw=0.5, c=color)
            # ax.fill_between(bins_l, m - sd, m + sd, facecolor=color, alpha=.3)
        vs_l, bin_l = bin_data_polar(dfs[CENTER_FLY]["rel_polar:r"], dfs[CENTER_FLY]["rel_polar:t"], bins=HEAT_BINS)
        hist_f = [np.median(vs) for vs in vs_l]
        bin_l = np.deg2rad(bin_l+90)
        ax.plot(bin_l, hist_f, color="k", alpha=0.3, linewidth=0.5)
    elif name.startswith("partial_track"):
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_ylim(0, 4.5)
        a_range_l = [[0, 45, "r"], [45, 90, "g"], [90, 135, "b"], [135, 180, "y"], [180, 270, "c"], [270, 360, "k"]]
        single = None
        is_len_hist = False
        if name != "partial_track":
            if name.endswith("_len"):
                is_len_hist = True
                a_range_l = [[0, 45, "r"], [45, 90, "g"], [90, 135, "b"], [135, 180, "y"], [180, 225, "c"],
                             [225, 270, "c"], [270, 315, "k"], [315, 360, "k"]]
            else:
                single = int(name[-1])
                a_range_l = [a_range_l[single-1]]
        r_ll, a_ll = calc_track_in_angle_range(dfs[CENTER_FLY]["rel_polar:r"], dfs[CENTER_FLY]["rel_polar:t"], bouts, a_range_l)
        if is_len_hist:
            py = [len(rl) for rl in r_ll]
            py = np.array(py) / np.sum(py)
            name = "line"
            fig_info = [[(a[0] + a[1])/2 for a in a_range_l], py, None, None, [0, 1], "b", "-", None]
            plot_line_by_info(ax, fig_info)
        else:
            for i, a_range in enumerate(a_range_l):
                color = "b"#a_range[2]
                r_l, a_l = r_ll[i], a_ll[i]
                if single is not None:
                    for r, a in zip(r_l, a_l):
                    # for r, a in zip(r_l[:10], a_l[:10]):
                        ax.plot(np.deg2rad(np.array(a[::2])+90), r[::2], lw=0.1, color=color, alpha=0.1, zorder=1)
                if not len(r_l):
                    continue
                r_lc = np.concatenate(r_l)
                a_lc = np.concatenate(a_l)
                vs_l, bins_l = bin_data_polar(r_lc, a_lc)
                bins_l = np.deg2rad(bins_l+90)
                m = np.array([np.median(vs) for vs in vs_l])
                # sd = np.array([np.std(vs) for vs in vs_l])
                ax.plot(bins_l, m, "-", lw=0.5, c=color, zorder=100)
                # ax.fill_between(bins_l, m - sd, m + sd, facecolor=color, alpha=.3)
            vs_l, bin_l = bin_data_polar(dfs[CENTER_FLY]["rel_polar:r"], dfs[CENTER_FLY]["rel_polar:t"], bins=HEAT_BINS)
            hist_f = [np.median(vs) for vs in vs_l]
            bin_l = np.deg2rad(bin_l+90)
            ax.plot(bin_l, hist_f, color="k", alpha=0.5, linewidth=0.5, zorder=200)
            grids = [0, 90, 180, 270]
            labels = ["$-\pi/2$", 0, "$\pi/2$", "$\pi$"]
            plt.thetagrids(grids, labels, fontsize=FONT_SIZE)
    elif name.startswith("time_align"):
        t_range = 4
        iz = calc_time_align_pcolor(dfs, bouts, name.split("_")[-1], t_range=t_range, bins=int(t_range*30), sort=True)
        iz = np.array([zz[1] for zz in iz])
        extent = [-t_range, t_range, 0, len(iz)]
        xa = np.array(dfx)
        z = np.array([xa[zz] for zz in iz], dtype=float)
        z[iz == 0] = np.nan
        lim_heat = lim_heat or HEAT_LIMIT["head"].get(xo, [None, None])
        im = ax.imshow(z, cmap="jet", vmin=lim_heat[0], vmax=lim_heat[1], extent=extent, interpolation="nearest", origin="lower", aspect="auto")
        if HEAT_CONFIG["color_bar"]:
            plot_colorbar(ax, lim_heat[0], lim_heat[1], im=im)
        ax.plot([0, 0], [0, extent[-1]], c="w", lw=1)
        ax.set_xlim([-t_range, t_range])
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
        ax.set_ylabel(REPLACE_LABEL.get(xo, xo), fontsize=FONT_SIZE)
    elif name.startswith("angle_align"):
        a_range = 180
        iz = calc_angle_align_pcolor(dfs, bouts, name.split("_")[-1], a_range=a_range)
        iz = np.array([zz[1] for zz in iz])
        extent = [-a_range, a_range, 0, len(iz)]
        xa = np.array(dfx)
        z = np.array([xa[zz] for zz in iz])
        z[iz == 0] = np.nan
        lim_heat = lim_heat or HEAT_LIMIT["head"].get(xo, [None, None])
        im = ax.imshow(z, cmap="jet", vmin=lim_heat[0], vmax=lim_heat[1], extent=extent, interpolation="nearest", origin="lower", aspect="auto")
        if HEAT_CONFIG["color_bar"]:
            plot_colorbar(ax, lim_heat[0], lim_heat[1], im=im)
        ax.plot([0, 0], [0, extent[-1]], c="w", lw=1)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
        ax.set_xlabel("$\\alpha$", fontsize=FONT_SIZE)
    elif name.startswith("angle_d_align"):
        if xo.endswith("rel_polar:t"):
            fly = 2
            df = dfs[fly]
            x, dfx = k_with_dfk(xo, df)
        if name.endswith("_all"):
            ret1 = plot_figure(ax, dfs, fly, "angle_d_align_head", xo, yo, bouts, color=COLOR_HEAD)
            ret2 = plot_figure(ax, dfs, fly, "angle_d_align_side", xo, yo, bouts, color=COLOR_SIDE)
            ret4 = plot_figure(ax, dfs, fly, "angle_d_align_tail", xo, yo, bouts, color=COLOR_TAIL)
            return [ret1, ret2, ret4]
        if name.endswith("_ud"):
            ret1 = plot_figure(ax, dfs, fly, "angle_d_align_sideu", xo, yo, bouts, color=COLOR_SIDE_TO_HEAD)
            ret2 = plot_figure(ax, dfs, fly, "angle_d_align_sided", xo, yo, bouts, color=COLOR_SIDE_TO_TAIL)
            return [ret1, ret2]
        # ax.set_xlabel("$\\alpha$", fontsize=FONT_SIZE)
        # ax.set_ylabel(REPLACE_LABEL.get(xo, xo), fontsize=FONT_SIZE)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
        ax.set_xlabel("")
        align = name.split("_")[-1]
        color = COLOR_MAP.get(align, color)
        a_range = 180
        iz = calc_angle_align_pcolor(dfs, bouts, align, a_range=a_range, random=name.endswith("_rand"))
        xa = np.array(dfx)
        if USE_MEAN_PAIR:
            iz_l = bouts_aligned_group_by_pair(iz)
            py_l = []
            for pair_iz in iz_l:
                pair_iz = np.array(pair_iz)
                z = np.array([xa[zz] for zz in pair_iz])
                z[pair_iz == 0] = np.nan
                py_l.append(np.nanmean(z, axis=0))
            py = np.nanmean(py_l, axis=0)
            err_l = np.nanstd(py_l, axis=0)
            if USE_SEM:
                err_l = err_l / np.sqrt(np.count_nonzero(~np.isnan(py_l), axis=0))
        else:
            iz = np.array([zz[1] for zz in iz])
            z = np.array([xa[zz] for zz in iz])  # NOTE: faster than dfx[zz]
            z[iz == 0] = np.nan
            py = np.nanmean(z, axis=0)
            if USE_SEM:
                err_l = np.nanstd(z, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(z), axis=0))
            else:
                err_l = np.nanstd(z, axis=0)
        px = np.linspace(-a_range, a_range, 60)
        lim_heat = lim_heat or HEAT_LIMIT["head"].get(xo, [np.nanmin(py), np.nanmax(py)])
        name, fig_info = "line", [px, py, err_l, [-a_range, a_range], lim_heat, color, "-", 0]
        plot_line_by_info(ax, fig_info)
    elif name.startswith("time_d_align"):
        if xo.endswith("rel_polar:t"):
            fly = 2
            df = dfs[fly]
            x, dfx = k_with_dfk(xo, df)
        if name.endswith("_all"):
            ret1 = plot_figure(ax, dfs, fly, "time_d_align_head", xo, yo, bouts, color=COLOR_HEAD)
            ret2 = plot_figure(ax, dfs, fly, "time_d_align_side", xo, yo, bouts, color=COLOR_SIDE)
            ret3 = plot_figure(ax, dfs, fly, "time_d_align_tail", xo, yo, bouts, color=COLOR_TAIL)
            # lim_heat = [min(ret1[2][4][0], ret2[2][4][0], ret3[2][4][0]) - 0.1, max(ret1[2][4][1], ret2[2][4][1], ret3[2][4][1]) + 0.1]
            # ax.plot([0, 0], lim_heat, c="r", lw=1)
            # ret3[2][4] = lim_heat
            return [ret1, ret2, ret3]
        if name.endswith("_ud"):
            ret1 = plot_figure(ax, dfs, fly, "time_d_align_sideu", xo, yo, bouts, color=COLOR_SIDE_TO_HEAD)
            ret2 = plot_figure(ax, dfs, fly, "time_d_align_sided", xo, yo, bouts, color=COLOR_SIDE_TO_TAIL)
            return [ret1, ret2]
        if name.endswith("_rand"):
            color = "k"
        align = name.split("_")[-1]
        iz = calc_time_align_pcolor(dfs, bouts, align, random=name.endswith("_rand"))
        xa = np.array(dfx)
        if USE_MEAN_PAIR:
            iz_l = bouts_aligned_group_by_pair(iz)
            py_l = []
            for pair_iz in iz_l:
                pair_iz = np.array(pair_iz)
                z = np.array([xa[zz] for zz in pair_iz], dtype=float)
                z[pair_iz == 0] = np.nan
                py_l.append(np.nanmean(z, axis=0))
            py = np.nanmean(py_l, axis=0)
            err_l = np.nanstd(py_l, axis=0)
            if USE_SEM:
                err_l = err_l / np.sqrt(np.count_nonzero(~np.isnan(py_l), axis=0))
        else:
            iz = np.array([zz[1] for zz in iz])
            z = np.array([xa[zz] for zz in iz])  # NOTE: faster than dfx[zz]
            z[iz == 0] = np.nan
            py = np.nanmean(z, axis=0)
            if USE_SEM:
                err_l = np.nanstd(z, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(z), axis=0))
            else:
                err_l = np.nanstd(z, axis=0)

        px = np.linspace(-1, 1, 60)
        lim_heat = lim_heat or HEAT_LIMIT["head"].get(xo, [np.nanmin(py), np.nanmax(py)])
        name, fig_info = "line", [px, py, err_l, [-1, 1], lim_heat, color, "-", 0]
        if not isinstance(py, np.float64):
            plot_line_by_info(ax, fig_info)
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
        # ax.set_xlabel("")
        ax.set_ylabel(REPLACE_LABEL.get(xo, xo), fontsize=FONT_SIZE)
    elif name == "fc_color":
        if HEAT_CONFIG["fill_fig"]:
            ax.set_position([0, 0, 1, 1], which="both")
        xs, ys = dfs[CENTER_FLY][WEIGHT_HEAT_X], dfs[CENTER_FLY][WEIGHT_HEAT_Y]
        weights = dfx
        lim_heat = lim_heat or HEAT_LIMIT[fly].get(xo)
        color = (np.clip(weights, lim_heat[0], lim_heat[1]) - lim_heat[0]) / (lim_heat[1] - lim_heat[0])
        fig_info = [xs, ys, CENTER_RANGE_X, CENTER_RANGE_X, .1, 2, color]
        name, fig_info = "scatter", plot_scatter_by_info(ax, fig_info, BODY_SIZE[2], cmap="jet", line_color="k", alpha=0.1)
    elif name.startswith("trigger_stop") or name.startswith("trigger_start"):
        if name.endswith("_len"):
            dfr, dft = np.array(dfs[CENTER_FLY]["rel_polar:r"]), np.array(dfs[CENTER_FLY]["rel_polar:t"])
            bouts_l = [calc_track_stop_angle(dfr, dft, bouts, i, is_start=name.startswith("trigger_start")) for i in [(0,), (1, 2), (3,)]]
            # bouts_len = [[(e-s)/66 for s, e in bouts] for bouts in bouts_l]
            bouts_len = [[np.nansum(np.sqrt(dfs[1]["pos:x"][s:e].diff()**2 + dfs[1]["pos:y"][s:e].diff()**2)) for s, e in bouts] for bouts in bouts_l]
            colors = COLOR_HEAD, COLOR_SIDE, COLOR_TAIL
            # ax.set_ylim(0, 0.1)
            for i, bl in enumerate(bouts_len):
                fig_info = [bl, [0, 30], 30, False, colors[i], x, "-"]
                plot_hist_by_info(ax, fig_info, need_norm=False)
            ax.set_ylabel("Trajectory number", fontsize=FONT_SIZE)
            ax.set_xlabel("Trajectory length (mm)", fontsize=FONT_SIZE)
            return
        if xo.endswith("rel_polar:t"):
            fly = 2
            df = dfs[fly]
            x, dfx = k_with_dfk(xo, df)
        sta = stop_trigger_avg(dfx, dfs, bouts, 66)
        px = np.linspace(-1, 0, 66)
        colors = COLOR_HEAD, COLOR_SIDE, COLOR_TAIL
        lim2 = HEAT_LIMIT["start"] if name == "trigger_start" else HEAT_LIMIT["head"]
        ret = []
        for i, (py, pe) in enumerate(sta):
            if isinstance(py, np.float64) or len(py) < len(px):
                continue
            lim_heat = lim_heat or lim2.get(xo, [np.nanmin(py), np.nanmax(py)])
            # name, fig_info = "line", [px, py, pe, [-180, 180], lim_heat, color, "-", px[-1]]
            name, fig_info = "line", [px, py, pe, [-1, 0], lim_heat, colors[i], "-", None]
            plot_line_by_info(ax, fig_info)
            ret.append([name, len(bouts), fig_info])
        # ax.set_xticks(np.linspace(-180, 180, 5))
        # ax.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
        ax.set_ylabel(REPLACE_LABEL.get(xo, xo), fontsize=FONT_SIZE)
        set_ytick_by_name(ax, xo)
        return ret
    elif name.startswith("stop_trigger_track"):
        dfr, dft = np.array(dfs[CENTER_FLY]["rel_polar:r"]), np.array(dfs[CENTER_FLY]["rel_polar:t"])
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_ylim(0, 4.5)
        POLAR = False
        PLOT_BOUTS = 1000
        bouts1 = calc_track_stop_angle(dfr, dft, bouts, int(name[-1]), PLOT_BOUTS)
        if POLAR:
            r_l, a_l = [], []
            for s, e in bouts1:
                if e - s > 66:
                    s = e - 66
                r_l.append(dfr[s:e])
                a_l.append(dft[s:e])
            for r, a in zip(r_l, a_l):
                ax.plot(np.deg2rad(np.array(a)+90), r, lw=1, alpha=0.6, zorder=1)
            ax.scatter([np.deg2rad(np.array(a[-1]) + 90) for a in a_l], [r[-1] for r in r_l], color="r", zorder=1, s=1)
            # ax.scatter([np.deg2rad(np.array(a[0]) + 90) for a in a_l], [r[0] for r in r_l], color="g", zorder=1, s=1)

            r_lc = np.concatenate(r_l)
            a_lc = np.concatenate(a_l)
            vs_l, bins_l = bin_data_polar(r_lc, a_lc)
            bins_l = np.deg2rad(bins_l+90)
            m = np.array([np.median(vs) for vs in vs_l])
            ax.plot(bins_l, m, "-", lw=0.5, c=color, zorder=100)

            vs_l, bin_l = bin_data_polar(dfs[CENTER_FLY]["rel_polar:r"], dfs[CENTER_FLY]["rel_polar:t"], bins=HEAT_BINS)
            hist_f = [np.median(vs) for vs in vs_l]
            bin_l = np.deg2rad(bin_l+90)
            ax.plot(bin_l, hist_f, color="k", alpha=0.5, linewidth=0.5, zorder=200)
            grids = [0, 90, 180, 270]
            labels = ["$-\pi/2$", 0, "$\pi/2$", "$\pi$"]
            plt.thetagrids(grids, labels, fontsize=FONT_SIZE)
        else:
            x_l = []
            y_l = []
            d_l = []
            df_x = np.array(dfs[CENTER_FLY]["rel_pos:x"])
            df_y = np.array(dfs[CENTER_FLY]["rel_pos:y"])
            df_d = np.array(dfs[CENTER_FLY]["dir"] - dfs[1]["dir"])
            for s, e in bouts1:
                if e - s > 7:
                    s = e - 7
                x_l.append(df_x[s:e] - df_x[e-1])
                y_l.append(df_y[s:e] - df_y[e-1])
                d_l.append(df_d[s:e] + 90)
            for i, (xs, ys, ds) in enumerate(zip(x_l, y_l, d_l)):
                # plot_overlap(ax, xs, ys, None, i)
                xs, ys, dirs = filter_wrong_pos(xs, ys, ds)
                ax.plot(xs, ys, zorder=10, lw=.5, color="C%d" % (i % 10), alpha=0.5)
            # ax.scatter([x[-1] for x in x_l], [y[-1] for y in y_l], color="r", zorder=1)
            # ax.scatter([x[0] for x in x_l], [y[0] for y in y_l], color="g", zorder=1, s=2)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.axis("equal")
    elif name.startswith("trigger_ac_etho"):  # NOTE: stat0.pickle, cir_meta
        is_end = name.endswith("_end")
        is_multi = name.find("_multi") > 0
        sec = 3
        lim_frame = 66 * sec
        px = np.linspace(-sec, 0, lim_frame)
        bouts_a = combine_bouts_multi(cir_meta) if is_multi else combine_bouts(cir_meta)
        ac_bouts = cir_meta[x or "acp_bouts"]
        ac_l = []
        for s, e in ac_bouts:
            if s > lim_frame:
                ss = e if is_end else s + 1
                if is_multi:
                    ac_l.extend(bouts_a[:, ss-lim_frame:ss])
                else:
                    ac_l.append(bouts_a[ss-lim_frame:ss])
        plot_stack_info(ax, px, np.array(ac_l), bouts_key_l, is_multi)
        ax.legend(loc="upper left")
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
        ax.set_ylabel("Actions", fontsize=FONT_SIZE)
    elif name == "trigger_ac":  # NOTE: stat0.pickle, cir_meta
        ac_bouts = cir_meta["acp_bouts"]
        ac_l = []
        sec = 3
        lim_frame = 66 * sec
        # dfxx = dfs[2]["rel_pos:x"]
        # dfyy = dfs[2]["rel_pos:y"]
        # xs, ys = [], []
        for s, e in ac_bouts:
            if s > lim_frame:
                ac_l.append(dfx[s-lim_frame:s])
        #     ac_l.append(fill_nan(dfx[s:e], lim_frame))
        #     xs.append(dfxx[e])
        #     ys.append(dfyy[e])

        # name, fig_info = "scatter", [xs, ys, CENTER_RANGE_X, CENTER_RANGE_X, .5, 2, color]
        # plot_scatter_by_info(ax, fig_info, (BODY_LEN[2], BODY_WID[2]), cmap="plasma", line_color="k")
        px = np.linspace(-sec, 0, lim_frame)
        py, err_l = lines_to_err_band(ac_l)
        name, fig_info = "line", [px, py, err_l, [-sec, 0], lim_heat or HEAT_LIMIT[fly].get(x), "k", "-", None]
        fig_info = plot_line_by_info(ax, fig_info)
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
        ax.set_ylabel(REPLACE_LABEL.get(xo, xo), fontsize=FONT_SIZE)
        set_ytick_by_name(ax, xo)
    elif name == "cor_court_fab":  # NOTE: stat0.pickle, cir_meta
        a_courtl = bouts_to_seq(cir_meta, "we_on_l_bouts")
        a_courtr = bouts_to_seq(cir_meta, "we_on_r_bouts")
        a_fabl = bouts_to_seq(cir_meta, "fabl_bouts")
        a_fabr = bouts_to_seq(cir_meta, "fabr_bouts")

        ts = np.arange(-66 * 3, 66 * 3)
        corll = cross_correlate(a_courtl, a_fabl, ts)
        corrr = cross_correlate(a_courtr, a_fabr, ts)
        corlr = cross_correlate(a_courtl, a_fabr, ts)
        corrl = cross_correlate(a_courtr, a_fabl, ts)
        ax.plot(ts/66, corll, label="courtl x fabl")
        ax.plot(ts/66, corrr, label="cooutr x fabr")
        ax.plot(ts/66, corlr, label="courtl x fabr")
        ax.plot(ts/66, corrl, label="cooutr x fabl")
        ax.legend(loc="upper left")
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
    elif name == "xcor":  # NOTE: dfs=pair_folder_l
        py_l = []
        fps = 66
        fs = np.arange(-int(fps * 1.5), int(fps * 1.5))
        for pair_folder in dfs:
            cir_meta = load_meta(pair_folder)
            fps = int(cir_meta["FPS"])
            if not cir_meta or cir_meta["copulate"] or fps != 66:
                continue
            bouts = cir_meta["center_bouts"]
            if x.startswith("rel") or y.startswith("rel"):
                dfsi, cir_meta = load_dfs_meta_cache(pair_folder)
            if x.startswith("rel"):
                dfx = np.array(dfsi[1][x])
            else:
                dfx = bouts_to_seq(cir_meta, x)
            if y.startswith("rel"):
                dfy = np.array(dfsi[1][y])
            else:
                dfy = bouts_to_seq(cir_meta, y)
            xcor = cross_corr_by_bouts(dfx, dfy, fs, bouts)
            # ax.plot(fs/fps, xcor)
            if not np.isnan(xcor.max()):  # NOTE: xcor=nan
                py_l.append(xcor)
        py, err_l = lines_to_err_band(py_l)
        name, fig_info = "line", [fs/fps, py, err_l, [-1.5, 1.5], [-0.3, 0.3], "k", "-", 0]
        plot_line_by_info(ax, fig_info)
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
        ax.set_ylabel("%s x %s (n=%d)" % (x, y, len(py_l)), fontsize=FONT_SIZE)
    elif name == "r_alpha_fab":  # NOTE: stat0.pickle, cir_meta
        # far_idx = np.nonzero(~dfs[0]["overlap"])[0]
        far_idx = np.nonzero(dfs[1]["rel_polar_ht:r"] > 1)[0]
        # far_idx = np.nonzero(dfs[2]["rel_polar_h:r"] > DIST_TO_FLY_FAR)[0]
        alpha = dfs[2]["rel_polar:t"]
        fabl = list(set(dfs_bouts_idx(cir_meta["fabl_bouts"])).intersection(far_idx))
        fabr = list(set(dfs_bouts_idx(cir_meta["fabr_bouts"])).intersection(far_idx))

        ret = []
        fig_info = [alpha[fabl], [-180, 180], 30, False, "m", None, "-"]
        plot_hist_by_info(ax, fig_info, label="left")
        ret.append(["hist", len(bouts), fig_info])
        fig_info = [alpha[fabr], [-180, 180], 30, False, "gold", None, "-"]
        plot_hist_by_info(ax, fig_info, label="right")
        ret.append(["hist", len(bouts), fig_info])
        ax.set_xlabel("$\\alpha$")
        return ret
    elif name == "r_sdx_sdy":
        dfs_nh = dfs_head_semicircle(dfs, True)
        xs_l, ys_l = split_pair2(dfs_nh[0]["pair"], dfs_nh[1]["rel_pos:x"], dfs_nh[1]["rel_pos:y"])
        sdx = [np.std(xs) for xs in xs_l if len(xs) > 20]
        sdy = [np.std(ys) for ys in ys_l if len(ys) > 20]
        sn = np.sqrt(len(sdx))
        sdx_m, sdx_e = np.mean(sdx), np.std(sdx) / sn
        sdy_m, sdy_e = np.mean(sdy), np.std(sdy) / sn
        ax.scatter(sdx, sdy, s=2)
        ax.errorbar([sdx_m], [sdy_m], xerr=[sdx_e], yerr=[sdy_e], color="r")
        ax.set_xlabel("SD(x) (mm)")
        ax.set_ylabel("SD(y) (mm)")
        return ["scatter_sd", len(bouts), [sdx, sdy, "r"]]
    else:
        return plot_figure2(ax, dfs, fly, name, x, y, bouts, color=color, count=count, lim_heat=lim_heat)
    return [name, len(bouts), fig_info]

def fill_nan(a, lim_frame):
    n = len(a)
    if n > lim_frame:
        return a[-lim_frame:]
    else:
        return np.concatenate([np.full((lim_frame - n), np.nan), a])

def set_ytick_by_name(ax, xo):
    PROPERTY_TICK = {
        # "rel_polar:r": [2, 4],
        "abs_rel_polar:t": [0, 90, 180],
        "rel_polar:t": [-180, 90, 0, 90, 180],
        # "v_len": [0, 15],
        # "av": [30, 220],
        # "wing_m": [60, 100],
        "theta": [-60, 60],
        "abs_theta": [50, 70, 90],
        # "vf": [-1, 4],
        # "vs": [-7.5, 7.5],
        # "abs_vs": [0, 15],
        # "acc": [-7, 7],
        # "abs_acc": [15, 40],
        # "speed": [0, 15],
        # "abs_av": [80, 220],
    }
    tick = PROPERTY_TICK.get(xo)
    if tick:
        ax.set_yticks(tick)
        ax.set_yticklabels(tick)

def plot_line_by_info(ax, info, label=None):
    px, py, pe, limx, limy, color, line, redx = info
    ax.plot(px, py, line, color=color, zorder=100, label=label)
    if pe is not None:
        py = np.array(py)
        ax.fill_between(px, py - pe, py + pe, facecolor=color, alpha=.3, zorder=100)
    if redx is not None:
        ax.plot([redx, redx], limy, "k--", lw=1, zorder=0)
    ax.set_xlim(limx)
    limy and ax.set_ylim(limy)

def plot_line_with_cmap(ax, info):
    bin_l1, hist_f, pval_c, cmap = info
    ax.scatter(bin_l1, hist_f, c=pval_c, cmap=cmap, marker="o", s=30)
    points = np.array([bin_l1, hist_f]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    t = pval_c[1:]
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(t)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
    ax.set_xlabel("$\\alpha$", fontsize=FONT_SIZE)
    ax.set_ylabel("Difference (mm)", fontsize=FONT_SIZE)

def pval_to_color(pval):  # -1~-5 -> 0~1
    min_log10 = 10
    return (np.clip(-np.log10(pval), 1, min_log10) - 1) / (min_log10 - 1)

def plot_weight_heat_by_weights(ax, dfs, weights, lim_heat, bouts, center_fly=CENTER_FLY, center_range=CENTER_RANGE):
    xs, ys = dfs[center_fly][WEIGHT_HEAT_X], dfs[center_fly][WEIGHT_HEAT_Y]
    if WEIGHT_HEAT_X.startswith("rel_polar"):
        fig_info = plot_weight_heat(ax, xs - REL_POLAR_T_OFFSET, ys, center_fly, POLAR_RANGE, lim_heat, None, weights=weights)
    else:
        fig_info = plot_weight_heat(ax, xs, ys, center_fly, center_range, lim_heat, BODY_SIZE[center_fly], weights=weights)
    return "weight_heat", len(bouts), fig_info

def get_switch_info(dfs, bouts, step_time=0.2, mv=2, save=False):
    all_switch_alpha = []
    all_switch_time = []
    turn_pos_x = []
    turn_pos_y = []
    turn_v = []
    # turn_counts = []
    info = []
    for s, e in bouts:
        # fps = (e-s)/(dfs[0]["time"][e-1] - dfs[0]["time"][s])
        fps = (e-s)/(dfs[0]["time"].iloc[e-1] - dfs[0]["time"].iloc[s])
        times = np.array(dfs[0][s:e]["time"])
        bout1 = dfs[1][s:e]
        bout2 = dfs[2][s:e]
        # if bout2["v_len"].max() < FINE_CIR_FEMALE_MAX_V and bout2["av"].max() < FINE_CIR_FEMALE_MAX_AV: # NOTE: unstable cause female motion
        alpha = np.array(correct_angle((bout2["rel_polar:t"] - REL_POLAR_T_OFFSET).tolist()))
        turn_frame = calc_switch_sections2(alpha, bout1["v_len"].tolist(),
                                         bout2["v_len"].tolist(), bout2["av"].tolist(), step_time * fps, min_mv=mv)
        # turn_frame = calc_switch_sections3(alpha)
        turns = len(turn_frame)
        # turn_counts.append(turns)
        if turns:
            all_switch_alpha.extend(alpha[turn_frame])
            all_switch_time.extend(times[turn_frame])
            turn_pos_x.extend(bout2["rel_pos:x"].iloc[turn_frame].tolist())
            turn_pos_y.extend(bout2["rel_pos:y"].iloc[turn_frame].tolist())
            h = int(step_time * fps / 2)
            for ff in turn_frame:
                turn_v.append(bout1["v_len"].iloc[ff-h:ff+h].mean())
            if save:
                row = dfs[0]["pair"][s], dfs[0]["frame"][s], dfs[0]["frame"][e - 1], " ".join(
                    ["%d" % t for t in turn_frame])
                info.append(row)
    if save:
        pd.DataFrame(info, columns=["pair", "s", "e", "switchs"]).to_csv("img/turn.csv", index=False)
    return all_switch_alpha, turn_pos_x, turn_pos_y, turn_v, all_switch_time

def find_ac(ov, n=3):
    c = 0
    for i, o in enumerate(ov):
        if o:
            c += 1
            if c >= n:
                return i
        else:
            c = 0
    return 0

def get_ac_info(dfs, bouts, save=True):
    info = []
    dft = np.array(dfs[0]["time"])
    dfp = np.array(dfs[0]["pair"])
    dff = np.array(dfs[0]["frame"])
    dfo = np.array(dfs[0]["overlap"])
    ret = []
    for s, e in bouts:
        fps = (e - s) / (dft[e - 1] - dft[s])
        if fps > 40:
            ac = find_ac(dfo[s:e])
            if ac > 5:
                row = dfp[s], dff[s], dff[s + ac - 1]
                info.append(row)
                ret.append([s, s + ac])
    if save:
        pd.DataFrame(info, columns=["pair", "s", "e"]).to_csv("img/ac.csv", index=False)
    return ret

def plot_stack_info(ax, xs, al, keys, is_multi=False):
    stack_info = [], []
    total = len(al)
    if is_multi:
        total /= len(keys)
    for i, key in enumerate(keys):
        s = np.count_nonzero(al == i + 1, axis=0) / total
        # if np.count_nonzero(s) > 0:
        stack_info[0].append(key)
        stack_info[1].append(s)
    if is_multi:
        for k, s in zip(stack_info[0], stack_info[1]):
            ax.plot(xs, s, label=k)
    else:
        ax.stackplot(xs, stack_info[1], labels=stack_info[0])

def plot_cw_wc_time_course(dfs, bouts, start_frame_l):
    fz = 24
    for s, e in bouts:
        if s in start_frame_l:
            df0 = dfs[0][s:e]
            df1 = dfs[1][s:e]
            df2 = dfs[2][s:e]
            ts = np.array(df0["time"])
            ts = ts - ts[0]
            alpha = np.array(correct_angle((df2["rel_polar:t"] - REL_POLAR_T_OFFSET).tolist()))
            wing_l = np.array(correct_wing(df1["wing_l"]))
            wing_r = np.array(correct_wing(df1["wing_r"]))
            we_l = np.abs(wing_l) > ANGLE_WE_MIN
            we_r = np.abs(wing_r) > ANGLE_WE_MIN
            lim_x = 0, ts[-1]

            plt.figure(figsize=(lim_x[1]+1, 2), constrained_layout=True, dpi=300)
            ax = plt.gca()
            ax.tick_params(labelsize=fz)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(lim_x)
            ax.set_xticks(np.arange(0, int(lim_x[1])+1))
            ax.set_xlabel("Time (s)", fontsize=fz)
            ax.set_ylabel("$\\alpha$ (°)", fontsize=fz)

            n = len(ts)
            cl = np.full((n,), COLOR_SIDE, dtype="object")
            cl[we_l > 0] = COLOR_LEFT
            cl[we_r > 0] = COLOR_RIGHT
            cl[(we_r > 0) & (we_l > 0)] = "#00ff00"
            ax.scatter(ts, alpha, c=cl, s=4)
            ax.plot(lim_x, [0, 0], "k--")
            ylim = ax.get_ylim()
            print(ylim)
            if ylim[1] - ylim[0] < 500:
                ticks = []
                ticklabels = []
                for y, t in zip([-180, -90, 0, 90, 180], ["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"]):
                    if ylim[0] < y < ylim[1]:
                        ticks.append(y)
                        ticklabels.append(t)
            else:
                if ylim[1] > 500:
                    ticks = [0, 540]
                    ticklabels = ["0", "$3\pi$"]
                else:
                    ticks = [-900, 0]
                    ticklabels = ["$-5\pi$", "0"]
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            # mid = (np.max(alpha) + np.min(alpha)) / 2
            # ax.set_ylim(mid - 240, mid + 240)
            # ax.set_title(dfs[0]["pair"][s] + " %d %d" % (dfs[0]["frame"][s], dfs[0]["frame"][e - 1]))
            save_and_open("img/wing_and_leg/time_course_%d_alpha" % s)

            plt.figure(figsize=(lim_x[1]+1, 2), constrained_layout=True, dpi=300)
            ax = plt.gca()
            ax.tick_params(labelsize=fz)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(lim_x)
            ax.set_xticks(np.arange(0, int(lim_x[1])+1))
            ax.set_xlabel("Time (s)", fontsize=fz)
            ax.set_ylabel("$\\beta$ (°)", fontsize=fz)

            ax.plot(ts, -wing_l, "-", color=COLOR_LEFT)
            ax.plot(ts, wing_r, "-", color=COLOR_RIGHT)
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 90])
            ax.set_yticklabels(["0", "$\pi/2$"])
            save_and_open("img/wing_and_leg/time_course_%d_wing" % s)

def plot_we_alpha_bout(dfs, bouts, path, n, longest=True, bout_ids=[]):
    # step_time, mv = 0.2, 2
    bouts_count = 8#min(72, len(bouts))#*5
    row = 4#*2
    fig, axes = plt.subplots(bouts_count//row, row, figsize=(row * 8, bouts_count / row * 2), sharex=False)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0, wspace=0)
    axes = axes.flatten()
    i = 0
    if longest:
        if bout_ids:
            el = []
            bouts.sort(key=lambda x: x[0] - x[1])
            bouts = bouts[:100]
            bouts1 = []
            for bout in bout_ids:
                bouts1.append(bouts[bout])
                print(bouts[bout])
            #     el.append(int(bout.split(" ")[-2]))
            bouts = bouts1
            exit(0)
        else:
            bouts.sort(key=lambda x: x[0] - x[1])
            bouts = bouts[:100]
    for s, e in bouts:
        if n == 1:
            bout0 = dfs[0].loc[s:e]
            bout1 = dfs[1].loc[s:e]
            bout2 = dfs[2].loc[s:e]
        else:
            bout0 = dfs[0][s:e]
            bout1 = dfs[1][s:e]
            bout2 = dfs[2][s:e]
        # if bout_ids and s not in dfs[0]["frame"][s] not in el:
            #s not in [16240,18183 ,65738 ,66977 ,298095,364412,644791,808830,]
            # continue
        print(s, e)
        ts = np.array(bout0["time"])
        if ts[-1] - ts[0] < 1.5:
            continue
        we_l = np.array(bout1["we_l"])
        we_r = np.array(bout1["we_r"])
        we_i = np.array(bout1["we_ipsi"])
        ts = ts - ts[0]
        alpha = np.array(correct_angle((bout2["rel_polar:t"] - REL_POLAR_T_OFFSET).tolist()))
        # turn_frame = calc_switch_sections2(alpha, bout1["v_len"].tolist(),
        #                                    bout2["v_len"].tolist(), bout2["av"].tolist(), step_time * fps,
        #                                    min_mv=mv)
        wing_l = np.array(correct_wing(bout1["wing_l"]))
        wing_r = np.array(correct_wing(bout1["wing_r"]))
        we_m = np.array(wing_r + wing_l) / 2
        we_ch_frame = np.diff(we_m > 0).nonzero()[0]
        lim_x = 0, 9
        if not longest:
            turn_frame = calc_switch_sections3(alpha)
            if len(turn_frame) == 0:
                continue
            lim_x = 0, 2
        ax = axes[i]
        n = len(ts)
        # lim_y = np.min(alpha) - 10, np.max(alpha) + 10
        mid = (np.max(alpha) + np.min(alpha)) / 2
        lim_y = mid - 240, mid + 240
        cl = np.full((n,), COLOR_SIDE, dtype="object")
        cl[we_l > 0] = COLOR_LEFT
        cl[we_r > 0] = COLOR_RIGHT
        cl[(we_r > 0) & (we_l > 0)] = COLOR_BACKWARD
        ax.scatter(ts, alpha, c=cl, s=4)
        # ax.plot([ts[turn_frame], ts[turn_frame]], lim_y, "b", alpha=.3)
        # ax.plot(lim_x, [0, 0], COLOR_HEAD, alpha=.3)
        # ax.plot(lim_x, [180, 180], COLOR_TAIL, alpha=.3)
        # ax.plot(lim_x, [-180, -180], COLOR_TAIL, alpha=.3)
        cli = np.full((n,), "gray", dtype="object")
        cli[we_i > 0] = "g"
        cli[we_i < 0] = "r"
        # ax.scatter(ts, np.full((n,), lim_y[1] - 10), marker="_", color=cli, s=16)
        # ax.set_ylim(lim_y)
        ax.set_xlim(lim_x)
        ax.set_title(dfs[0]["pair"][s] + " %d %d" % (dfs[0]["frame"][s], dfs[0]["frame"][e - 1]))

        ax1 = plt.twinx(ax)
        # ax1.scatter(ts, we_m, marker="_", color=cli, s=16)
        # ax1.scatter(ts, -wing_l, marker="_", color=COLOR_LEFT, s=16)
        ax1.plot(ts, -wing_l, "-", color=COLOR_LEFT, alpha=.5)
        # ax1.scatter(ts, wing_r, marker="_", color=COLOR_RIGHT, s=16)
        ax1.plot(ts, wing_r, "-", color=COLOR_RIGHT, alpha=.5)
        lim_y = [0, 105]
        ax1.set_ylim(lim_y)
        # ax1.plot(lim_x, [0, 0], "y", alpha=.3)
        # if len(we_ch_frame):
        #     ax1.plot([ts[we_ch_frame], ts[we_ch_frame]], lim_y, "y", alpha=.3)
        i += 1
        if i >= bouts_count:
            break
    # plt.show()
    plt.tight_layout()
    save_and_open(path, save_svg=True)

def plot_ratio_heat(ax, xs1, ys1, xs2, ys2, xy_range, fly=1):
    # NOTE: 1/(1+2)
    bin_h, hist_bx, hist_by = bin_data_2d(xs1, ys1, np.ones((len(xs1),)), xy_range, 60)
    bin_t, hist_bx, hist_by = bin_data_2d(xs2, ys2, np.ones((len(xs2),)), xy_range, 60)
    extent = [hist_bx[0], hist_bx[-1], hist_by[-1], hist_by[0]]

    bin_h = np.array([[len(binj) for binj in bini] for bini in bin_h])
    bin_t = np.array([[len(binj) for binj in bini] for bini in bin_t])
    bin_r = bin_h / (bin_h + bin_t)
    bin_r[(bin_h + bin_t) < HEAT_CONFIG["min_bin"]] = np.nan
    lim_heat = [0, 1]
    plot_heat_by_info(ax, bin_r, extent, lim_heat, fly, xy_range, BODY_SIZE[fly])

def plot_figure2(ax, dfs, fly, name, x=None, y=None, bouts=[], color=None, count=None, lim_heat=None):
    ret = None

    if name.startswith("ccw_"):
        ax.set_position([0, 0, 1, 1], which="both")
        if name.endswith("_head"):
            x, y = "rel_pos_h:x", "rel_pos_h:y"
        elif name.endswith("_center"):
            x, y = "rel_pos:x", "rel_pos:y"
        elif name.endswith("_tail"):
            x, y = "rel_pos_t:x", "rel_pos_t:y"

        if name.find("_far") > 0:
            dfs = dfs_far(dfs, True)
        elif name.find("_near") > 0:
            dfs = dfs_far(dfs, False)

        if name.find("_FL") > 0:
            ids = dfs[2]["vs"] > 2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]
        elif name.find("_FR") > 0:
            ids = dfs[2]["vs"] < -2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]
        elif name.find("_FF") > 0:
            ids = dfs[2]["vf"] > 2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]
        elif name.find("_FB") > 0:
            ids = dfs[2]["vf"] < -2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]

        dfs[1].sort_values("frame", inplace=True)
        cw = dfs[1]["theta"] > 0
        if name.startswith("ccw_roc"):
            dfs_r1 = dfs[1][cw]
            dfs_l1 = dfs[1][~cw]
            xswl, xswr = dfs_l1[x], dfs_r1[x]
            wl, wr = len(xswl), len(xswr)
            xts = np.linspace(-5, 5, 100)
            TPR, FPR = [], []
            for xt in xts:
                hl_wl = np.count_nonzero(xswl < xt)
                hl_wr = np.count_nonzero(xswr < xt)
                TPR.append(hl_wl / wl)
                FPR.append(hl_wr / wr)
            fig_info = [FPR, TPR, [0, 1], [0, 1], 30, fly, "b"]
            plot_scatter_by_info(ax, fig_info)
            ax.axis("on")
            ret = ["scatter", len(bouts), fig_info]

            ax.set_xlabel("FPR", fontsize=FONT_SIZE)
            ax.set_ylabel("TPR", fontsize=FONT_SIZE)
            ax.plot([0, 1], [0, 1], "r")
        else:
            ret1 = [dfs[1][x], dfs[1][y], CENTER_WE_RANGE_X, CENTER_WE_RANGE_Y, 0.1, fly, cw]
            plot_scatter_by_info(ax, ret1, cmap="viridis", body_size=BODY_SIZE[1], alpha=0.1, line_color="k")
            ret = [fly, (CENTER_WE_RANGE_X, CENTER_WE_RANGE_Y), BODY_SIZE[fly], "k"]
    elif name == "toht":
        tot = ((dfs[2]["rel_pos_h:x"] > 0) & (dfs[1]["theta"] > 0)) | ((dfs[2]["rel_pos_h:x"] < 0) & (dfs[1]["theta"] < 0))
        ret1 = [dfs[2]["rel_pos:x"], dfs[2]["rel_pos:y"], CENTER_RANGE_X, CENTER_RANGE_X, 0.1, 2, tot]
        plot_scatter_by_info(ax, ret1, cmap="viridis", body_size=BODY_SIZE[1], alpha=0.1, line_color="k")
        ret = [fly, (CENTER_RANGE_X, CENTER_RANGE_X), BODY_SIZE[2], "k"]
    elif name == "wci_dist":
        x, y = "rel_pos_h:x", "rel_pos_h:y"
        ds = np.arange(2, 5, .1)
        dfss, _ = split_pair_dfs(dfs, bouts)
        wcis_l = []
        for dfs1 in dfss:
            dfs_l = dfs_only_we(dfs1, "we_l")
            dfs_r = dfs_only_we(dfs1, "we_r")
            wcis = []
            for d in ds:
                dfs_l1 = dfs_dist_range(dfs_l, d, d+.1)
                dfs_r1 = dfs_dist_range(dfs_r, d, d+.1)
                xswl, xswr = dfs_l1[1][x], dfs_r1[1][x]
                ipsi1 = np.count_nonzero(xswl < 0)
                ipsi2 = np.count_nonzero(xswr > 0)
                contra1 = len(xswl) - ipsi1
                contra2 = len(xswr) - ipsi2
                if len(xswl)+len(xswr) > 0:
                    wcis.append((ipsi1+ipsi2-contra1-contra2)/(len(xswl)+len(xswr)))
                else:
                    wcis.append(np.nan)
            wcis_l.append(wcis)
        py, err_l = lines_to_err_band(wcis_l)
        fig_info = [ds, py, err_l, [2, 5], [0, 0.7], "k", "-", None]
        plot_line_by_info(ax, fig_info)
        return ["line", len(bouts), fig_info]
    elif name.startswith("wc_"):
        if name.endswith("_head"):
            x, y = "rel_pos_h:x", "rel_pos_h:y"
        elif name.endswith("_center"):
            x, y = "rel_pos:x", "rel_pos:y"
        elif name.endswith("_tail"):
            x, y = "rel_pos_t:x", "rel_pos_t:y"

        if name.find("_FL") > 0:
            ids = dfs[2]["vs"] > 2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]
        elif name.find("_FR") > 0:
            ids = dfs[2]["vs"] < -2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]
        elif name.find("_FF") > 0:
            ids = dfs[2]["vf"] > 2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]
        elif name.find("_FB") > 0:
            ids = dfs[2]["vf"] < -2
            dfs = dfs[0][ids], dfs[1][ids], dfs[2][ids]

        dfs_l = dfs_only_we(dfs, "we_l")
        dfs_r = dfs_only_we(dfs, "we_r")
        if name.find("_nh") > 0:
            dfs_l = dfs_head_semicircle(dfs_l, True)
            dfs_r = dfs_head_semicircle(dfs_r, True)
        elif name.find("_nt") > 0:
            dfs_l = dfs_head_semicircle(dfs_l, False)
            dfs_r = dfs_head_semicircle(dfs_r, False)
        elif name.find("_far") > 0:
            dfs_l = dfs_far(dfs_l, True)
            dfs_r = dfs_far(dfs_r, True)
        elif name.find("_near") > 0:
            dfs_l = dfs_far(dfs_l, False)
            dfs_r = dfs_far(dfs_r, False)

        if name.startswith("wc_x"):
            r2 = plot_figure(ax, dfs_l, fly, "histo", x, color=COLOR_LEFT, bouts=bouts, limx=CENTER_WE_RANGE_X)
            r1 = plot_figure(ax, dfs_r, fly, "histo", x, color=COLOR_RIGHT, bouts=bouts, limx=CENTER_WE_RANGE_X)
            plot_shade(ax, (-FLY_AVG_WID/2, FLY_AVG_WID/2))
            ret = [r1, r2]
        elif name.startswith("wc_roc"):
            xswl, xswr = dfs_l[1][x], dfs_r[1][x]
            wl, wr = len(xswl), len(xswr)
            xts = np.linspace(-5, 5, 100)
            TPR, FPR = [], []
            for xt in xts:
                hl_wl = np.count_nonzero(xswl < xt)
                hl_wr = np.count_nonzero(xswr < xt)
                TPR.append(hl_wl / wl)
                FPR.append(hl_wr / wr)
            fig_info = [FPR, TPR, [0, 1], [0, 1], 30, fly, "b"]
            plot_scatter_by_info(ax, fig_info)
            ax.axis("on")
            ret = ["scatter", len(bouts), fig_info]

            ax.set_xlabel("FPR", fontsize=FONT_SIZE)
            ax.set_ylabel("TPR", fontsize=FONT_SIZE)
            ax.plot([0, 1], [0, 1], "k")
        else:
            if HEAT_CONFIG["fill_fig"]:
                ax.set_position([0, 0, 1, 1], which="both")
            # print_heat_xy_video(dfs_l[fly][x], dfs_l[fly][y], dfs_l, (-2.5, 2), postfix="l1_nt")
            # print_heat_xy_video(dfs_r[fly][x], dfs_r[fly][y], dfs_r, (-2.5, 2), postfix="r1_nt")
            # print_heat_xy_video(dfs_l[fly][x], dfs_l[fly][y], dfs_l, (-0.8, 2.8), postfix="l2_nt")
            # print_heat_xy_video(dfs_r[fly][x], dfs_r[fly][y], dfs_r, (-0.8, 2.8), postfix="r2_nt")
            # plot_ratio_heat(ax, dfs_l[fly][x], dfs_l[fly][y], dfs_r[fly][x], dfs_r[fly][y],
            #                 (CENTER_WE_RANGE_X, CENTER_WE_RANGE_Y))

            columns_f = [x, y, "frame"]
            columns_1 = ["we_r", "we_l"]
            dfs_rr = pd.concat([dfs_r[fly][columns_f], dfs_r[1][columns_1]], axis=1)
            dfs_ll = pd.concat([dfs_l[fly][columns_f], dfs_l[1][columns_1]], axis=1)
            dfs_c = pd.concat([dfs_rr, dfs_ll])
            dfs_c.sort_values("frame", inplace=True)
            xs = dfs_c[x]
            ys = dfs_c[y]
            colors = []
            for we_r in dfs_c["we_r"]:
                colors.append(COLOR_RIGHT if we_r > 0 else COLOR_LEFT)
            if fly == 1:
                ret1 = [xs, ys, CENTER_WE_RANGE_X, CENTER_WE_RANGE_Y, 5, fly, colors]
            else:
                ret1 = [xs, ys, CENTER_RANGE_X, CENTER_RANGE_X, .1, fly, colors]
            body_size = BODY_SIZE[fly]
            plot_scatter_by_info(ax, ret1, body_size, alpha=0.5, line_color="k")
            ret = [fly, (CENTER_RANGE_X, CENTER_RANGE_X), body_size, "k"]
            # r1 = plot_figure(ax, dfs_r, fly, "scatter", x, y, color=COLOR_FORWARD, bouts=bouts, limx=[-4, 4], limy=[0, 6])
            # r2 = plot_figure(ax, dfs_l, fly, "scatter", x, y, color=COLOR_RIGHT, bouts=bouts, limx=[-4, 4], limy=[0, 6])
            # ret = [r1, r2]
    elif name.startswith("c_we_ipsi") or name.startswith("c_we_contra"):
        if name.startswith("c_we_ipsi"):
            dfs_i = dfs_positive(dfs, "we_ipsi")
        else:
            dfs_i = dfs_negative(dfs, "we_ipsi")
        lim_heat = [0, 0.003]
        if name.endswith("_ccw"):
            dfs_i = dfs_negative(dfs_i, "theta")
            lim_heat = [0, 0.001]
        ret = plot_figure(ax, dfs_i, 2, "heat", "rel_pos:x", "rel_pos:y", bouts=bouts, lim_heat=lim_heat, limx=CENTER_RANGE_X, limy=CENTER_RANGE_X, count=count or len(dfs[0]))
    elif name == "near_dmin_hc":
        dfs_c = dfs_head_semicircle(dfs, True)
        r1 = plot_figure(ax, dfs_c, 2, "histo", "rel_polar_h:r", color=COLOR_FORWARD, bouts=bouts)
        dfs_c = dfs_head_semicircle(dfs, False)
        r2 = plot_figure(ax, dfs_c, 2, "histo", "rel_polar_h:r", color=COLOR_BACKWARD, bouts=bouts)
        ret = [r1, r2]
    elif name == "near_dmin_cc":
        dfs_c1 = dfs_head_semicircle(dfs, True)
        r1 = plot_figure(ax, dfs_c1, 1, "histo", "rel_polar:r", color=COLOR_FORWARD, bouts=bouts)
        d1 = r1[2][0]
        dfs_c2 = dfs_head_semicircle(dfs, False)
        r2 = plot_figure(ax, dfs_c2, 1, "histo", "rel_polar:r", color=COLOR_BACKWARD, bouts=bouts)
        d2 = r2[2][0]
        ax.set_xlabel("Distance $D^{McFc}$ (mm)", fontsize=FONT_SIZE)
        ax.set_ylim(0, 0.07)
        ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        ax.set_yticklabels(["0%", "", "2%", "", "4%", "", "6%", ""])
        ax.set_xticks([0, 3, 6])
        ax.set_xticklabels([0, 3, 6])
        # if USE_MEAN_PAIR:
        #     from scipy.stats import ks_2samp
        #     xs_l = split_pair1(dfs_c1[0]["pair"], dfs_c1[1]["rel_polar:r"])
        #     m_l1 = [np.nanmedian(xs) for xs in xs_l]
        #     xs_l = split_pair1(dfs_c2[0]["pair"], dfs_c2[1]["rel_polar:r"])
        #     m_l2 = [np.nanmedian(xs) for xs in xs_l]
        #     ax.set_title("n=%s p=%s" % (len(xs_l), ks_2samp(m_l1, m_l2)[1]))
        ret = [r1, r2]
    elif name == "near_dmin_fly":
        dfs_c = dfs_head_semicircle(dfs, False)
        xs = np.sort(dfs_c[1]["rel_polar:r"])
        ax.plot(xs, np.linspace(0, 1, len(xs)))
        ax.set_xscale("log")
        dfs_c = dfs_head_semicircle(dfs, True)
        xs = np.sort(dfs_c[1]["rel_polar:r"])
        ax.plot(xs, np.linspace(0, 1, len(xs)))
        ax.set_xscale("log")
        # xs_l, ys_l = split_pair(dfs_c[0]["pair"], dfs_c[1]["rel_polar:r"], dfs_c[1]["rel_polar:r"])
        # hist_l = [np.histogram(xs, range=[0, 6], bins=50) for xs in xs_l]
        # hist_f = np.mean([h[0] for h in hist_l], axis=0)
        # hist_b = hist_l[0][1][:-1]
        # info = [hist_b, hist_f, None, [0, 6], None, "k", "-", None]
        # plot_line_by_info(ax, info)
    elif name == "near_dmin":
        dfs_c = dfs_head_semicircle(dfs, True)
        r1 = plot_figure(ax, dfs_c, 1, "histo", "rel_polar_hh:r", color=COLOR_FORWARD, bouts=bouts)
        dfs_c = dfs_head_semicircle(dfs, False)
        r2 = plot_figure(ax, dfs_c, 1, "histo", "rel_polar_ht:r", color=COLOR_BACKWARD, bouts=bouts)
        ret = [r1, r2]
    elif name == "near_dmin_2":
        dfs_c = dfs_head_semicircle(dfs, False)
        r1 = plot_figure(ax, dfs_c, fly, "histo", "dist_MhFh", color=COLOR_BACKWARD, bouts=bouts, limx=(0, 7))
        dfs_c = dfs_head_semicircle(dfs, True)
        r2 = plot_figure(ax, dfs_c, fly, "histo", "dist_MhFt", color=COLOR_FORWARD, bouts=bouts, limx=(0, 7))
        ret = [r1, r2]
    elif name == "near_dmin_d" or name == "near_dmin_d_2":
        if name == "near_dmin_d":
            dfs_nh = dfs_head_semicircle(dfs, True)
            dfs_nt = dfs_head_semicircle(dfs, False)
        else:
            dfs_nh = dfs_head_semicircle(dfs, False)
            dfs_nt = dfs_head_semicircle(dfs, True)
        bins, range_x = 90, [0, 6]
        hist_f1, hist_b = np.histogram(dfs_nh[1]["rel_polar:r"], bins=bins, range=range_x)
        hist_f1 = hist_f1 / np.sum(hist_f1)
        hist_f2, hist_b = np.histogram(dfs_nt[1]["rel_polar:r"], bins=bins, range=range_x)
        hist_f2 = hist_f2 / np.sum(hist_f2)
        hist_f = hist_f1 - hist_f2
        hist_b = hist_b[:-1]
        info = [hist_b, hist_f, None, range_x, [-0.05, 0.05], "k", "-", None]
        plot_line_by_info(ax, info)
        ax.set_xlabel("Distance (mm)", fontsize=FONT_SIZE)
        ret = ["line", len(bouts), info]
    elif name == "fc_speed":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "v_len", bouts=bouts)
    elif name == "a_speed":
        ret = plot_figure(ax, dfs, fly, "adist", "v_len", color="k", bouts=bouts, lim_heat=[0, 12])
    elif name == "fc_speed_lim":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "lim20_v_len", bouts=bouts)
    elif name == "fc_abs_av":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "abs_av", bouts=bouts)
    elif name == "a_abs_av":
        ret = plot_figure(ax, dfs, fly, "adist", "abs_av", color="k", bouts=bouts, lim_heat=[0, 220])
    elif name == "fc_wing_m":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "wing_m", bouts=bouts)
    elif name == "a_wing_m":
        ret = plot_figure(ax, dfs, fly, "adist", "wing_m", color="k", bouts=bouts, lim_heat=[0, 100])
    elif name.startswith("fc_cw"):
        if name.endswith("nh"):
            dfs = dfs_head_semicircle(dfs, True)
        elif name.endswith("nt"):
            dfs = dfs_head_semicircle(dfs, False)
        ret = plot_weight_heat_by_weights(ax, dfs, dfs[1]["theta"] > 0, [0, 1], bouts, center_fly=fly,
                                          center_range=CENTER_RANGE if fly == 2 else (CENTER_T2_RANGE_X, CENTER_T2_RANGE_Y))
    elif name.startswith("c_cw") or name.startswith("c_ccw"):
        if name.startswith("c_cwf") or name.startswith("c_ccwf"):
            # dfs = dfs_filter_track_len(dfs, bouts, 90)
            dfs = dfs_far(dfs, True)
        if name.startswith("c_cw"):
            dfs_i = dfs_positive(dfs, "theta")
        else:
            dfs_i = dfs_negative(dfs, "theta")

        if name.endswith("nh"):
            dfs_i = dfs_head_semicircle(dfs_i, True)
        elif name.endswith("nt"):
            dfs_i = dfs_head_semicircle(dfs_i, False)
        elif name.endswith("qh"):
            dfs_i = dfs_head_quadrant(dfs_i)
        elif name.endswith("qt"):
            dfs_i = dfs_tail_quadrant(dfs_i)
        elif name.endswith("rh"):
            dfs_i = dfs_head_right_quadrant(dfs_i)
        elif name.endswith("lh"):
            dfs_i = dfs_head_left_quadrant(dfs_i)
        ret = plot_figure(ax, dfs_i, fly, "heat", "rel_pos:x", "rel_pos:y", bouts=bouts, lim_heat=[0, 0.005 if fly==1 else 0.003],
                          limx=CENTER_T2_RANGE_X if fly == 1 else CENTER_RANGE_X,
                          limy=CENTER_T2_RANGE_Y if fly == 1 else CENTER_RANGE_X, count=count or len(dfs[0]))
    elif name == "fc_we_lr":
        ret = plot_figure(ax, dfs, 1, "weight_heat", "we_lr", bouts=bouts, lim_heat=[0, 0.1])
    elif name == "fc_contact":
        HEAT_CONFIG["fc_contact"] = 1
        ret = plot_figure(ax, dfs, 0, "weight_heat", "overlap", bouts=bouts, lim_heat=[0, 0.1])
        HEAT_CONFIG["fc_contact"] = None
    elif name == "a_we_lr":
        ret = plot_figure(ax, dfs, 1, "adist", "we_lr", color="k", bouts=bouts, lim_heat=[0, 0.1])
    elif name == "a_contact":
        ret = plot_figure(ax, dfs, 0, "adist", "overlap", color="k", bouts=bouts, lim_heat=[0, 0.1])

    elif name == "fc_theta":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "theta", bouts=bouts)
    elif name == "fc_abs_theta":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "abs_theta", bouts=bouts, lim_heat=[50, 100])
    elif name == "a_theta":
        ret = plot_figure(ax, dfs, fly, "adist", "theta", color="k", bouts=bouts, lim_heat=[-100, 100])
    elif name == "a_abs_theta":
        ret = plot_figure(ax, dfs, fly, "adist", "abs_theta", color="k", bouts=bouts, lim_heat=[50, 100])
    elif name == "fc_acc":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "acc", bouts=bouts)
    elif name == "fc_acc_len":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "acc_len", bouts=bouts, lim_heat=[0.5, 2.5])
    elif name == "fc_acc_dir":
        ret = plot_weight_heat_by_weights(ax, dfs, lim_dir_a(dfs[1]["acc_dir"] - dfs[1]["dir"]), [-40, 40], bouts)
    elif name == "a_acc":
        ret = plot_figure(ax, dfs, fly, "adist", "acc", color="k", bouts=bouts, lim_heat=[-50, 50])
    elif name == "fc_abs_acc":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "abs_acc", bouts=bouts, lim_heat=[10, 30])
    elif name == "a_abs_acc":
        ret = plot_figure(ax, dfs, fly, "adist", "abs_acc", color="k", bouts=bouts, lim_heat=[0, 60])
    elif name == "fc_vf":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "vf", bouts=bouts)
    elif name == "a_vf":
        ret = plot_figure(ax, dfs, fly, "adist", "vf", color="k", bouts=bouts, lim_heat=[-5, 5])
    elif name == "fc_abs_vs":
        ret = plot_figure(ax, dfs, fly, "weight_heat", "abs_vs", bouts=bouts, lim_heat=[0, 9])
    elif name == "a_abs_vs":
        ret = plot_figure(ax, dfs, fly, "adist", "abs_vs", color="k", bouts=bouts, lim_heat=[0, 10])
    elif name == "fc_vs_nh":
        dfs_c = dfs_head_semicircle(dfs, True)
        ret = plot_figure(ax, dfs_c, fly, "weight_heat", "vs", bouts=bouts)
    elif name == "fc_vs_nt":
        dfs_c = dfs_head_semicircle(dfs, False)
        ret = plot_figure(ax, dfs_c, fly, "weight_heat", "vs", bouts=bouts)
    elif name == "fc_rel_dir":
        ret = plot_figure(ax, dfs, 0, "weight_heat", "rel_dir", bouts=bouts)
    elif name == "fc_v_ori":
        dfs_c = dfs_greater(dfs, "v_len", 3)
        v_ori = lim_ori_a(dfs_c[1]["v_dir"] - dfs_c[2]["dir"])
        ret = plot_weight_heat_by_weights(ax, dfs_c, v_ori, [-90, 90], bouts)
    elif name == "fc_abs_theta_ori":
        dfs_c = dfs_greater(dfs, "v_len", 3)
        v_ori = np.abs(lim_ori_a(dfs_c[1]["v_dir"] - dfs_c[1]["dir"]))
        ret = plot_weight_heat_by_weights(ax, dfs_c, v_ori, None, bouts)
    elif name == "fc_v_dir":
        ret = plot_weight_heat_by_weights(ax, dfs, lim_dir_a(dfs[1]["v_dir"] - dfs[2]["dir"]), [-100, 100], bouts)
    elif name == "a_v_dir":
        ts = dfs[2]["rel_polar:t"] - REL_POLAR_T_OFFSET
        weights = lim_dir_a(dfs[1]["v_dir"] - dfs[2]["dir"])
        ax.set_ylabel("v_dir", fontsize=FONT_SIZE)
        fig_info = plot_adist_by_info(ax, [ts, weights, 30, False, "k", [-100, 100]])
        ret = ["adist", len(bouts), fig_info]
    elif name.startswith("fc_to_"):  # NOTE: toh/(toh+tot)
        dfs_h = dfs_towards_head(dfs, True)
        dfs_t = dfs_towards_head(dfs, False)
        bin_h, hist_bx, hist_by = bin_data_2d(dfs_h[2]["rel_pos:x"], dfs_h[2]["rel_pos:y"],
                                        np.ones((len(dfs_h[2]), )), CENTER_RANGE, 60)
        bin_t, hist_bx, hist_by = bin_data_2d(dfs_t[2]["rel_pos:x"], dfs_t[2]["rel_pos:y"],
                                        np.ones((len(dfs_t[2]), )), CENTER_RANGE, 60)
        extent = [hist_bx[0], hist_bx[-1], hist_by[-1], hist_by[0]]
        lim_heat = [None, None]
        if name.endswith("_h"):
            bin_r = np.array([[len(binj) for binj in bini] for bini in bin_h])
        elif name.endswith("_t"):
            bin_r = np.array([[len(binj) for binj in bini] for bini in bin_t])
        else:
            bin_h = np.array([[len(binj) for binj in bini] for bini in bin_h])
            bin_t = np.array([[len(binj) for binj in bini] for bini in bin_t])
            bin_r = bin_h / (bin_h + bin_t)
            bin_r[(bin_h + bin_t) < HEAT_CONFIG["min_bin"]] = np.nan
            lim_heat = [0, 1]
        plot_heat_by_info(ax, bin_r, extent, lim_heat, 2, CENTER_RANGE, BODY_SIZE[2])
    elif name == "a_e_maj":
        ret = plot_figure(ax, dfs, fly, "adist", "e_maj", color="k", bouts=bouts)
    elif name == "a_c_dev":
        ys = dfs[1]["dir"] - vec_angle([dfs[2]["pos:x"] - dfs[1]["pos:x"], dfs[2]["pos:y"] - dfs[1]["pos:y"]])
        xs = dfs[2]["rel_polar:t"]
        fig_info = plot_adist_by_info(ax, [xs, ys, 30, False, "k", [-90, 90]])
        ret = ["adist", len(bouts), fig_info]
    elif name.startswith("a_d") or name.startswith("polar_d"):
        pname = "adist"
        limy = 5
        if name.startswith("polar"):
            pname = "polar_adist"
        if name.endswith("_d_ud"):
            dfs_c = dfs_towards_head(dfs, False)
            r2 = plot_figure(ax, dfs_c, 2, pname, "dist_MhFc", color=COLOR_SIDE_TO_TAIL, bouts=bouts)
            dfs_c = dfs_towards_head(dfs, True)  # "dist_MhFc"
            r1 = plot_figure(ax, dfs_c, 2, pname, "dist_MhFc", color=COLOR_SIDE_TO_HEAD, bouts=bouts)
            ret = [r1, r2]
            ax.set_xlabel("")
            limy = 3.5
        elif name.endswith("_nhnt"):
            dfs_c = dfs_head_semicircle(dfs, True)
            r1 = plot_figure(ax, dfs_c, 1, "polar_adist_h", "dist_MhFh", color=COLOR_FORWARD, bouts=bouts)
            dfs_c = dfs_head_semicircle(dfs, False)
            r2 = plot_figure(ax, dfs_c, 1, "polar_adist_h", "dist_MhFt", color=COLOR_BACKWARD, bouts=bouts)
            ret = [r1, r2]
            limy = 3
        else:
            # ret = plot_figure(ax, dfs, 0, pname, "dist_McFc", color="k", bouts=bouts)
            ret = plot_figure(ax, dfs, 2, pname, "dist_MhFc", color="k", bouts=bouts)
            # r3 = plot_figure(ax, dfs, 2, pname, "dist_MtFc", color="k", bouts=bouts)
            # ret = [r1, r2, r3]
        ax.set_ylim(0, limy)
        if name.startswith("polar"):
            ax.set_yticks([0, limy])
            ax.set_yticklabels(["0", "%.1fmm" % limy])
        else:
            ax.set_ylabel("Distance (mm)", fontsize=FONT_SIZE)
    elif name == "diff_a_d_ud":
        dfs_c_u = dfs_towards_head(dfs, True)
        dfs_c_d = dfs_towards_head(dfs, False)
        vs_l1, bin_l = bin_data_polar(dfs_c_u[0]["dist_McFc"], dfs_c_u[2]["rel_polar:t"] - REL_POLAR_T_OFFSET, bins=60)
        hist_f1 = np.array([np.median(vs) for vs in vs_l1])
        vs_l2, bin_l = bin_data_polar(dfs_c_d[0]["dist_McFc"], dfs_c_d[2]["rel_polar:t"] - REL_POLAR_T_OFFSET, bins=60)
        hist_f2 = np.array([np.median(vs) for vs in vs_l2])
        pval = [mannwhitneyu(vs1, vs2)[1] for vs1, vs2 in zip(vs_l1, vs_l2)]

        xs_l, ys_l = split_pair2(dfs_c_u[0]["pair"], dfs_c_u[2]["rel_polar_h:t"], dfs_c_u[0]["dist_McFc"])
        pair_l1 = [p.iloc[0] for p in split_pair1(dfs_c_u[0]["pair"], dfs_c_u[0]["pair"])]
        hist_f_l1 = []
        for xs, ys in zip(xs_l, ys_l):
            vs_l, bin_l = bin_data_polar(ys, xs, bins=60)
            hist_f_l1.append([np.nanmedian(vs) for vs in vs_l])
        hist_f_l1 = np.array(hist_f_l1)
        hist_f1 = np.nanmedian(hist_f_l1, axis=0)

        xs_l, ys_l = split_pair2(dfs_c_d[0]["pair"], dfs_c_d[2]["rel_polar_h:t"], dfs_c_d[0]["dist_McFc"])
        pair_l2 = [p.iloc[0] for p in split_pair1(dfs_c_d[0]["pair"], dfs_c_d[0]["pair"])]
        hist_f_l2 = []
        for xs, ys in zip(xs_l, ys_l):
            vs_l, bin_l = bin_data_polar(ys, xs, bins=60)
            hist_f_l2.append([np.nanmedian(vs) for vs in vs_l])
        hist_f_l2 = np.array(hist_f_l2)
        hist_f2 = np.nanmedian(hist_f_l2, axis=0)

        pval = []
        for i in range(len(bin_l)):
            pval.append(mannwhitneyu(hist_f_l1[:, i], hist_f_l2[:, i])[1])

        # hist_f = hist_f1 - hist_f2  # NOTE: difference of mean
        hist_f = []
        for pair, hist_f_l in zip(pair_l2, hist_f_l2):  # NOTE: mean of difference of pair
            try:
                idx = pair_l1.index(pair)
                hist_f.append(hist_f_l1[idx] - hist_f_l)
            except:
                pass
        hist_f = np.nanmedian(hist_f, axis=0)

        pval_c = pval_to_color(pval)

        cmap = "Reds"
        ax.scatter(bin_l, hist_f, c=pval_c, cmap=cmap, marker="o", s=30)

        points = np.array([bin_l, hist_f]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        t = pval_c[1:]
        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(t)  # [::-1]
        lc.set_linewidth(2)
        ax.add_collection(lc)

        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
        ax.set_xlabel("$\\alpha$", fontsize=FONT_SIZE)
        ax.set_ylabel("Difference (mm)", fontsize=FONT_SIZE)
        # fig_info = plot_adist(bin_l1, hist_f, "-", "gray", [0, 0.8], ax=ax)
        # plot_colorbar(ax, im=m)
        ret = ["line_cmap", len(bouts), [bin_l, hist_f, pval_c, cmap]]
    elif name == "c_pos" or name == "c_pos_p":
        if WEIGHT_HEAT_X.startswith("rel_polar"):
            ret = plot_figure(ax, dfs, fly, "heat", "rel_polar:t", "rel_polar:r", bouts=bouts, count=count)
        else:
            if FLY_NUM == 1:
                if fly == 2:
                    lim_heat = [0, 0.017]
                else:
                    lim_heat = [0, 0.04]
            ret = plot_figure(ax, dfs, fly, "heat", "rel_pos:x", "rel_pos:y", bouts=bouts, limx=CENTER_RANGE_X, limy=CENTER_RANGE_X, count=count, lim_heat=lim_heat)
    elif name.startswith("c_pos_w"):
        if HEAT_CONFIG["fill_fig"]:
            ax.set_position([0, 0, 1, 1], which="both")
        if name.startswith("c_pos_wl"):
            dfs_c = dfs_only_we(dfs, "we_l")
            lim_heat = [0, 0.003]
        else:
            dfs_c = dfs_only_we(dfs, "we_r")
            lim_heat = [0, 0.003]
        if name.endswith("_ccw"):
            dfs_c = dfs_negative(dfs_c, "theta")
        else:
            dfs_c = dfs_positive(dfs_c, "theta")
        ret = plot_figure(ax, dfs_c, fly, "heat", "rel_pos:x", "rel_pos:y", bouts=bouts, limx=CENTER_RANGE_X,
                          limy=CENTER_RANGE_X, lim_heat=lim_heat, count=count or len(dfs[0]))  # NOTE: use total count
    elif name == "c_pos_3d":
        ret = plot_figure(ax, dfs, fly, "heat_3d", "rel_pos:x", "rel_pos:y", bouts=bouts, lim_heat=[0, 0.004], count=count)
    elif name == "c_pos_nh":
        dfs_c = dfs_head_semicircle(dfs, True)
        ret = plot_figure(ax, dfs_c, fly, "heat", "rel_pos:x", "rel_pos:y", bouts=bouts, lim_heat=[0, 0.003], limx=CENTER_H_RANGE_X, limy=CENTER_H_RANGE_Y, count=count or len(dfs[0]))
    elif name == "c_pos_nt":
        dfs_c = dfs_head_semicircle(dfs, False)
        ret = plot_figure(ax, dfs_c, fly, "heat", "rel_pos:x", "rel_pos:y", bouts=bouts, lim_heat=[0, 0.012], limx=CENTER_H_RANGE_X, limy=CENTER_H_RANGE_Y[::-1] if fly == 2 else CENTER_H_RANGE_Y, count=count or len(dfs[0]))
    elif name == "c_pos_h":
        if WEIGHT_HEAT_X.startswith("rel_polar"):
            ret = plot_figure(ax, dfs, fly, "heat", "rel_polar_h:t", "rel_polar_h:r", bouts=bouts, count=count)
        else:
            ret = plot_figure(ax, dfs, fly, "heat", "rel_pos_h:x", "rel_pos_h:y", bouts=bouts, lim_heat=[0, 0.01], limx=CENTER_H_RANGE_X, limy=CENTER_H_RANGE_Y, count=count)
        # if fly == 2:
        #     plot_inner_limit(ax)
    elif name == "c_pos_h_nt":
        dfs_c = dfs_head_semicircle(dfs, False)
        ret = plot_figure(ax, dfs_c, fly, "heat", "rel_pos_h:x", "rel_pos_h:y", bouts=bouts, lim_heat=[0, 0.01], limx=CENTER_H_RANGE_X, limy=CENTER_H_RANGE_Y[::-1] if fly == 2 else CENTER_T_RANGE_Y, count=count or len(dfs[0]))
    elif name == "c_pos_h_nh":
        if lim_heat is None:
            lim_heat = [0, 0.01]
        dfs_c = dfs_head_semicircle(dfs, True)
        if HEAT_CONFIG["female_fast"] == 1:
            ids = dfs[2]["v_len"] >= 2
            dfs_c = dfs_c[0][ids], dfs_c[1][ids], dfs_c[2][ids]
        elif HEAT_CONFIG["female_fast"] == 0:
            ids = dfs[2]["v_len"] < 2
            dfs_c = dfs_c[0][ids], dfs_c[1][ids], dfs_c[2][ids]
        ret = plot_figure(ax, dfs_c, fly, "heat", "rel_pos_h:x", "rel_pos_h:y", bouts=bouts, lim_heat=lim_heat, limx=CENTER_T_RANGE_X, limy=CENTER_T_RANGE_Y, count=count or len(dfs[0]))
    elif name == "c_pos_t":
        ret = plot_figure(ax, dfs, fly, "heat", "rel_pos_t:x", "rel_pos_t:y", bouts=bouts, lim_heat=[0, 0.006 if fly == 2 else 0.015], limx=CENTER_T_RANGE_X, limy=CENTER_T_RANGE_Y, count=count)
    elif name == "c_pos_t_nh":
        dfs_c = dfs_head_semicircle(dfs, True)
        ret = plot_figure(ax, dfs_c, fly, "heat", "rel_pos_t:x", "rel_pos_t:y", bouts=bouts, lim_heat=[0, 0.003], limx=CENTER_T_RANGE_X, limy=CENTER_T_RANGE_Y, count=count or len(dfs[0]))
    elif name == "c_pos_t_nt":
        if lim_heat is None:
            lim_heat = [0, 0.015]
        dfs_c = dfs_head_semicircle(dfs, False)
        ret = plot_figure(ax, dfs_c, fly, "heat", "rel_pos_t:x", "rel_pos_t:y", bouts=bouts, lim_heat=lim_heat, limx=CENTER_T_RANGE_X, limy=CENTER_T_RANGE_Y, count=count or len(dfs[0]))
    elif name == "c_track":
        ret = plot_figure(ax, dfs, fly, "heat", "pos:x", "pos:y", bouts=bouts, lim_heat=lim_heat, count=count)
        ax.set_xlabel("%d %d" % (np.count_nonzero(dfs[1]["pos:x"]<10), np.count_nonzero(dfs[1]["pos:x"]>10)))
    elif name == "d_wing":
        r1 = plot_figure(ax, dfs, fly, "hist", "wing_l", color=COLOR_LEFT, bouts=bouts)
        r2 = plot_figure(ax, dfs, fly, "hist", "wing_r", color=COLOR_RIGHT, bouts=bouts)
        ret = [r1, r2]
    elif name == "d_speed":
        ret = plot_figure(ax, dfs, fly, "hist", "v_len", bouts=bouts)
    elif name == "d_rel_dir":
        ret = plot_figure(ax, dfs, 0, "hist", "rel_dir", bouts=bouts)
    elif name.startswith("d_rel_polar"):
        x = "rel_polar_h:t" if name.endswith("_h") else "rel_polar:t"
        fig_info = [dfs[2][x], [-180, 180], 50, False, "r", x, "-"]
        plot_hist_by_info(ax, fig_info)
        if "pair" not in dfs[0]:
            xp = [dfs[2][x]]
        else:
            xp = split_pair1(dfs[0]["pair"], dfs[2][x])
        lines = []
        for xpp in xp:
            hist_f, hist_b = np.histogram(xpp, bins=50, range=[-180, 180])
            lines.append(hist_f / np.sum(hist_f))
        py, err_l = lines_to_err_band(lines)
        bin_l = hist_b[:-1]
        name, fig_info = "line", [bin_l, py, err_l, [-180, 180], [0, 0.035], "k", "-", None]
        plot_line_by_info(ax, fig_info)

        # ax.set_ylim(0, 0.035)
        ax.set_yticks([0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
        ax.set_yticklabels(["0%", "", "1%", "", "2%", "", "3%", ""])
        ret = ["line", len(bouts), fig_info]
    elif name == "d_nhnt":
        x = "rel_polar_h:t"
        hist_f, hist_b = np.histogram(np.abs(dfs[2][x]), bins=4, range=[0, 180])
        ax.bar(hist_b[:-1]+22.5, hist_f, width=hist_b[1] - hist_b[0], color=color, edgecolor="w", alpha=0.5)
        s = np.sum(hist_f) / 100
        for i in range(4):
            ax.text(hist_b[i], hist_f[i], "%.2f%%" % (hist_f[i]/s))
    elif name == "d_nhnt_pair":
        pair = dfs[0]["pair"].to_frame()
        pair["data"] = dfs[1]["nh"]#dfs[2]["rel_polar_h:t"]
        grouped = pair.groupby("pair")
        nh_m = grouped.aggregate({"data": np.mean})
        count_m = grouped.aggregate({"data": lambda tt: len(tt)})
        ax.scatter(nh_m["data"].tolist(), count_m["data"].tolist(), s=1, c=color)
        ax.scatter([nh_m["data"].median(), (nh_m["data"]*count_m["data"]).sum()/count_m["data"].sum()], [0, 0], marker="x", color="rb")
        ax.set_xlabel("nh")
        ax.set_ylabel("frame count")
        # ax1 = ax.twinx()
        # ax1.set_ylabel("nh*frame")
        # ax1.hist(count_m["data"]*nh_m["data"], alpha=0.3)
    elif name == "d_av":
        ret = plot_figure(ax, dfs, fly, "hist", "av", bouts=bouts)
    elif name == "d_v":
        ret = plot_figure(ax, dfs, fly, "histo", "v_len", bouts=bouts)
    elif name == "d_theta":
        ret = plot_figure(ax, dfs, fly, "hist", "theta", bouts=bouts)
    elif name == "d_e_maj":
        ret = plot_figure(ax, dfs, fly, "hist", "e_maj", bouts=bouts)
    elif name == "d_e_min":
        ret = plot_figure(ax, dfs, fly, "hist", "e_min", bouts=bouts)
    elif name == "d_dist":
        ret = plot_figure(ax, dfs, 0, "hist", "dist_McFc", bouts=bouts)
    elif name == "d_dist_whole":
        fig_info = [dfs[0]["dist_McFc"], (0, 20), 90, False, "b", "dist_McFc", None]
        plot_hist_by_info(ax, fig_info)
    elif name == "d_span":
        span = np.abs(dfs[1]["rel_polar_hh:t"] - dfs[1]["rel_polar_ht:t"])
        fig_info = [span, (0, 180), 90, False, "b", "span", "-"]
        plot_hist_by_info(ax, fig_info)
        ret = ["hist", len(bouts), fig_info]
    elif name.startswith("d_pair"):
        if name.endswith("far"):
            dfs = dfs_far(dfs, True)
        elif name.endswith("near"):
            dfs = dfs_far(dfs, False)
        elif name.endswith("qh"):
            dfs = dfs_head_quadrant(dfs)
        if x.startswith("sd_"):
            xs_d = split_pair_d(dfs[0]["pair"], dfs[fly][x[3:]])
        else:
            xs_d = split_pair_d(dfs[0]["pair"], dfs[fly][x])
        xs_l = []
        # invalid_pair_l = get_m5_pairs("img/swarm/_center.csv")
        for pair, xs in xs_d.items():
            # if pair not in invalid_pair_l:
            if count_bouts(xs.index) > 3:  # NOTE: pairs with more than 5 bouts
                xs_l.append(xs)

                m = np.mean(xs)
                if m > 0.5:
                    info = dfs[1].loc[xs.index]
                    for idx, f in info.iterrows():
                        print("%s,%d,%d,%d" % (pair, f["frame"], f["frame"]+1, f["we_ipsi"]))
        # xs_l = split_pair1(dfs[0]["pair"], dfs[fly][x])
        if x.startswith("sd_"):
            m_l = [std_angle(xx, mean_angle(xx)) for xx in xs_l]
        else:
            m_l = [np.mean(xx) for xx in xs_l]
        fig_info = [m_l, (-1, 1), 100, False, COLOR_HEAD, "", "-"]
        plot_hist_by_info(ax, fig_info)
        ret = ["hist", len(bouts), fig_info]
    elif name.startswith("bar_nhnt"):
        we_lr_ll, t_ll = split_pair2(dfs[0]["pair"], dfs[fly][x], dfs[2]["rel_polar_h:t"])
        m_nh, m_nt = [], []
        for we_lr_l, t_l in zip(we_lr_ll, t_ll):
            nh_idx = np.abs(t_l) < 90
            nt_idx = ~nh_idx
            if len(we_lr_l[nh_idx]) > 0:# and len(we_lr_l[nt_idx]) > 0:
                m_nh.append(np.mean(we_lr_l[nh_idx]))
            if len(we_lr_l[nt_idx]) > 0:
                m_nt.append(np.mean(we_lr_l[nt_idx]))
        r1 = [m_nh, (0, 0.5), 100, False, COLOR_HEAD, "nh", "-"]
        plot_hist_by_info(ax, r1)
        r2 = [m_nt, (0, 0.5), 100, False, COLOR_TAIL, "nt", "-"]
        plot_hist_by_info(ax, r2)
        ret = [r1, r2]
    elif name == "t_polar_h":
        ret = plot_figure(ax, dfs, 2, "split_time", "rel_polar_h:t", bouts=bouts)
    elif name == "se_pos":
        if HEAT_CONFIG["fill_fig"]:
            ax.set_position([0, 0, 1, 1], which="both")
        dfs_c = dfs_start_end(dfs, bouts)
        fig_info1 = [dfs_c[:3][2]["rel_pos:x"], dfs_c[:3][2]["rel_pos:y"], CENTER_RANGE_X, CENTER_RANGE_X, 0.1, 2, COLOR_START]
        plot_scatter_by_info(ax, fig_info1, alpha=0.05)
        fig_info2 = [dfs_c[3:][2]["rel_pos:x"], dfs_c[3:][2]["rel_pos:y"], CENTER_RANGE_X, CENTER_RANGE_X, 0.1, 2, COLOR_END]
        plot_scatter_by_info(ax, fig_info2, BODY_SIZE[2], line_color="k", alpha=.05)
        ret = [2, (CENTER_RANGE_X, CENTER_RANGE_X), BODY_SIZE[2], "k"]
    elif name.startswith("d_se_polar"):
        dfs_c = dfs_start_end(dfs, bouts)
        x = "abs_rel_polar:t" if name.endswith("abs") else "rel_polar:t"
        x0 = 0 if name.endswith("abs") else -180
        y1 = 0.08 if name.endswith("abs") else 0.2
        bins = 15 if name.endswith("abs") else 30
        ret = []
        if not name.endswith("_end"):
            r1 = plot_figure(ax, dfs_c[:3], 2, "histo", x, color=COLOR_START, bouts=bouts, limx=[x0, 180], limy=[0, y1], bins=bins)
            ret.append(r1)
        if not name.endswith("_start"):
            r2 = plot_figure(ax, dfs_c[3:], 2, "histo", x, color=COLOR_END, bouts=bouts, limx=[x0, 180], limy=[0, y1], bins=bins)
            ret.append(r2)
        # labels = np.array([0, 90, 180, 270])
        # plt.thetagrids(labels, labels - 90)
        if len(ret) < 2:
            ret = ret[0]
    elif name == "c_cir_s":
        dfs_c = dfs_start_end(dfs, bouts)
        ret = plot_figure(ax, dfs_c[:3], fly, "heat", "rel_pos_h:x", "rel_pos_h:y", bouts=bouts)
    elif name == "c_cir_e":
        dfs_c = dfs_start_end(dfs, bouts)
        ret = plot_figure(ax, dfs_c[3:], fly, "heat", "rel_pos_h:x", "rel_pos_h:y", bouts=bouts)
    elif name.startswith("d_cir"):
        xs = dfs[0][dfs[fly]["circle"] > 0]["time"]
        fig_info = [xs, (0, ETHOGRAM_TIME_RANGE), 60, False, "k", "circle", None]
        plot_hist_by_info(ax, fig_info, need_norm=False)
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE)
        ax.set_ylabel("Frame count", fontsize=FONT_SIZE)
        ret = ["hist", len(bouts), fig_info]
    elif name == "r_theta_v":
        if HEAT_CONFIG["fill_fig"]:
            ax.set_position([0, 0, 1, 1], which="both")
        ret = plot_figure(ax, dfs, fly, "scatter", "theta", "v_len", limx=[-180, 180], limy=[0, 30], color=color)
    elif name == "r_wing_lr":
        ret = plot_figure(ax, dfs, fly, "scatter", "wing_l", "wing_r", limx=[-120, 10], limy=[-10, 120], color=color)
        print(np.corrcoef(dfs[fly]["wing_l"], dfs[fly]["wing_r"])[0][1])
    elif name.startswith("r_theta_wing") or name.startswith("heat_theta_wing"):
        if HEAT_CONFIG["fill_fig"]:
            ax.set_position([0, 0, 1, 1], which="both")
        if name.endswith("_m"):
            center_ids = (dfs[2]["dist_c"] < DIST_TO_CENTER_THRESHOLD_FEMALE) & (dfs[1]["dist_c"] < DIST_TO_CENTER_THRESHOLD_MALE)
            dfs_c = None, dfs[1][center_ids], None
        elif name.endswith("toh") or name.endswith("tot"):
            dfs_c = dfs_towards_head(dfs, name.endswith("h"))
        else:
            dfs_c = dfs_head_semicircle(dfs, name.endswith("nh"))
        if name.startswith("r_theta_wing"):
            if name.endswith("_m"):
                # min_bin=0
                dfs_cv = dfs_c[1][dfs_c[1]["v_len"] > 2]
                fig_info = plot_heat(ax, dfs_cv["theta"], dfs_cv["wing_m"], 0, [[-180, 180], [0, 90]], [0, 0.001])
                ax.add_line(plt.Line2D([-90, -90], [0, 90], linewidth=2, color="w", linestyle="--", zorder=1))
                ax.add_line(plt.Line2D([90, 90], [0, 90], linewidth=2, color="w", linestyle="--", zorder=1))
                ax.add_line(plt.Line2D([45, 45], [0, 90], linewidth=2, color="w", linestyle="--", zorder=1))
                ax.add_line(plt.Line2D([-45, -45], [0, 90], linewidth=2, color="w", linestyle="--", zorder=1))
                ax.set_ylim(0, 90)
            else:
                ret = plot_figure(ax, dfs, fly, "scatter", "theta", "wing_m", limx=[-180, 180], limy=[0, 120], color=color)
                return
            # fig_info1 = [dfs_c[1]["theta"], wing_mid, [-180, 180], [-100, 100], 0.001, 2, "k"]
            # plot_scatter_by_info(ax, fig_info1, alpha=0.1)
        else:
            wing_mid = (dfs_c[1]["wing_l"] + dfs_c[1]["wing_r"])
            fig_info = plot_heat(ax, dfs_c[1]["theta"], wing_mid, 0, [[-180, 180], [-90, 90]], [0, 0.005])

        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
        ax.set_yticks([-90, 0, 90])
        ax.set_yticklabels(["$-\pi/2$", 0, "$\pi/2$"])
        # ax.set_xlabel("$\\theta$ (CCW->CW)", fontsize=FONT_SIZE)
        # ax.set_ylabel("Wing (Left->Right)", fontsize=FONT_SIZE)
    elif name.startswith("r_d_v_we"):
        if name.endswith("nh"):
            dfs = dfs_head_semicircle(dfs, True)
        elif name.endswith("nt"):
            dfs = dfs_head_semicircle(dfs, False)
        dfs_c = dfs_non_zero(dfs, "we_ipsi")
        # fig_info1 = [dfs_c[0]["dist_McFc"], dfs_c[1]["v_len"], [0, 6], [0, 15], 0.01, 2, dfs_c[1]["we_ipsi"]]
        # fig_info1 = [dfs_c[2]["rel_polar:t"], dfs_c[2]["rel_polar:r"], [-180, 180], [0, 6], 0.01, 2, dfs_c[1]["we_ipsi"]]
        # plot_scatter_by_info(ax, fig_info1, cmap="jet")
        ax.set_xlabel("$D_{McFc}$", fontsize=FONT_SIZE)
        ax.set_ylabel("v", fontsize=FONT_SIZE)
        fig_info = plot_weight_heat(ax, dfs_c[0]["dist_McFc"], dfs_c[1]["v_len"], 0, [[0, 6], [0, 15]], [-1, 1], None, weights=dfs_c[1]["we_ipsi"])
        # fig_info = plot_weight_heat(ax, np.abs(dfs_c[1]["theta"]), dfs_c[1]["wing_m"], 0, [[0, 180], [0, 120]], [-1, 1], None, weights=dfs_c[1]["we_ipsi"])
        # fig_info = plot_weight_heat(ax, dfs_c[0]["dist_McFc"], np.abs(dfs_c[1]["theta"]), 0, [[0, 6], [0, 180]], [-1, 1], None, weights=dfs_c[1]["we_ipsi"])
        ret = ["weight_heat", len(bouts), fig_info]
    elif name == "fc_we_ipsi":
        dfs_c = dfs_non_zero(dfs, "we_ipsi")
        ret = plot_weight_heat_by_weights(ax, dfs_c, dfs_c[1]["we_ipsi"], [-1, 1], bouts)
    elif name.startswith("stream_dir") or name == "stream_v_dir" or name == "stream_acc_dir":
        move_to_y = False
        if name.endswith("_toh"):
            dfs = dfs_towards_head(dfs, True)
            name = "stream_dir"
            move_to_y = True
        elif name.endswith("_tot"):
            dfs = dfs_towards_head(dfs, False)
            name = "stream_dir"
            move_to_y = True
        ax.axis("off")
        df = dfs[fly]
        bin_2d, binx, biny = bin_data_2d(df["rel_pos:x"], df["rel_pos:y"], dfs[1][name[7:]] - dfs[2]["dir"], CENTER_RANGE2, STREAM_BINS)
        if name.endswith("v_dir"):
            bin_2d_len, binx, biny = bin_data_2d(df["rel_pos:x"], df["rel_pos:y"], dfs[1]["v_len"], CENTER_RANGE2, STREAM_BINS)
            lim_heat = HEAT_LIMIT[1].get("v_len")
        elif name.endswith("acc_dir"):
            bin_2d_len, binx, biny = bin_data_2d(df["rel_pos:x"], df["rel_pos:y"], dfs[1]["acc_len"], CENTER_RANGE2, STREAM_BINS)
            lim_heat = HEAT_LIMIT[1].get("acc_len")
        elif move_to_y:
            ox, oy = (binx[1] - binx[0]) / 2, (biny[1] - biny[0]) / 2
            ny_l = []
            for i, row in enumerate(bin_2d):
                if i == 0 or i == len(bin_2d) - 1:
                    continue
                for j, v in enumerate(row):
                    m, l = mean_angle(v) + 90, len(v)
                    if l > HEAT_CONFIG["min_bin"]:
                        x, y = binx[j], biny[i]
                        nx, ny = 0, y - x * np.tan(np.deg2rad(m))
                        if -5 < ny < 5:
                            ny_l.append(ny)
                        xa, ya = np.cos(np.deg2rad(m)), np.sin(np.deg2rad(m))
                        nx, ny = nx - xa * ox, ny - ya * oy
                        # rect = plt.Polygon(triangle_for_angle(m, nx, ny, ox), alpha=0.5, linewidth=0, zorder=10, color="b")
                        # ax.add_patch(rect)
            # ax.axis("equal")
            # if HEAT_CONFIG["fill_fig"]:
            #     ax.set_position([0.25, 0, 0.5, 1], which="both")
            # plot_fly_body(ax, 2, CENTER_RANGE2, BODY_SIZE[2], line_color="k", lw=0)
            # ax.set_xlim([-2, 2])
            # ax.set_ylim([-5, 5])
            ret = ny_l
            df_ny = pd.DataFrame(np.vstack([np.zeros((len(ny_l),)), ny_l]).T, columns=["x", "y"])
            sns.swarmplot(ax=ax, data=df_ny, x="x", y="y", size=3)
            ax.set_ylim([-5, 5])
            ax.axis("on")
            return ret
        else:
            bin_2d_len, lim_heat = None, [0, 60]
        ret = plot_stream(ax, bin_2d, binx, biny, bin_2d_len, lim_heat)
        ax.axis("equal")
        if HEAT_CONFIG["fill_fig"]:
            ax.set_position([0, 0, 1, 1], which="both")
        plot_fly_body(ax, 2, CENTER_RANGE2, BODY_SIZE[2], line_color="k")
        ax.set_xlim(CENTER_RANGE_X2)
        ax.set_ylim(CENTER_RANGE_X2)
    elif name.startswith("r_e_maj_dist") or name.startswith("r_far_e_maj_dist"):
        dfss, _ = split_pair_dfs(dfs, bouts)
        e_maj_l, dist_l = [], []
        for dfs1 in dfss:
            if name.find("far") > 0:
                dfs = dfs_far(dfs1)
            else:
                dfs = dfs_not_overlap(dfs1)
            if name.endswith("nh"):
                dfs = dfs_head_semicircle(dfs, True)
            elif name.endswith("nt"):
                dfs = dfs_head_semicircle(dfs, False)
            if len(dfs[1]) > HEAT_CONFIG["min_bin"]:
                e_maj_l.append(np.nanmean(dfs[1]["e_maj"]))
                dist_l.append(np.nanmean(dfs[2]["rel_polar_h:r"]))
        ax.scatter(e_maj_l, dist_l, s=1)
        ax.set_title(str(np.corrcoef(e_maj_l, dist_l)[0][1]))
        ax.set_xlabel("male body length")
        ax.set_ylabel("dist_MhFc")
        ax.set_xlim([1.5, 2.5])
        ax.set_ylim([0, 5])
    return ret

# if __name__ == '__main__':
#     import sys
#     from fpt_analysis import load_folder_dfs_bouts
#     geno = sys.argv[1]
#     path = os.path.join(GENO_DATA_DIR, geno)
#     # dfs = load_dfs(sys.argv[1])
#     PRINT_HEAT_XY_VIDEO = (-4.5, 2.5)
#     if False: # we
#         dfs, bouts, c = load_folder_dfs_bouts(path, postfix="_we_stat0.pickle")
#         dfs = dfs_no_touch(dfs) # TODO: NO edge
#     else:
#         dfs, bouts, c = load_folder_dfs_bouts(path, postfix="_cir_center_stat0.pickle")
#     # bouts = load_bouts(sys.argv[1])
#     # dfs = dfs_test(dfs, 0, 3200)
#     # dfs = dfs_circle(dfs)
#     # dfs = dfs_zero(dfs, "overlap")
#     # HEAT_CONFIG["min_bin"] = 10
#     # bouts = cir_bouts(bouts)
#     # plot_by_name(dfs, r"D:\exp\code\img\R76XShi-heat-Male-pos,x-pos,y")
#     # plot_by_name(dfs, r"D:\exp\code\img\R76XShi-weight_heat-Male-v_len")
#     # plot_by_name(dfs, r"D:\exp\code\img\OLD2-heat-female-rel_pos_t,x-rel_pos_t,y", bouts=bouts)
#     # plot_by_name(dfs, r"D:\exp\code\img\OLD2-polar_d-female", polar=True, bouts=bouts)
#     plot_by_name(dfs, os.path.join(DATA_DIR, "_img/%s-fc_wing_m" % geno), bouts=bouts)
#     # i1 = (dfs[2]["rel_polar_t:t"] > -5) & (dfs[2]["rel_polar_t:t"] < 5)
#     # i2 = (dfs[2]["rel_polar_t:t"] > 175) & (dfs[2]["rel_polar_t:t"] < 185)
#     # plt.hist(dfs[2]["rel_polar_t:r"][i1], bins=30)
#     # plt.hist(dfs[2]["rel_polar_t:r"][i2], bins=30)
#     # plt.show()
#
#     # from fpt_analysis import SUMMARY_FIGURES
#     # plot_summary_by_name(SUMMARY_FIGURES, dfs, "img/summary_calib", bouts, n=len(bouts))
