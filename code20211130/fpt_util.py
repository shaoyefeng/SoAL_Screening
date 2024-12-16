# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import sys

from fpt0_util import id_assign_dict, get_feat_header
from fpt_consts import FLY_NUM, FEMALE_AVG_LEN, FEMALE_AVG_LEN_L, FEMALE_AVG_LEN_H, DIST_TO_CENTER_THRESHOLD_MALE, \
    DIST_TO_CENTER_THRESHOLD_FEMALE, TURN_BINS, FINE_CIR_FEMALE_MAX_AV, FINE_CIR_FEMALE_MAX_V, REL_POLAR_T_OFFSET, \
    DIST_TO_FLY_FAR, DURATION_LIMIT, ALL_VIDEO_DIR

STATE_STATIC = 2

HALF_PEAK_PATH = "img/Model/half_peak.txt"
median_info = None
def get_median_info():
    global median_info
    if not median_info:
        median_info = load_dict(HALF_PEAK_PATH)
    return median_info

def find_file(folder, postfix):
    ret = []
    for name in os.listdir(folder):
        if name.endswith(postfix):
            ret.append(os.path.join(folder, name))
    return ret

def get_video_in_dir(video_dir):
    f = os.path.basename(video_dir)
    video = os.path.join(video_dir, f + "_t.avi")
    if os.path.exists(video):
        return video
    return os.path.join(video_dir, f + ".avi")

def rename_video_t(video):
    if video.endswith("_t.avi"):
        video_o = video.replace("_t.avi", ".avi")
        if os.path.exists(video):
            # if os.path.exists(video_o):
            #     os.rename(video_o, video + ".old")
            os.rename(video, video_o)
        return video_o
    return video

def pair_to_video_path(pair):
    _idx = pair.rfind("_")
    video_name, pair_no = pair[:_idx], pair[_idx+1:]
    # return os.path.join(r"G:/_video_syf/", video_name, video_name + ".avi"), video_name, pair_no, _idx
    for exp_folder in ALL_VIDEO_DIR:
        video_folder = os.path.join("E:/", exp_folder, video_name)
        if os.path.exists(video_folder):
            return os.path.join(video_folder, video_name + ".avi"), video_name, pair_no, _idx

def pair_to_video_path2(pair):
    _idx = pair.rfind("_")
    video_name, pair_no = pair[:_idx], pair[_idx+1:]
    return os.path.join("G:/_video_part/%s/%s.avi" % (video_name, video_name)), \
           os.path.join("G:/_video_part/%s/%s/%s_%s_meta.txt" % (video_name, pair_no, video_name, pair_no))

def pair_to_dfs_path(pair):
    return "E:/data_hrnet/geno_data/CS22/" + pair

def pair_name_str(name):
    for a, b in [("K", "2"), ("A", "1"), ("B", "2")]:
        name = name.replace(a, b)
    # if name[-2] == "_":
    #     name = name[:-1] + "0" + name[-1]
    return name

def get_real_fps(pair):
    camera = pair.split("_")
    if len(camera) < 2:
        return None
    if int(camera[0]) < 20191120:
        return None
    camera = camera[-2]
    if camera in ["1", "A"]:
        return 66.07
    elif camera in ["2", "B"]:
        return 66.33
    return None

def cmp_file(f1, f2):
    t1 = os.path.basename(f1).split("_")
    t2 = os.path.basename(f2).split("_")
    if int(t1[0]) != int(t2[0]):
        return int(t1[0]) - int(t2[0])
    return int(t1[1]) - int(t2[1])

def save_dict(filename, obj):
    f = open(filename, "w")
    json.dump(obj, f, indent=4)
    f.close()
    print("save_dict %s" % filename)

def load_dict(filename):
    if not os.path.exists(filename):
        return None
    # print("load_dict %s" % filename)
    f = open(filename, "r")
    j = json.load(f)
    f.close()
    return j

def save_dataframe(df, filename):
    print("save_df: ", filename)
    if filename.endswith(".pickle"):
        df.to_pickle(filename, compression="gzip")  # "infer", "gzip"
    else:
        df.to_csv(filename, index=False)

def load_dataframe(filename):
    pickle = filename.replace(".csv", ".pickle")
    if os.path.exists(pickle):
        print("load_df pickle: ", pickle)
        return pd.read_pickle(pickle, compression='gzip')
    if os.path.exists(filename):
        print("load_df: ", filename)
        return pd.read_csv(filename)
    return None

def calc_bouts(a, find_v=1):
    s = -1
    ret = []
    end = len(a) - 1
    for i, v in enumerate(a):
        condition = v == find_v
        if condition:
            if s < 0:
                s = i
        if not condition or i == end:
            if i == end:
                i += 1
            if s >= 0:
                ret.append((s, i))
            s = -1
    return ret

def count_bouts(idx_l, cont=1):
    d = np.diff(idx_l)
    return np.count_nonzero(d > cont)


bouts_key_l = "copulate acp_bouts ac_bouts cir_bouts1 we_on_l_bouts we_on_r_bouts fabl_bouts fabr_bouts fmotion_bouts engaged_bouts".split()[::-1]

def bouts_to_seq(cir_meta, bouts_key):
    a = np.zeros((cir_meta["total_frame"],))
    ids = dfs_bouts_idx(cir_meta.get(bouts_key, []))
    a[ids] = 1
    return a

def bouts_to_seq_with_window(cir_meta, bouts_key, window, ratio=0.5, a=None, label=1):
    bouts = cir_meta.get(bouts_key, [])
    start_before = window * (1 - ratio)
    end_before = window * ratio
    bw = []
    for s, e in bouts:
        ss, ee = s - start_before, e - end_before
        if ss > 0 and ee > s:
            bw.append([ss, ee])
    if a is None:
        a = np.zeros((cir_meta["total_frame"],))
    ids = dfs_bouts_idx(bouts)
    a[ids] = label
    return a

def bouts_start_seq(cir_meta, bouts_key, frames=0):
    bouts = cir_meta.get(bouts_key, [])
    idb = np.zeros((frames or cir_meta["total_frame"],), dtype=bool)
    idb[[s for s, e in bouts]] = True
    return idb

def combine_bouts(cir_meta):
    a = np.zeros((cir_meta["total_frame"],))
    for i, bouts_key in enumerate(bouts_key_l):
        ids = dfs_bouts_idx(cir_meta.get(bouts_key, []))
        a[ids] = i + 1
    # fabr_bouts_key_idx = bouts_key_l.index("fabr_bouts")
    # a[a == fabr_bouts_key_idx] = fabr_bouts_key_idx - 1
    return a

def combine_bouts_multi(cir_meta):
    sh = (cir_meta["total_frame"],)
    al = []
    for i, bouts_key in enumerate(bouts_key_l):
        ids = dfs_bouts_idx(cir_meta.get(bouts_key, []))
        a = np.zeros(sh)
        a[ids] = i + 1
        al.append(a)
    return np.array(al)

g_meta = {}
def get_meta(video_or_meta):
    meta_file = video_or_meta.replace(".avi", "_meta.txt")
    print("get_meta:", meta_file)
    meta = g_meta.get(meta_file)
    if not meta:
        meta = load_dict(meta_file)
        g_meta[meta_file] = meta
    return meta

def load_feat_csv(feat_file, infer_male=True):
    if feat_file.endswith("_feat_correct.csv"):
        meta = load_dict(feat_file.replace("feat_correct.csv", "meta.txt"))
    else:
        meta = load_dict(feat_file.replace("feat.csv", "meta.txt"))
    featc_df = pd.read_csv(feat_file, nrows=DURATION_LIMIT * int(meta["FPS"] + 0.5) * 2)
    if "2:pos:x" in featc_df.keys() and np.isnan(featc_df["2:pos:x"][0]):
        # NOTE: fpt_1 output, 1 fly per row
        featc_df.sort_values("frame", inplace=True)
        featc = featc_df.to_dict(orient="records")
        featc = id_assign_dict(featc)
        featc_df = pd.DataFrame(featc)
    if infer_male:
        featc_df = correct_id_by_stat(featc_df, feat_file.replace("_feat.csv", "_stat0.pickle"))
    # save_dataframe(featc_df, feat_file.replace("feat.csv", "feat_ida.pickle"))
    return meta, featc_df

def correct_id_by_stat(df, stat0_pickle):
    if not os.path.exists(stat0_pickle):
        return df
    stat0 = load_dataframe(stat0_pickle)
    court_infer_male = stat0["court_infer_male"]

    frames = len(df)
    for last_male in court_infer_male:
        if last_male > 0:
            break
    i_l = []
    for i in range(frames):
        male = court_infer_male[i]
        if male != 0:
            last_male = male
        if last_male == 2:
            i_l.append(i)
    for k in df.keys():
        if k.startswith("1:"):
            k1 = k.replace("1:", "2:")
            df.loc[i_l, k], df.loc[i_l, k1] = df.loc[i_l, k1], df.loc[i_l, k]
    return df

def load_feat_pickle(feat_file):
    meta = load_dict(feat_file.replace("feat_ida.pickle", "meta.txt"))
    featc_df = load_dataframe(feat_file)
    return meta, featc_df

def load_feat(feat_file, need_extra=True):
    if feat_file.endswith("pickle"):
        meta, df = load_feat_pickle(feat_file)
    else:
        meta, df = load_feat_csv(feat_file)
    if not need_extra:
        return df, meta
    for i in range(1, 1 + FLY_NUM):
        df["%d:point:xs" % i] = df["%d:point:xs" % i].map(parse_float_list)
        df["%d:point:ys" % i] = df["%d:point:ys" % i].map(parse_float_list)

        headx = df["%d:point:xs" % i].map(lambda t: t[0])
        heady = df["%d:point:ys" % i].map(lambda t: t[0])
        tailx = df["%d:point:xs" % i].map(lambda t: t[2])
        taily = df["%d:point:ys" % i].map(lambda t: t[2])
        body_dir_v = headx - tailx, heady - taily
        df["%d:dir" % i] = np.rad2deg(np.arctan2(body_dir_v[1], body_dir_v[0]))
    return df, meta

def save_feat(filename, feat):
    df = pd.DataFrame(feat)
    keys = get_feat_header()
    for i in range(1, 1 + FLY_NUM):
        df["%d:point:xs" % i] = df["%d:point:xs" % i].map(lambda t: " ".join([str(tt) for tt in t]))
        df["%d:point:ys" % i] = df["%d:point:ys" % i].map(lambda t: " ".join([str(tt) for tt in t]))
    return df.to_csv(filename, index=False, columns=keys)

def load_dfs(feat_file, need_abs_and_scale=False, only=None):  # NOTE: stat0.pickle|pair_folder...
    if os.path.isdir(feat_file):
        b = os.path.basename(feat_file)
        if len(b) <= 2:
            pp = os.path.basename(os.path.dirname(feat_file))
            prefix = os.path.join(feat_file, "%s_%s" % (pp, b))
        else:
            prefix = os.path.join(feat_file, os.path.basename(feat_file))
    else:
        prefix = feat_file[:feat_file.rfind("_")]
    if only is None:
        stat0 = load_dataframe(prefix + "_stat0.pickle")
        stat1 = load_dataframe(prefix + "_stat1.pickle")
        stat2 = load_dataframe(prefix + "_stat2.pickle")
        dfs = [stat0, stat1, stat2]
    else:
        dfs = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
        if only == 3:
            dfs[1] = load_dataframe(prefix + "_stat1.pickle")
            dfs[2] = load_dataframe(prefix + "_stat2.pickle")
        else:
            dfs[only] = load_dataframe(prefix + "_stat%d.pickle" % only)
    if need_abs_and_scale:
        return dfs_scale_pos(dfs_calc_abs(dfs))
    return dfs

def load_meta(pair_folder):
    prefix = os.path.join(pair_folder, os.path.basename(pair_folder))
    cir_meta_file = prefix + "_cir_meta.txt"
    return load_dict(cir_meta_file)

g_dfs_meta_cache = {}
def load_dfs_meta_cache(pair_folder, cache=True):
    ret = g_dfs_meta_cache.get(pair_folder)
    if ret:
        return ret

    prefix = os.path.join(pair_folder, os.path.basename(pair_folder))
    cir_meta_file = prefix + "_cir_meta.txt"
    cir_meta = load_dict(cir_meta_file)
    dfs = load_dfs(pair_folder)
    from fpt_frame_stat import detect_interaction
    detect_interaction(cir_meta, dfs, cir_meta_file)
    if cache and len(g_dfs_meta_cache) < 45:
        g_dfs_meta_cache[pair_folder] = [dfs, cir_meta]
    return dfs, cir_meta

def remove_dfs_meta_cache(pair_folder):
    ret = g_dfs_meta_cache.get(pair_folder)
    if ret:
        del g_dfs_meta_cache[pair_folder]

def dfs_calc_abs(dfs):
    for k in ["acc", "av", "theta", "v_dir", "vs"]:
        dfs[1]["abs_" + k] = dfs[1][k].abs()
    for k in ["rel_pos_h:x", ]:
        dfs[2]["abs_" + k] = dfs[2][k].abs()
    # dfs[1]["nh"] = np.array(dfs[1]["rel_polar_hh:r"] < dfs[1]["rel_polar_ht:r"]).astype(int)
    d = np.abs(dfs[2]["rel_polar_h:t"] - REL_POLAR_T_OFFSET)
    dfs[1]["nh"] = (d < 45).astype(int)
    dfs[1]["nt"] = (d > 135).astype(int)
    # dfs[2]["nh"] = dfs[2]["rel_polar_hh:r"] < dfs[2]["rel_polar_ht:r"]
    return dfs

def dfs_scale_pos(dfs):
    # fly1_sc = dfs[1]["e_maj"].clip(FEMALE_AVG_LEN_L, FEMALE_AVG_LEN_H) / FEMALE_AVG_LEN
    fly2_sc = dfs[2]["e_maj"].clip(FEMALE_AVG_LEN_L, FEMALE_AVG_LEN_H) / FEMALE_AVG_LEN
    for k in ["rel_pos:x", "rel_pos:y", "rel_polar:r",
              "rel_pos_h:x", "rel_pos_h:y", "rel_polar_h:r",
              "rel_pos_t:x", "rel_pos_t:y", "rel_polar_t:r"]:
        dfs[1][k] = dfs[1][k] / fly2_sc  # NOTE: use female length
        dfs[2][k] = dfs[2][k] / fly2_sc
    return dfs

def save_dfs(dfs, prefix):  #.../20190902_174522_B_2_
    save_dataframe(dfs[0], prefix + "_stat0.pickle")
    save_dataframe(dfs[1], prefix + "_stat1.pickle")
    save_dataframe(dfs[2], prefix + "_stat2.pickle")

def load_bouts(path):
    if path.endswith(".txt"):
        return load_dict(path)["cir_bouts1"]
    if path.endswith("feat.pickle"):
        meta = load_dict(path.replace("feat.pickle", "cir_meta.txt"))
        return meta["cir_bouts1"]
    if path.endswith("feat.csv"):
        meta = load_dict(path.replace("feat.csv", "cir_meta.txt"))
        return meta["cir_bouts1"]
    meta_file = find_file(path, "_cir_meta.txt")[0]
    meta = load_dict(meta_file)
    return meta["cir_bouts1"]

def squeeze_cir_bouts(bouts):
    ret = []
    f = 0
    for s, e in bouts:
        l = e - s
        ret.append([f, f + l])
        f += l
    return ret

stat_keys_simple = ["1:circle", "1:circle_s1", "1:walk", "1:sidewalk_s", "0:dist_McFc", "0:court_as_male", "0:court_infer_male",
                    "1:wing_l", "1:wing_r", "1:court_s_30", "1:court_s", "1:on_edge", "0:overlap", "0:copulate",
                    "1:dir", "1:d_dir", "1:v_len", "1:dist_ht", "1:we_ipsi",
                    "1:pos:x", "1:pos:y", "1:point:xs", "1:point:ys",
                    "2:pos:x", "2:pos:y", "2:point:xs", "2:point:ys"]
def load_simple(filename):
    return load_dataframe(filename)
    # df.rename({a: a[2:] for a in stat_keys_simple})
    # return df, df, df

def save_simple(filename, stats):
    ret = pd.DataFrame()
    for k in stat_keys_simple:
        dfs_i = int(k[0])
        key = k[2:]
        if key in stats[dfs_i]:
            ret[k] = stats[dfs_i][key]
    save_dataframe(ret, filename)
    return ret

def dfs_circle(dfs):
    return dfs_positive(dfs, "circle")

def dfs_we(dfs):
    ids = (dfs[1]["court"] > 0) & (dfs[2]["dist_c"] < DIST_TO_CENTER_THRESHOLD_MALE) & (dfs[1]["dist_c"] < DIST_TO_CENTER_THRESHOLD_FEMALE)
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_only_we(dfs, key):
    key_o = "we_l" if key == "we_r" else "we_r"
    ids = (dfs[1][key] > 0) & (dfs[1][key_o] <= 0)
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_we_min_pickle(dfs):
    male_keys = ["rel_pos:x", "rel_pos:y", "rel_pos_h:x", "rel_pos_h:y", "rel_pos_t:x", "rel_pos_t:y",
                 "we_l", "we_r", "frame", "e_maj", "e_min", "we_ipsi"]
    female_keys = ["rel_pos:x", "rel_pos:y", "rel_polar_h:r", "rel_polar_h:t", "frame", "e_maj", "e_min"]
    return dfs[0][["overlap",]], dfs[1][male_keys], dfs[2][female_keys]

def dfs_pred_min_pickle(dfs):
    male_keys = ["rel_pos_h:x", "rel_pos_h:y", "rel_polar_h:r", "rel_polar_h:t", "rel_polar:r", "rel_polar:t",
                 "rel_pos_t:x", "rel_pos_t:y", "rel_polar_t:r", "rel_polar_t:t", "we_ipsi",
                 "theta", "wing_l", "wing_r", "we_l", "we_r", "circle", "court", "dist_c"]
    female_keys = ["rel_polar_h:r", "rel_polar_h:t", "rel_pos_h:x", "rel_pos_h:y", "dist_c"]
    return pd.DataFrame(), dfs[1][male_keys], dfs[2][female_keys]

def dfs_positive(dfs, key): # we_l, circle
    ids = dfs[1][key] > 0
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_negative(dfs, key): # we_r
    ids = dfs[1][key] < 0
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_non_zero(dfs, key):
    ids = dfs[1][key] != 0
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_not_overlap(dfs):  # overlap
    ids = dfs[0]["overlap"] != 0
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_greater(dfs, key, v):
    ids = dfs[1][key] > v
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

# def dfs_near_head(dfs, near_head):
#     if near_head:
#         # return dfs_head(dfs)
#         ids = dfs[1]["rel_polar_hh:r"] < dfs[1]["rel_polar_ht:r"]
#     else:
#         # return dfs_tail(dfs)
#         ids = dfs[1]["rel_polar_hh:r"] > dfs[1]["rel_polar_ht:r"]
#     return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_head_semicircle(dfs, near_head):
    if near_head:
        ids = (np.abs(dfs[2]["rel_polar_h:t"] - REL_POLAR_T_OFFSET) < 90)# & (np.abs(dfs[1]["rel_polar_h:t"]) < 90)
    else:
        ids = (np.abs(dfs[2]["rel_polar_h:t"] - REL_POLAR_T_OFFSET) > 90)# & (np.abs(dfs[1]["rel_polar_h:t"]) > 90)
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

# def dfs_near_head2(dfs, near_head):
#     if near_head:
#         ids = (np.abs(dfs[2]["rel_polar_h:t"] - 90) < 90)# & (np.abs(dfs[1]["rel_polar_h:t"] - 90) < 90)
#     else:
#         ids = (np.abs(dfs[2]["rel_polar_h:t"] - 90) > 90)# & (np.abs(dfs[1]["rel_polar_h:t"] - 90) > 90)
#     return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_head_quadrant(dfs):
    ids = np.abs(dfs[2]["rel_polar_h:t"] - REL_POLAR_T_OFFSET) < 45
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_head_right_quadrant(dfs):
    ids = (dfs[2]["rel_polar_h:t"] < 0) & (dfs[2]["rel_polar_h:t"] > -90)
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_head_left_quadrant(dfs):
    ids = (dfs[2]["rel_polar_h:t"] > 0) & (dfs[2]["rel_polar_h:t"] < 90)
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_lateral(dfs):
    d = np.abs(dfs[2]["rel_polar_h:t"] - REL_POLAR_T_OFFSET)
    ids = (d > 45) & (d < 135)
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_tail_quadrant(dfs):
    ids = np.abs(dfs[2]["rel_polar_h:t"] - REL_POLAR_T_OFFSET) > 135
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_inner(dfs, d):
    ids = dfs[2]["rel_polar_h:r"] <= d
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_far(dfs, is_far=True):
    if is_far:
        ids = dfs[2]["rel_polar_h:r"] > DIST_TO_FLY_FAR
    else:
        ids = dfs[2]["rel_polar_h:r"] <= DIST_TO_FLY_FAR
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_dist_range(dfs, near, far):
    ids = (dfs[2]["rel_polar:r"] > near) & (dfs[2]["rel_polar:r"] < far)
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_no_touch(dfs):
    ids = dfs[0]["reg_n"] > 1
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_non_static(dfs):
    ids = dfs[1]["walk"] != STATE_STATIC
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_ts_range(dfs, ts, te):
    ids = (dfs[0]["time"] > ts) & (dfs[0]["time"] < te)
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_towards_head(dfs, towards_head):
    ids = ((dfs[2]["rel_pos_h:x"] > 0) & (dfs[1]["theta"] > 0)) | ((dfs[2]["rel_pos_h:x"] < 0) & (dfs[1]["theta"] < 0))
    if towards_head:
        ids = ~ids
    # head_15 = abs(dfs[2]["rel_polar:t"]) < 15
    # tail_15 = abs(dfs[2]["rel_polar:t"]) > 165
    # ids = ids & (dfs[1]["v_len"] > 3) & (dfs[2]["v_len"] < 1) #& ~head_15 & ~tail_15
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_towards_head2(dfs, towards_head):
    ccw = dfs[1]["theta"] < 0#dfs[2]["rel_polar:t"].diff() > 0#
    right = dfs[2]["rel_pos:x"] > 0
    ids = (ccw & ~right) | (~ccw & right)
    if towards_head:
        ids = ~ids
    ids = ids & (dfs[1]["v_len"] > 3) & (dfs[2]["v_len"] < 1) #& ~head_15 & ~tail_15
    return dfs[0][ids], dfs[1][ids], dfs[2][ids]

def dfs_bouts_idx(bouts):
    ids = []
    for s, e in bouts:
        ids.extend(range(s, e))
    return ids

def dfs_bouts(dfs, bouts):
    ids = []
    for s, e in bouts:
        ids.extend(range(s, e))
    return dfs[0].iloc[ids], dfs[1].iloc[ids], dfs[2].iloc[ids]

def dfs_start_end(dfs, bouts):
    sl, el = [], []
    ext_frames = 5
    for s, e in bouts:  # NOTE: s + 5 < e
        sl.extend(range(s, s + ext_frames))
        el.extend(range(e - ext_frames, e))
    # return dfs[0].iloc[sl], dfs[1].iloc[sl], dfs[2].iloc[sl], dfs[0].iloc[el], dfs[1].iloc[el], dfs[2].iloc[el]
    return dfs[0].loc[sl], dfs[1].loc[sl], dfs[2].loc[sl], dfs[0].loc[el], dfs[1].loc[el], dfs[2].loc[el]

def dfs_mid_part(dfs, bouts, start_r, count):
    sl = []
    for s, e in bouts:
        l = e - s
        ss = int(s + l * start_r)
        ee = ss + count #s + l * end_r
        if ee > e:
            ee = e#continue #
            ss = ee - count
        sl.extend(range(ss, ee))
    return dfs[0].loc[sl], dfs[1].loc[sl], dfs[2].loc[sl]

def parse_float_list(xs):
    return [float(x) for x in xs.split()]

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# def load_dir(folder):
#     df = pd.DataFrame()
#     for f in os.listdir(folder):
#         ff = os.path.join(folder, f)
#         if os.path.isdir(ff):
#             for f2 in os.listdir(ff):
#                 if f2.endswith(".csv"):
#                     ff2 = os.path.join(ff, f2)
#                     df_1 = pd.read_csv(ff2)
#                     if len(df_1) > 0:
#                         df_1["dtc"] = track_dist_to_center(df_1)
#                         df = df.append(df_1)
#     return df, df_to_track_info(df)

def df_to_track_info(df):
    return df.to_dict(orient="records")

def traverse_folder(folder, cb):
    if os.path.isdir(folder):
        i = 0
        for name in os.listdir(folder):
            if not name.endswith("remove"):
                d = os.path.join(folder, name)
                if os.path.isdir(d):#and i > 30 and i < 40:
                    cb(d)
                i += 1

def traverse_file(folder, cb):
    for name in os.listdir(folder):
        d = os.path.join(folder, name)
        if os.path.isdir(d):
            if not name.endswith("remove"):
                traverse_file(d, cb)
        else:
            cb(d)

def get_all_feat_file(exp_folder):
    ret = []
    if os.path.basename(exp_folder).startswith("video"):
        for sub in sub_folders(exp_folder):
            ret.extend(get_all_feat_file(sub))
    else:
        subs = sub_folders(exp_folder)
        if not subs:
            subs = [exp_folder]
        for sub in subs:
            fs = find_file(sub, "feat.csv")
            if len(fs):
                for f in fs:
                    if not f.endswith("calib_feat.csv"):
                        ret.append(f)
                        break
            else:
                print("!!! feat.csv not found in", sub)
    return ret

def get_all_cir_meta(geno_folder):
    if os.path.basename(geno_folder).startswith("20"):
        return find_file(geno_folder, "cir_meta.txt")
    ret = []
    for sub in sub_folders(geno_folder):
        fs = find_file(sub, "cir_meta.txt")
        if len(fs):
            ret.append(fs[0])
            # break
        else:
            print("!!! cir_meta.txt not found in", sub)
    return ret

def get_meta_file(feat_file):
    if feat_file.endswith("feat.pickle"):
        return feat_file.replace("feat.pickle", "meta.txt")
    if feat_file.endswith("feat.csv"):
        return feat_file.replace("feat.csv", "meta.txt")

def sub_folders(folder):
    ret = []
    for name in os.listdir(folder):
        d = os.path.join(folder, name)
        if os.path.isdir(d):
            ret.append(d)
    return ret

def get_all_pair_folder(ana_dir):
    if os.path.basename(ana_dir).startswith("20"):
        video_folder_l = [ana_dir]
    else:
        video_folder_l = sub_folders(ana_dir)
    pair_folder_l = []
    for video_folder in video_folder_l:
        pair_folder_l.extend(sub_folders(video_folder))
    return pair_folder_l

def mp_process_files(cb, files, pool_size=0):
    if pool_size:
        from multiprocessing import Pool
        p = Pool(pool_size)
        p.map(cb, files)
        p.close()
    else:
        for f in files:
            cb(f)

WRONG_ANGLE_CHANGE_MIN = 50
def correct_angle(v):
    # return np.array(v) - v[0]
    ret = []
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
        if abs(i1 - i3) < WRONG_ANGLE_CHANGE_MIN and abs(i2 - i3) > WRONG_ANGLE_CHANGE_MIN and abs(i2 - i1) > WRONG_ANGLE_CHANGE_MIN:
            ret[j] = (i1 + i3) / 2
    return ret


def correct_wing(wing_l):
    # NOTE: tracking error: 60-0-60
    # return wing_l
    thresh = 45
    d1 = np.diff(wing_l)
    d1_1 = np.roll(d1, -1)
    idx = ((d1 <= -thresh) & (d1_1 >= thresh)) | ((d1 >= thresh) & (d1_1 <= -thresh))
    idx_1 = np.roll(idx, -1)
    ret = np.array(wing_l)
    ret[np.concatenate([[False], idx])] = ret[np.concatenate([[False], idx_1])]
    return ret

def redirect_output(filename):
    return
    f = open(filename, "w")
    sys.stdout = f
    sys.stderr = f

def center_img(img_gray, center, theta, out_shape, out_shape_w, fill_gray=255):
    from scipy import ndimage
    sub0 = int(center[1] - out_shape_w[1]/2)
    sub1 = int(center[0] - out_shape_w[0]/2)
    end0 = sub0 + out_shape_w[1]#img_gray.shape[0]+5
    end1 = sub1 + out_shape_w[0]#img_gray.shape[1]+5
    img_shift = img_gray[max(sub0, 0):end0, max(sub1, 0):end1]#ndimage.shift(img_gray, shift)
    s0, s1 = img_shift.shape
    if sub0 < 0:
        img_shift = np.vstack((np.full((-sub0, s1), fill_gray), img_shift))
    if end0 > img_gray.shape[0]:
        img_shift = np.vstack((img_shift, np.full((end0 - img_gray.shape[0], s1), fill_gray)))
    if sub1 < 0:
        img_shift = np.hstack((np.full((img_shift.shape[0], -sub1), fill_gray), img_shift))
    if end1 > img_gray.shape[1]:
        img_shift = np.hstack((img_shift, np.full((img_shift.shape[0], end1 - img_gray.shape[1]), fill_gray)))
    img_rotate = ndimage.rotate(img_shift, theta+90)
    shape_r = img_rotate.shape
    sub0 = int((shape_r[0] - out_shape[1]) / 2)
    sub1 = int((shape_r[1] - out_shape[0]) / 2)
    return img_rotate[sub0:(sub0 + out_shape[1]), sub1:(sub1 + out_shape[0])]

def lim_dir(dir1):
    if dir1 > 180:
        dir1 -= 360
    elif dir1 < -180:
        dir1 += 360
    return dir1

def lim_dir_a(a):
    a[a > 180] -= 360
    a[a < -180] += 360
    return a

def lim_ori_a(a):
    ori = lim_dir_a(a)
    ori[ori > 90] = ori[ori > 90] - 180
    ori[ori < -90] = ori[ori < -90] + 180
    return ori

def angle_diff(a1, a2, r=360):
    t = (a1 - a2) % r
    return t if t < r/2 else t - r

def angle_diff_a(a1, a2, r=360):
    t = (a1 - a2) % r
    t[t >= r/2] -= r
    return t

def angle_sec(angle):
    a = angle % 360
    w = 360 / TURN_BINS
    return int(a / w)

def calc_sections(angles, min_cont=3):
    # NOTE: sequence of sections moved across (return frame index)
    sections = np.array(list(map(angle_sec, angles)))
    last, cont = 0, 0
    ret = []
    secs_uniq = []
    last_secs_uniq = None
    for i, s in enumerate(sections):
        if s == last:
            cont += 1
        else:
            cont = 0
        if cont >= min_cont and last_secs_uniq != s:
            ret.append(i)
            secs_uniq.append(s)
            last_secs_uniq = s
        last = s
    return ret, secs_uniq

def calc_switch_sections(secs, secs_frames):
    if len(secs) <= 2:
        return []
    h = TURN_BINS / 2
    def cmp(s1, s2):
        if s1 > s2:
            if s1 > s2 + h:
                return 1
            return -1
        if s2 > s1:
            if s2 > s1 + h:
                return -1
            return 1
        return 0
    last_sec = secs[0]
    last_rot = cmp(secs[0], secs[1])
    ret = []
    for i, sec in enumerate(secs[1:]):
        rot = cmp(last_sec, sec)
        if rot != 0 and rot != last_rot:
            ret.append(secs_frames[i])
            last_rot = rot
        last_sec = sec
    return ret

def calc_switch_sections2(ts, m_v_len, f_v_len, f_av, frame_step, min_mv=2, min_deg=0):
    h = int(frame_step / 2)
    if len(ts) < h * 6:
        return []
    a = [0] * (h*3)
    for i in range(h*3, len(ts) - h*3):
        before, cur, after = np.mean(ts[i-h*3:i-h]), np.mean(ts[i-h:i+h]), np.mean(ts[i+h:i+h*3])
        cur_f_v = np.max(f_v_len[i-h*3:i+h*3])
        cur_f_av = np.max(np.abs(f_av[i-h*3:i+h*3]))
        cur_m_v = np.mean(np.abs(m_v_len[i-h*3:i+h*3]))
        if cur_m_v < min_mv or cur_f_av > FINE_CIR_FEMALE_MAX_AV or cur_f_v > FINE_CIR_FEMALE_MAX_V:
            a.append(0)
        else:
            if cur > (before+min_deg) and cur > (after+min_deg):
                a.append(1)
            elif cur < (before-min_deg) and cur < (after-min_deg):
                a.append(-1)
            else:
                a.append(0)
    bout1 = calc_bouts(a, 1)
    bout2 = calc_bouts(a, -1)
    ret = []
    for b in bout1:
        ret.append(b[0] + np.argmax(ts[b[0]:b[1]]))
    for b in bout2:
        ret.append(b[0] + np.argmin(ts[b[0]:b[1]]))
    return ret

def calc_switch_sections3(ts):
    from scipy import signal
    ret = list(signal.find_peaks(ts, prominence=(10, None), distance=5)[0])
    ret.extend(list(signal.find_peaks(-ts, prominence=(10, None), distance=5)[0]))
    return ret

def calc_jaaba_bouts(mat):
    import scipy.io as sio
    all_scores = sio.loadmat(mat)["allScores"]
    bouts1 = calc_bouts(all_scores["postprocessed"][0][0][0][0][0])
    bouts2 = calc_bouts(all_scores["postprocessed"][0][0][0][1][0])

    return bouts1, bouts2

def cross_correlate(xs, ys, ts, x, k):
    # NOTE: r < 0: xs prior
    from scipy.stats import zscore
    xs, ys = zscore(xs), zscore(ys)
    n = len(xs)
    ret = []
    # for i in -ts:
    #     xss = np.roll(xs, i)
    for t in ts:
        xss = x[k + "T%d" % t]
        ret.append(np.dot(zscore(xss), ys)/n)
    return ret

def cross_correlate2(xs, ys, ts, x, k):
    # NOTE: r < 0: xs prior
    ux, uy = xs.mean(), ys.mean()
    ox = np.sqrt(np.sum((xs-ux) ** 2))
    oy = np.sqrt(np.sum((ys-uy) ** 2))
    zy = (ys - uy) / oy

    n = len(xs)
    ret = []
    for t in ts:
        xss = x[k + "T%d" % t]
        ret.append(np.dot((xss - ux) / ox, zy))
    return ret

def cross_correlate_w(xs, ys, ts, window):
    # NOTE: r < 0: xs prior
    from scipy.stats import zscore
    xs, ys = zscore(xs), zscore(ys)
    n = len(xs)
    ret = []
    for i in -ts:
        xss = np.roll(xs, i)
        retj = []
        for j in range(0, n, window):
            retj.append(np.dot(xss[j:j + window], ys[j:j + window])/window)
        ret.append(retj)
    return np.array(ret).T

def auto_cor(xs):
    xm = xs - xs.mean()
    xn = np.sum(xm ** 2)
    return np.correlate(xm, xm, "same")/xn

def cross_cor(xs, ys):
    xm = xs - xs.mean()
    xn = np.sum(xm ** 2)
    ym = ys - ys.mean()
    yn = np.sum(ym ** 2)
    return np.correlate(xm, ym, "same")/np.sqrt(xn * yn)
    
def cross_corr_by_bouts(xs, ys, fs, bouts):
    # NOTE: fs=np.arange(-int(fps*1.5), int(fps*1.5))
    # NOTE: output length: min_frame
    min_frame = fs[-1] - fs[0] + 1
    ret = []
    for s, e in bouts:
        if e - s > min_frame:  # TODO: < min_frame, fill with nan
            xcor = cross_cor(xs[s:e], ys[s:e])
            if not np.isnan(xcor).all():
                center = int((e - s) / 2)
                ret.append(xcor[center+fs[0]:center+fs[-1]+1])
    return np.nanmean(ret, axis=0)

def plot_pca(X, y, c=None, title="", tsne=False, path=None, only_points=False, alpha=0.3, components=2, lim=None, extra_cb=None, colors=None):
    import matplotlib.pyplot as plt
    colors = colors or "kbgycm"
    if components == 0:
        X_pca = X
    elif tsne:
        from sklearn import manifold
        X_pca = manifold.TSNE(n_components=components, init='pca', random_state=0).fit_transform(X)
    else:
        from sklearn import decomposition
        svd = decomposition.TruncatedSVD(n_components=components)
        X_pca = svd.fit_transform(X)
        print(svd.explained_variance_ratio_)
        print(svd.singular_values_)

    fig = plt.figure(title, figsize=(4, 3), constrained_layout=True, dpi=300)
    if components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.gca(projection="3d")
    else:
        ax = plt.subplot()
    plt.title(title)
    onselect = None
    if c is None:
        color = "k"
    else:
        color = [colors[cc] for cc in c]
    if components == 3:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], s=1, color=color, alpha=alpha)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=1, color=color, alpha=alpha)

        if extra_cb:
            COLORS = "rgykmc"
            def onselect(i):
                xy = X_pca[i]
                c = plt.Circle(xy[:2], 0.4, color=COLORS[i % 6], linewidth=0.5, fill=False)
                ax.add_patch(c)
                fig.canvas.draw()
            def onclick(event):
                if not event.xdata:
                    return
                cx, cy = event.xdata, event.ydata
                m_i = np.argmin(np.abs(X_pca[:, 0] - cx) + np.abs(X_pca[:, 1] - cy))
                print(extra_cb(m_i))
                # onselect(m_i)
            fig.canvas.mpl_connect('button_press_event', onclick)
    if lim:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        if components == 3:
            ax.set_zlim(-lim, lim)
    if not only_points and y is not None:
        for i in range(X_pca.shape[0]):
            ax.text(X_pca[i, 0], X_pca[i, 1], str(y[i]),
                    color="b" if (c is not None and c[i] == 1) else "r",
                    fontdict={'size': 6})
    if path:
        plt.savefig(path)
    else:
        return onselect

def plot_kpt(img, xs, ys, fig=None, name=None):
    # import cv2
    # cv2.imwrite(name, img)
    # return
    import matplotlib.pyplot as plt
    from matplotlib.colors import NoNorm
    if not fig:
        fig = plt.figure("kpt")
    ax = fig.gca()
    ax.cla()
    ax.axis("off")
    ax.set_position([0, 0, 1, 1], which="both")
    ax.imshow(img.astype(int), cmap=plt.cm.gray, norm=NoNorm())
    ax.plot(xs[:3], ys[:3], 0.5, "r")
    ax.plot(xs[1:4:2], ys[1:4:2], 0.5, "g")
    ax.plot(xs[1:5:3], ys[1:5:3], 0.5, "b")
    ax.scatter(xs, ys, c="rgbkw")
    s = img.shape
    ax.set_xlim((0, s[1]))
    ax.set_ylim((s[0], 0))
    fig.canvas.draw()
    name and fig.savefig("img/track_cmp/%s.png" % name)

def plot_bar(X, y, c, title="", path=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(title)
    Xn = [np.count_nonzero(x) for x in X]
    t = np.array([int(yy) for yy in y]) * 4 + np.array(c)
    plt.bar(t, Xn)
    plt.xticks(t, t//4)
    if path:
        plt.savefig(path)
    else:
        plt.show()

def list_to_csv(info, columns, path):
    pd.DataFrame(info, columns=columns).to_csv(path, index=False)

def prepare_figure(xlabel, ylabel, xlim=None, ylim=None, ax=None, x_alpha=False, figsize=(3.6, 2.4)):
    import matplotlib.pyplot as plt
    if not ax:
        plt.figure(figsize=figsize, dpi=300)
        ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xlim and ax.set_xlim(xlim)
    ylim and ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if x_alpha:
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
    return ax

