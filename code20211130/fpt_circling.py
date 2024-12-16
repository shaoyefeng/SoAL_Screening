# -*- coding: utf-8 -*-
"""
1. do_detect_behavior
2. extract_cir

directory structure:
   |-exp
       |-20190807_140000_A
           |-0
               |-20190807_140000_A_0_meta.txt
               |-20190807_140000_A_0_feat.csv
               |-20190807_140000_A_0_stat0.pickle
               |-20190807_140000_A_0_stat1.pickle
               |-20190807_140000_A_0_stat2.pickle
               |-20190807_140000_A_0_cir_meta.txt
                   |-*meta
                   |-cir_bouts
               |-20190807_140000_A_0_cir_info.csv
                   |-frames, start_angle, end_angle, speed, av, ...
           |-1
           |-...
       |-20190807_140000_B
       |-...
"""
import os
import sys

import cv2
# import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from fpt_consts import FIX_VIDEO_SIZE, ALL_VIDEO_DIR, DURATION_LIMIT, DIST_TO_CENTER_THRESHOLD_FEMALE, \
    DIST_TO_CENTER_THRESHOLD_MALE
from fpt_util import save_dict, load_dict, save_dataframe, load_dataframe, \
    get_video_in_dir, mp_process_files, correct_angle, load_dfs, get_all_feat_file, calc_bouts, \
    calc_jaaba_bouts, dfs_bouts, save_dfs, pair_to_video_path, squeeze_cir_bouts, find_file, bouts_to_seq, plot_pca, \
    plot_bar, get_all_cir_meta
from fpt_frame_stat import calc_frame_stat, detect_behavior_and_correct_id, STATE_SIDE_WALK, \
    T_HEAD_TAIL_CLOSE_DIST, detect_interaction, shrink_and_smooth, detect_we_start

JAABA_BEHS = []#"Approaching", "Chasing", "Escaping", "Fencing", "Flicking", "Lunging", "Ruaw", "Threat", "Tuaw", "Tuto"]

def do_detect_behavior(feat_file, stat_prefix=None, force=True):
    if stat_prefix:
        pickle0 = stat_prefix + "_stat0.pickle"
        pickle1 = stat_prefix + "_stat1.pickle"
        pickle2 = stat_prefix + "_stat2.pickle"
    else:
        pickle0 = feat_file.replace("feat.csv", "stat0.pickle")
        pickle1 = feat_file.replace("feat.csv", "stat1.pickle")
        pickle2 = feat_file.replace("feat.csv", "stat2.pickle")
    # if os.path.exists(pickle0):
    #     if os.path.basename(feat_file).startswith("2020"):
    #         print("finish: skip", feat_file)
    #         return
        # mtime = os.stat(pickle0).st_mtime
        # if time.time() - mtime < 10800:
        #     if os.path.getsize(pickle0) > 20000*1000 and os.path.getsize(pickle1) > 70000*1000 and os.path.getsize(pickle2) > 70000*1000:
        #         print("finish: skip", feat_file)
        #         return

    # meta = load_dict(feat_file.replace("feat.csv", "meta.txt"))
    # geno = meta["ROI"]["male_geno"]
    # if meta and (geno == "CS" or geno == "IR") and int(meta["temperature"]) == 22: #test
    #     print("%s DDD" % geno)
    #     return
    # else:
    #     if int(meta["temperature"]) != 31:
    #         print("%s DDD" % geno)
    #         return
    #     print("not CS do ", geno)
    #     return
    if os.path.exists(pickle0) and not force:
        dfs = load_dfs(pickle0)
        cir_meta = load_dict(stat_prefix + "_cir_meta.txt")
        return dfs, cir_meta
    dfs, meta = calc_frame_stat(feat_file)
    if meta["duration"] > 10000:
        meta["duration"] = 3600
    fps = int(meta["total_frame"]/meta["duration"] + 0.5)  # NOTE: modify FPS
    print("fps=", fps)
    dfs = detect_behavior_and_correct_id(dfs, fps)
    save_dataframe(dfs[0], pickle0)  # NOTE: save stat0
    save_dataframe(dfs[1], pickle1)  # NOTE: save stat1
    save_dataframe(dfs[2], pickle2)  # NOTE: save stat2

    cir_meta = meta
    cir_meta["FPS"] = fps
    cir_meta["copulate"] = calc_bouts(dfs[0]["copulate"])
    cir_meta["cir_bouts1"] = calc_bouts(dfs[1]["circle"])
    cir_meta["cir_bouts2"] = calc_bouts(dfs[2]["circle"])
    if JAABA_BEHS:
        jaaba_mat = os.path.join(os.path.dirname(feat_file), "scores_%s.mat")
        for beh in JAABA_BEHS:
            cir_meta[beh + "1"], cir_meta[beh + "2"] = calc_jaaba_bouts(jaaba_mat % beh)
            for fly in ["1", "2"]:
                if beh == "Threat":
                    cir_meta[beh + fly] = list(filter(lambda t: t[1] - t[0] > 50, cir_meta[beh + fly]))
                if beh == "Flicking":
                    cir_meta[beh + fly] = max_wing_in_bouts(cir_meta[beh + fly], dfs[int(fly)], 3)
    else:
        detect_interaction(cir_meta, dfs)
    save_dict(stat_prefix + "_cir_meta.txt", cir_meta)  # NOTE: save cir_meta

    # do_extract_avi(feat_file)
    # cir_info = calc_all_cir_info(dfs, cir_meta)
    # cir_info.to_csv(feat_file.replace("feat.csv", "cir_info.csv"))
    return dfs, meta

def max_wing_in_bouts(bouts, dfsf, frames):
    ret = []
    for s, e in bouts:
        if e - s >= 2:
            wm = np.argmax(dfsf[s:e]["wing_r"] - dfsf[s:e]["wing_l"])
            ret.append((wm, min(e, wm + frames)))
    return ret

def do_extract_avi(cir_meta_file, cir_meta=None):
    do_update_interaction(cir_meta_file)
    pair_name = os.path.basename(cir_meta_file).replace("_cir_meta.txt", "")
    m = cir_meta or load_dict(cir_meta_file)
    if not m:
        return
    roi = m["ROI"]["roi"]
    video_info = pair_to_video_path(pair_name)#get_video_in_dir(os.path.dirname(os.path.dirname(feat_file)))
    if not video_info:
        return
    video = video_info[0]
    # if os.path.exists(cir1_avi):
    #     print("skip already exists", cir1_avi)
    #     return
    # extract_avi(video, m["cir_bouts1"], roi, cir_meta_file.replace("_cir_meta.txt", "_cir1.avi"))
    # extract_avi(video, m["cir_bouts2"], roi, cir_meta_file.replace("_cir_meta.txt", "_cir2.avi"))
    # test
    # extract_avi(video, m["fabl_bouts"], roi, cir_meta_file.replace("_cir_meta.txt", "_fabl.avi"))
    # extract_avi(video, m["fabr_bouts"], roi, cir_meta_file.replace("_cir_meta.txt", "_fabr.avi"))
    # behs = ["we_l_start"]
    behs_bouts = {
        # "cir_bouts1": 30, "we_l_start": 30, "we_on_l_bouts": 60, "wl_g_wr": 200,
        # "fabl_bouts": 30, "ac_bouts": 30, "acp_bouts": 300
        "fabl_far_bouts": 30,
    }#"engaged_bouts": 1, "fmotion_bouts": 200,
    for k, v in behs_bouts.items():
        extract_avi(video, m[k][:v], roi, cir_meta_file.replace("_cir_meta.txt", "_%s.avi" % k))
    # dfs = load_dfs(cir_meta_file.replace("_cir_meta.txt", "_stat0.pickle"), only=1)
    # extract_avi(video, calc_bouts(dfs[1]["on_edge"]), roi, cir_meta_file.replace("_cir_meta.txt", "_edge.avi"))

    for beh in JAABA_BEHS:
        for fly in ["1", "2"]:
            extract_avi(video, m.get(beh + fly), roi, cir_meta_file.replace("cir_meta.txt", beh + fly + ".avi"))

def do_extract_stat(cir_meta_file, cir_meta=None, dfs=None):
    dfs = dfs or load_dfs(cir_meta_file.replace("_cir_meta.txt", "_stat0.pickle"))
    m = cir_meta or load_dict(cir_meta_file)
    for beh in JAABA_BEHS:
        for fly in ["1", "2"]:
            if beh + fly in m:
                dfs_beh = dfs_bouts(dfs, m[beh + fly])
                save_dfs(dfs_beh, cir_meta_file.replace("meta.txt", beh + fly))
    
def do_update_copulation(feat_file):
    meta_file = feat_file.replace("feat.csv", "cir_meta.txt")
    cir_meta = load_dict(meta_file)
    if cir_meta.get("copulate") is not None:
        print("exist.")
        return
    dfs = load_dfs(feat_file)
    cir_meta["copulate"] = calc_bouts(dfs[0]["copulate"])
    save_dict(meta_file, cir_meta)

def do_update_interaction(meta_file):
    cir_meta_file = meta_file.replace("feat.csv", "cir_meta.txt")
    cir_meta = load_dict(cir_meta_file)
    dfs = load_dfs(os.path.dirname(meta_file))
    detect_interaction(cir_meta, dfs)
    save_dict(meta_file, cir_meta)
    return cir_meta

def do_calc_all_cir_info(feat_file):
    dfs = load_dfs(feat_file)
    cir_meta_file = feat_file.replace("feat.csv", "cir_meta.txt")
    cir_info = calc_all_cir_info(dfs, load_dict(cir_meta_file))
    cir_info.to_csv(feat_file.replace("feat.csv", "cir_info.csv"))

def get_center(roi, scale):
    return (roi[1][0] - roi[0][0])/2/scale, (roi[1][1] - roi[0][1])/2/scale

def calc_all_cir_info(dfs, cir_meta):
    cir_bouts = cir_meta["cir_bouts1"]
    # center = get_center(cir_meta["ROI"]["roi"], cir_meta["FEAT_SCALE"])
    ret = []
    for s, e in cir_bouts:
        info = calc_cir_info(dfs[0][s:e], dfs[1][s:e], dfs[2][s:e], s, e)
        info and ret.append(info)
    return pd.DataFrame(ret)

def calc_cir_info(stat0, stat1, stat2, s, e):
    frames = len(stat1)
    if not frames:
        return None
    angle_l = correct_angle(stat2["rel_polar:t"].tolist())
    angle_range = np.max(angle_l) - np.min(angle_l)
    ipsi_c = np.count_nonzero(stat1["we_ipsi"] == 1)
    contra_c = np.count_nonzero(stat1["we_ipsi"] == -1)
    wec_head = (ipsi_c - contra_c) / (ipsi_c + contra_c + 1e-6)
    return {
        "frames": frames,
        "s": s,
        "e": e,
        "dist_c": np.median(stat1["dist_c"]),
        "dist_c2": np.median(stat2["dist_c"]),
        "side_ratio": np.count_nonzero(stat1["walk"] == STATE_SIDE_WALK) / frames,
        "we_ratio": np.count_nonzero(stat1["we_l"] | stat1["we_r"]) / frames,
        "start_angle": stat2.iloc[0]["rel_polar:t"],
        "end_angle": stat2.iloc[-1]["rel_polar:t"],
        "accum_move": np.nansum(np.sqrt(stat1["pos:x"].diff()**2 + stat1["pos:y"].diff()**2)),  # move_dist
        "accum_move_f": np.nansum(np.sqrt(stat2["pos:x"].diff()**2 + stat2["pos:y"].diff()**2)),
        "accum_rotate": np.sum(np.abs(stat1["d_dir"])),  # angle_change
        "accum_rotate_f": np.sum(np.abs(stat2["d_dir"])),
        "angle_range": angle_range,
        "speed": np.mean(stat1["v_len"]),
        "av": np.mean(np.abs(stat1["av"])),
        "dir_std": np.std(correct_angle(stat1["dir"].tolist())),
        "vs": np.mean(stat1["vs"]),
        "vf": np.mean(stat1["vf"]),
        "dist_McFc": np.mean(stat0["dist_McFc"]),
        "dist_MhFh": np.mean(stat1["rel_polar_hh:r"]),
        "dist_MhFt": np.mean(stat1["rel_polar_ht:r"]),
        "f_y": np.mean(stat1["rel_pos:y"]),  # -: female in the back
        "e_maj": np.mean(stat1["e_maj"]),

        "we_l": np.count_nonzero(stat1["we_l"]) / frames, "we_r": np.count_nonzero(stat1["we_r"]) / frames,
        "we_lr": np.count_nonzero(stat1["we_l"] & stat1["we_r"]) / frames, "wec_head": wec_head,
        # TODO:
        # "towards_head": towards_head, "towards_tail": towards_tail,
        # "switch_wing": switch_wing, "switch_theta": switch_theta,

        "i_swap_r": np.count_nonzero(np.abs(stat1["d_dir"]) > 120) / frames,
        "i_ht_close_r": np.count_nonzero(np.abs(stat1["dist_ht"]) < T_HEAD_TAIL_CLOSE_DIST) / frames,
    }

def csv_to_pickle(csv):
    save_dataframe(load_dataframe(csv), csv.replace(".csv", ".pickle"))

def extract_avi(video, bouts, roi, out_video, need_text=True, extend_pixel=0):
    if not bouts:
        return
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)# / 2
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if roi is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        if FIX_VIDEO_SIZE:
            fps /= 2
            width, height = FIX_VIDEO_SIZE
        left, top = int(roi[0][0]), int(roi[0][1])
        right, bottom = int(roi[1][0]), int(roi[1][1])
        if extend_pixel:
            left = max(0, left - extend_pixel)
            top = max(0, top - extend_pixel)
            right = min(width, right + extend_pixel)
            bottom = min(height, bottom + extend_pixel)
    output_video = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, (int(right - left), int(bottom - top)))
    i = 0
    for s, e in tqdm(bouts):
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for frame in range(s, e):
            ret, img = cap.read()
            if not ret:
                break
            if FIX_VIDEO_SIZE and roi is not None:
                img = cv2.resize(img, FIX_VIDEO_SIZE)
            img = img[top:bottom, left:right]
            if need_text:
                # BGR
                cv2.putText(img, "%d %d-%d %d" % (i, s, e, frame), (6, 21), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                cv2.putText(img, "%d %d-%d %d" % (i, s, e, frame), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            output_video.write(img)
        i += 1
    cap.release()
    output_video.release()

def wrap_func(feat_file):
    # dfs, meta = do_detect_behavior(feat_file)
    do_extract_avi(feat_file)
    # do_extract_stat(feat_file, dfs)
    ## do_update_copulation(feat_file)
    ## do_update_interaction(feat_file)
    ## do_calc_all_cir_info(feat_file)
    # log = "temp/" + os.path.basename(feat_file) + ".txt"
    # f = open(log, "w")
    # f.write(datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S"))
    print("finish: ", feat_file)
    # f.close()

BEH_BOUTS_LENGTH_MIN = {"Flicking": 2, "Lunging": 5}
def remove_short_bouts(cir_meta, beh, length_min):
    cir_meta[beh] = [a for a in cir_meta[beh] if (a[1] - a[0] >= length_min)]

def beh_cluster(root):
    X_beh = {}
    y_beh = {}
    c_beh = {}
    for f in os.listdir(root):
        ff = find_file(os.path.join(root, f), "cir_meta.txt")[0]
        print(ff)
        cir_meta = load_dict(ff)
        for beh in JAABA_BEHS:
            X_beh.setdefault(beh, [])
            y_beh.setdefault(beh, [])
            c_beh.setdefault(beh, [])
            for fly in ["1", "2"]:
                behf = beh + fly
                remove_short_bouts(cir_meta, behf, BEH_BOUTS_LENGTH_MIN.get(beh, 20))
                seq = bouts_to_seq(cir_meta, behf)
                # print(behf, len(seq))
                X_beh[beh].append(seq)
                y_beh[beh].append(f)
                c_beh[beh].append(int(fly))

    # for beh in JAABA_BEHS:
    #     X = np.array(X_beh[beh])
    #     print(X.shape)
    #     # plot_pca(X, y_beh[beh], c_beh[beh], title=beh, path=None, tsne=False)#r"D:\exp\video_wjl\WJL_aggressionData\tsne_" + beh
    #     plot_bar(X, y_beh[beh], c_beh[beh], title=beh, path=r"D:\exp\video_wjl\WJL_aggressionData\bar_" + beh)

    for i in range(0, 24):
        X = []
        for beh in JAABA_BEHS:
            X.append(np.array(X_beh[beh][i]))
        # plot_pca(X, y_beh[beh], c_beh[beh], title=beh, path=None, tsne=False)#r"D:\exp\video_wjl\WJL_aggressionData\tsne_" + beh

    for beh1 in JAABA_BEHS:
        for beh2 in JAABA_BEHS:
            for i in range(0, 24):
                if beh1 != beh2:
                    s1 = X_beh[beh1][i]
                    s2 = X_beh[beh2][i]
                    cor = np.corrcoef(s1, s2)[0][1]
                    fly = i % 2 + 1
                    if cor > 0.2:
                        print("%s%d %s%d %d %.4f" % (beh1, fly, beh2, fly, i/2, cor))
            # for i in range(0, 24, 2):
            #     s1 = X_beh[beh1][i]
            #     s2 = X_beh[beh2][i + 1]
            #     cor = np.corrcoef(s1, s2)[0][1]
            #     if cor > 0.2:
            #         print("%s1 %s2 %d %.4f" % (beh1, beh2, i/2, cor))

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        beh_cluster(r"D:\exp\video_wjl\WJL_aggressionFeatfile")
        # for ana_dir in ["video7_ir"]:
        #     mp_process_files(do_detect_behavior, get_all_feat_file("E:/" + ana_dir))
        # for f in os.listdir(VIDEO_ANA_DIR):
        #     if os.path.isdir(VIDEO_ANA_DIR + f):
        #         # if True:#int(f.split("_")[0]) >= 20190727:#
        #         if len(get_all_feat_file(VIDEO_ANA_DIR + f)) == 0:
        #             mp_process_files(do_detect_behavior, get_all_feat_file(VIDEO_ANA_DIR + f))
        #         else:
        #             print("skip " + f)
    else:
        feat_file = sys.argv[1]
        if feat_file == "all":
            # NOTE: all 7
            # NOTE: all video7_ir
            # NOTE: all video7_ir/20200602_144311_A
            # NOTE: all F:\temp\video_todo\20200602_144312_2
            cmd2 = sys.argv[2]
            if cmd2.find(":") > 0:
                VIDEO_ANA_DIR = cmd2
            else:
                if len(cmd2) < 4:
                    cmd2 = [vd for vd in ALL_VIDEO_DIR if vd.startswith("video" + cmd2 + "_")][0]
                VIDEO_ANA_DIR = "G:/" + cmd2
            mp_process_files(wrap_func, get_all_cir_meta(VIDEO_ANA_DIR))
            # mp_process_files(wrap_func, get_all_feat_file(VIDEO_ANA_DIR))
        elif os.path.isfile(feat_file):
            wrap_func(feat_file)


