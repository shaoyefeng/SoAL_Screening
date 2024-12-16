# -*- coding: utf-8 -*-
import functools
import os
import cv2
import shutil
from os.path import join as pjoin
from fpt_consts import ALL_VIDEO_DIR, MIN_DURATION
from fpt_util import traverse_folder, load_dict, save_dict, find_file, lim_dir
from fpt1_preprocess import get_video_static_info

def make_old_video_folder(src, dest):
    def cb(path):
        video = find_file(path, "_t.avi")
        if not video:
            return
        video = video[0]
        s = os.path.basename(video).split("_")
        camera = s[-2]
        pair_name = s[0] + "_" + s[1] + "_" + camera[0]  # K or T
        pair_name = pair_name.replace("-", "")
        pair_folder = pjoin(dest, pair_name)
        os.makedirs(pair_folder, exist_ok=True)
        new_video = pjoin(pair_folder, pair_name + ".avi")
        if os.path.exists(new_video):
            return
        log_file = pjoin(path, s[0] + "_" + s[1] + ".log")
        print("copy", video, new_video)
        not os.path.exists(new_video) and shutil.copy2(video, new_video)
        os.path.exists(log_file) and shutil.copy2(log_file, pair_name)
        old_meta = load_dict(video.replace(".avi", "_param.log"))
        cap = cv2.VideoCapture(video)
        meta = get_video_static_info(new_video, cap)
        cap.release()
        meta["camera"] = camera
        meta["FPS"] = meta["orig_fps"]
        meta["FEAT_SCALE"] = old_meta["FEAT_SCALE"]
        meta["AREA_RANGE"] = old_meta["AREA_RANGE"]
        meta["temperature"] = 22
        meta["exp_date"] = s[0]
        meta["female_date"] = s[0]
        meta["female_days"] = 0
        meta["ROI"] = [{"idx": i, "roi": roi, "male_geno": "CS", "info": "_Z_0", "male_date": s[0], "male_days": 0} for i, roi in enumerate(old_meta["ROI"])]

        save_dict(pjoin(pair_folder, pair_name + "_meta.txt"), meta)
    def cb2(path):
        fl = find_file(path, ".avi")
        fc = None
        for f in fl:
            if f.endswith("8383305_t.avi"):
                fc = f
            else:
                os.remove(f)
        if fc:
            os.rename(fc, fc.replace("8383305_t.avi", ".avi"))
    # traverse_folder("F:/video/R4/", cb)
    traverse_folder(src, cb)

def modify_meta_file(meta_file):
    if not os.path.exists(meta_file):
        print("error:", meta_file)
        return
    meta = load_dict(meta_file)
    # print(meta_file)
    # if isinstance(meta["ROI"], dict):
    #     rois = [meta["ROI"]]
    # else:
    #     rois = meta["ROI"]
    # for roi in rois:
    #     roi["male_geno"] = roi["male_geno"].replace("LC10ad", "LC15")
    #     # if roi["male_geno"].startswith("O"):
    #     #     print(roi["male_geno"])
    #     #     roi["male_geno"] = roi["male_geno"][1:]
    #     if roi["male_geno"] == "AP":
    #         roi["male_geno"] = "A"
    #     roi["info"] = "_Z_0"
    #     roi["male_days"] = 0
    #     roi["male_date"] = meta["exp_date"]
    # meta["temperature"] = 25
    meta["end"] = "16:54:06"
    # meta["log"] = meta["log"].replace("LC10ad", "LC15")
    # if meta["GRAY_THRESHOLD"] > 160:
    #     print(meta["file"])
    # meta["total_frame"] = int(meta["duration"] * 30)
    # if meta["FPS"] < 66:
    #     print("wrong:", meta_file)
    # meta["FPS"] = 66
    save_dict(meta_file, meta)

def modify_meta_dir():
    def cb1(path):
        name = os.path.basename(path)
        prefix = pjoin(path, name)
        modify_meta_file(prefix + "_meta.txt")
        for n in os.listdir(path):
            nd = pjoin(path, n)
            if os.path.isdir(nd):
                prefix1 = pjoin(nd, "%s_%s" % (name, n))
                modify_meta_file(prefix1 + "_meta.txt")
                modify_meta_file(prefix1 + "_calib_meta.txt")
                modify_meta_file(prefix1 + "_cir_meta.txt")
    def cb2(path):
        for n in os.listdir(path):
            if True:#n.startswith("20200627_"):
                nd = pjoin(path, n)
                if os.path.isdir(nd):
                    prefix1 = pjoin(nd, n)
                    # print(prefix1)
                    modify_meta_file(prefix1 + "_cir_meta.txt")
                    modify_meta_file(prefix1 + "_cir_meta_center.txt")
                    modify_meta_file(prefix1 + "_cir_meta_center_pre.txt")
    # cb1(r"G:\_video_suzukii\20201114_090820_C")
    cb1(r"G:\_video_screen\20201230_155330_E")
    # traverse_folder(r"G:\_video_screen", cb1)
    # cb2(r"G:\data_suzukii\geno_data\CtrlOne31")
    # cb2(r"G:\data_suzukii\geno_data\CtrlTwo31")
    # cb2(r"G:\data_suzukii\geno_data\ExpNone31")
    # cb2(r"G:\data_suzukii\geno_data\ExpTwo31")
    # traverse_folder(r"D:\exp\data7\geno_data", cb2)

def t1(p):
    k = os.path.basename(p)
    print("ffmpeg -i %s/%s.avi -vcodec libx264 -to 1:00:01 %s/%s_t.avi" % (k, k, k, k))

def copy_raw_video(to, p):
    d, f = os.path.split(p)
    # if not f.startswith("20200703"):
    #     return
    # meta = load_dict(pjoin(p, f + "_meta.txt"))
    # if meta and meta["GRAY_THRESHOLD"] > 165:
    #     print(p)
    # else:
    #     return
    v2 = pjoin(to, f)
    os.makedirs(v2, exist_ok=True)
    for postfix in ["_meta.txt", ".log", ".bmp", ".avi"]:
        v = pjoin(p, f + postfix)
        if os.path.exists(v):
            dest = pjoin(v2, f + postfix)
            print("copy", v, dest)
            if not os.path.exists(dest):
                shutil.copy2(v, dest)
        else:
            print("lost " + v)

def rename_camera(d):
    d = rename_camera_file(d)
    if os.path.isdir(d):
        for name in os.listdir(d):
            rename_camera(pjoin(d, name))

def rename_camera_file(d):
    dst = None
    d0, d1 = os.path.dirname(d), os.path.basename(d)
    dp = d1.find(".")
    pair_name, ext = d1[:dp], d1[dp:]
    s = list(pair_name.split("_"))
    if len(s) > 2:
        cm = s[2]
        if cm == "A":
            s[2] = "1"
            dst = pjoin(d0, "_".join(s) + ext)
        elif cm == "B" or cm == "K":
            s[2] = "2"
            dst = pjoin(d0, "_".join(s) + ext)
    if dst:
        print(d, dst)
        os.rename(d, dst)
    return dst or d

def copy_data(src, dst):
    not os.path.exists(dst) and os.makedirs(dst)
    def cb(path):
        pair_name = os.path.basename(path)
        prefix = pjoin(path, pair_name)
        dst_prefix = pjoin(dst, pair_name)
        not os.path.exists(dst_prefix) and os.mkdir(dst_prefix)
        print(prefix, dst_prefix)
        for postfix in ["_all.txt", "_center.txt",
                        "_cir_center_stat0.pickle", "_cir_center_stat1.pickle", "_cir_center_stat2.pickle",
                        "_cir_stat0.pickle", "_cir_stat1.pickle", "_cir_stat2.pickle",
                        "_we_stat0.pickle", "_we_stat1.pickle", "_we_stat2.pickle",
                        "_cir_info.csv", "_cir_info_center.csv", "_cir_meta.txt", "_cir_meta_center.txt",
                        ]:
            if os.path.exists(prefix + postfix):
                shutil.copy2(prefix + postfix, dst_prefix)

    traverse_folder(src, cb)

    prefix = pjoin(src, "_" + os.path.basename(src))
    for postfix in ["_all.txt", "_center.txt", "_cir_all.csv", "_cir_center.csv"]:
        shutil.copy2(prefix + postfix, dst)

def copy_feat(src, dst):
    if not os.path.exists(dst):
        return
    pair_name = os.path.basename(src)
    pair_name_dst = os.path.basename(dst)
    def cb(path):
        pair_no = os.path.basename(path)
        prefix = pjoin(path, "%s_%s" % (pair_name, pair_no))
        dst_prefix = pjoin(dst, pair_no, "%s_%s" % (pair_name_dst, pair_no))
        for postfix in ["_meta.txt", "_feat.csv"]:
            if os.path.exists(prefix + postfix):
                print(prefix + postfix, dst_prefix + postfix)
                shutil.copy2(prefix + postfix, dst_prefix + postfix)
    traverse_folder(src, cb)

def t_dir():
    import numpy as np
    from fpt_frame_stat import angle_to_vector
    for dir1 in np.arange(-360, 360, 37):
        dirv = angle_to_vector(dir1)
        theta1 = np.deg2rad(lim_dir(90 - dir1))
        theta2 = np.arctan2(dirv[0], dirv[1])
        # print("%.2f"%(theta1-theta2))
        print("%.2f %.2f %.2f"%(np.sin(theta1), np.sin(theta2), dirv[0]))
        print("%.2f %.2f %.2f"%(np.cos(theta1), np.cos(theta2), dirv[1]))

def recover_meta(geno_data_folder, out_folder):
    meta_d = {}
    for geno in os.listdir(geno_data_folder):
        geno_folder = pjoin(geno_data_folder, geno)
        for pair in os.listdir(geno_folder):
            pair_folder = pjoin(geno_folder, pair)
            if os.path.isdir(pair_folder):
                video_name = pair[:pair.rfind("_")]
                meta = load_dict(pjoin(pair_folder, pair + "_cir_meta.txt"))
                del meta["cir_bouts1"]
                del meta["cir_bouts2"]
                del meta["copulate"]
                if not meta_d.get(video_name):
                    meta_d[video_name] = meta
                    meta_d[video_name]["ROI"] = [meta_d[video_name]["ROI"]]
                else:
                    meta_d[video_name]["ROI"].append(meta["ROI"])
    for video_name, meta in meta_d.items():
        save_dict(pjoin(out_folder, video_name + "_meta.txt"), meta)

def backup_raw_video(src_dir, dest_dir):
    pair_list = []

    # ev = open("D:/exp/exp_video.txt", "r").readlines()
    # for e in ev:
    #     if len(e) > 3:
    #         video = e.split()[1]
    #         video = video.replace("A", "1").replace("B", "2").replace("K", "2")
    #         pair_list.append(video)

    sz_a = 0
    for pair in os.listdir(src_dir):
        if pair_list and pair not in pair_list:
            continue
        pair_folder = pjoin(src_dir, pair)
        if os.path.isdir(pair_folder):
            meta = pjoin(pair_folder, pair + "_meta.txt")
            if load_dict(meta)["duration"] < MIN_DURATION:
                print("skip", pair)
                continue
            avi = pjoin(pair_folder, pair + ".avi")
            dest_pair_dir = pjoin(dest_dir, pair)
            os.makedirs(dest_pair_dir, exist_ok=True)
            if os.path.exists(meta) and os.path.exists(avi) and not os.path.exists(pjoin(dest_dir, pair + ".avi")):
                shutil.copy2(meta, dest_pair_dir)
                shutil.copy2(avi, dest_pair_dir)
                shutil.copy2(pjoin(pair_folder, pair + ".bmp"), dest_pair_dir)
                sz = os.path.getsize(avi)
                sz_a += sz
                print("finish: %s %.2fG" % (avi, sz / 1073741824))
    print("total: %.2fG" % (sz_a / 1073741824))

def valid_videos():
    valid_genos = "CS22 IR22 A22".split()
    # valid_genos = "CS22 IR22 FW22 MW22 A22 AIR22 MWIR22".split()
    # valid_genos = "CSXCS31 CSXShi31 CSXTrp31 LC24XCS31 LC24XShi31 LC24XTrp31 " \
    #               "LC26XCS31 LC26XShi31 LC26XTrp31 LC6XCS31 LC6XShi31 LC6XTrp31 " \
    #               "LC18XCS31 LC18XShi31 LC18XTrp31 T45XCS31 T45XShi31 T45XTrp31 "\
    #               "LC11XCS31 LC11XShi31 LC11XTrp31 LC12XCS31 LC12XShi31 LC12XTrp31 "\
    #               "LC16XCS31 LC16XShi31 LC16XTrp31 LPLC2XCS31 LPLC2XShi31 LPLC2XTrp31".split()
    video_set = set()
    geno_info = {}
    pairs = 0
    for i in ["_hrnet"]:#range(0, 10):["4s"]:#
        data_name = "data" + str(i)
        geno_data = pjoin("G:/", data_name, "geno_data")
        if os.path.exists(geno_data):
            # for geno in os.listdir(geno_data):
            for geno in valid_genos:
                geno_info.setdefault(geno, {"pairs": 0, "bouts_all": 0, "bouts_center": 0, "data": []})
                geno_folder = pjoin(geno_data, geno)
                if os.path.exists(geno_folder):
                    geno_info[geno]["data"].append(data_name)
                    info_center = load_dict(pjoin(geno_folder, "_%s_center.txt" % geno))
                    if not info_center:
                        print("no info_center", geno)
                        continue
                    for pair, pair_info in info_center["pairs"].items():
                        if pair_info.get("cop_bouts"):
                            cb = pair_info["cop_bouts"][0]
                            if cb[1] - cb[0] > 66*3600*2/3:
                                print("cop:", pair)
                    pairs_l = info_center["pairs"].values()
                    bouts_all = [a["cir_count"] for a in pairs_l]
                    bouts_center = [a["cir_with_cond"] for a in pairs_l]
                    geno_info[geno]["bouts_all"] += sum(bouts_all)
                    geno_info[geno]["bouts_center"] += sum(bouts_center)
                    geno_info[geno]["pairs"] += len(pairs_l)
                    pairs += len(pairs_l)
                    for pair in os.listdir(geno_folder):
                        if os.path.isdir(pjoin(geno_folder, pair)):
                            video_set.add(pair[:pair.rfind("_")])
                            # pairs += 1
                            # geno_info[geno]["pairs"] += 1
    # return
    video_l = list(video_set)
    video_l.sort()
    ev = open("D:/exp/exp_video.txt", "w")
    for i, v in enumerate(video_l):
        if v.startswith("2"):
            print(i, v)
            ev.write("%s %s\n" % (i, v))
    msg = "%d pairs, ~%d frames, ~%dG" % (pairs, 66*3600*pairs, 3*len(video_l))
    print(msg)
    ev.write(msg)
    ev.close()

    print(geno_info)
    import pandas as pd
    pd.DataFrame.from_dict(geno_info, orient="index").to_csv("D:/exp/exp_data.csv")
    return video_l

def test(folder):
    for f in os.listdir(folder):
        pos = f.find("_")
        shutil.move(os.path.join(folder, f), os.path.join(folder, f[pos+1:]))

if __name__ == '__main__':
    # for video_dir in ALL_VIDEO_DIR:
    #     backup_raw_video("G:/%s" % video_dir, "G:/_video_hrnet")
    # backup_raw_video("G:/_video_suzukii_finish", "Z:/nj/temp")
    # make_old_video_folder(r"F:\video_old\old1_manip\R2Audio", r"F:\video_todo\old1_audio")
    # make_old_video_folder(r"F:\video_old\old0_no_food", r"G:/video_todo/old0_no_food/")
    # make_old_video_folder(r"F:\video_old\old1_manip\Cross", r"G:/video_todo/old2_cross/")
    # make_old_video_folder(r"F:\video_old\old1_manip\Fix", r"G:/video_todo/old3_fix/")
    # make_old_video_folder(r"F:\video_old\old1_manip\R2CutAnt", r"G:/video_todo/old4_cutant/")
    # make_old_video_folder(r"F:\video_old\old1_manip\R2CutWing", r"G:/video_todo/old5_cutwing/")
    # make_old_video_folder(r"F:\video_old\old1_manip\R2Headless", r"G:/video_todo/old6_cuthead/")
    # make_old_video_folder(r"F:\video_old\old1_manip\R2Paint", r"G:/video_todo/old7_painteye/")
    # make_old_video_folder(r"F:\video_old\old1_manip\R4", r"G:/video_todo/old8_r4/")
    # make_old_video_folder(r"F:\video_old\old1_manip\R4Fw1118", r"G:/video_todo/old8_r4/")
    # make_old_video_folder(r"F:\video_old\old1_manip\R4MFw1118", r"G:/video_todo/old8_r4/")
    # make_old_video_folder(r"F:\video_old\old1_manip\R4T2", r"G:/video_todo/old9_r4t2/")
    # modify_meta_dir()
    # recover_meta(r"D:\exp\data7\geno_data", r"F:\recovery\1")
    test(r"D:\exp\code\img\fab_alpha2\select")
    # copy_data(r"G:\data_hrnet\geno_data\A22", r"D:\exp\data_hrnet\geno_data\A22")
    # for video_dir in "old0_nofood  old2_cross  old4_cutant   old6_cuthead   old8_r4    video3_fps30\
    # old1_audio   old3_fix    old5_cutwing  old7_painteye  old9_r4t2".split():
    #     traverse_folder("/media/nj/RawData/video_todo/%s/" % video_dir, functools.partial(copy_raw_video, "/mnt/disk2/RawVideo"))
    # rename_camera(r"G:\data4s\geno_data\CSXCS31\20191207_131429_B_12")
    # traverse_folder(r"G:\data4s\geno_data", rename_camera)
    # t_dir()
    # valid_videos()
    # copy_feat(r"Z:\nj\video_todo2\20200626_143715_B", r"F:\temp\video_todo\20200626_143715_2")

