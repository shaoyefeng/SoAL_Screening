# -*- coding: utf-8 -*-
import json
import os
import numpy as np
from fpt_consts import FLY_NUM, POINT_NUM

def load_dict(filename):
    if not os.path.exists(filename):
        return None
    # print("load_dict %s" % filename)
    f = open(filename, "r")
    j = json.load(f)
    f.close()
    return j

def save_dict(filename, obj):
    f = open(filename, "w")
    json.dump(obj, f, indent=4)
    f.close()
    print("save_dict %s" % filename)

def array_to_str(a):
    return " ".join(["%.2f" % p for p in a])

def distance2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

FEAT_KEYS_G = ["frame", "reg_n"]
FEAT_KEYS_F = [":area", ":pos:x", ":pos:y", ":ori", ":e_maj", ":e_min", ":point:xs", ":point:ys"]

def get_feat_header():
    ret = list(FEAT_KEYS_G)
    for j in range(FLY_NUM):
        sj = str(j + 1)
        ret.extend([sj + k for k in FEAT_KEYS_F])
    return ret

def to_feat_s(info_n):
    # "frame", "reg_n"
    # ":area", ":pos:x", ":pos:y", ":ori", ":e_maj", ":e_min", ":point:xs", ":point:ys"
    ret = ""
    frame = 0
    reg_n = 0
    for info in info_n:
        frame, reg, reg_n, points = info
        area, center, orient90, major, minor, center_global = reg
        ret += ",%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%s" % (area, center[0], center[1],
                                                        orient90, major, minor,
                                                        array_to_str(points[:POINT_NUM]),
                                                        array_to_str(points[POINT_NUM:POINT_NUM + POINT_NUM]))
    return "%d,%d" % (frame, reg_n) + ret

def id_assign(fly_info, last_fly_info):
    # return fly_info
    if not last_fly_info or FLY_NUM == 1:
        return fly_info
    ret = [None] * FLY_NUM
    flag = [False] * FLY_NUM
    for i in range(FLY_NUM):
        posi = fly_info[i][1][1]
        dl = []
        for j in range(FLY_NUM):
            if flag[j]:
                dl.append(np.inf)
            else:
                posj = last_fly_info[j][1][1]
                dl.append(distance2(posi, posj))
        min_j = np.argmin(dl)
        ret[min_j] = fly_info[i]
        flag[min_j] = True
    # print("fly_info", fly_info[0][1][1])
    # print("last_fly_info", last_fly_info[0][1][1])
    # print("ret", ret[0][1][1])
    return ret

def id_assign_dict(d):
    fly_info = []
    last_fly_info = None
    ret = []
    for t in d:
        pos = (t["1:pos:x"], t["1:pos:y"])
        fly_info.append([t, (0, pos)])
        if len(fly_info) >= FLY_NUM:
            fly_info = id_assign(fly_info, last_fly_info)
            t_all = {"frame": t["frame"], "reg_n": t["reg_n"]}
            for i, info in enumerate(fly_info):
                for k, v in info[0].items():
                    if k.startswith("1:"):
                        t_all[k.replace("1:", "%d:" % (i+1))] = v
            ret.append(t_all)
            last_fly_info = fly_info
            fly_info = []
    return ret
