# -*- coding: utf-8 -*-
"""
frame statistics

directory structure:
   |-exp
      |-20190807_140000_A
          |-0
              |-20190807_140000_A_0_feat.csv
              |-20190807_140000_A_0_meta.txt

              |-20190807_140000_A_0_stat0.csv (scaled)
                  |-frame, time, reg_n, 1:area, 1:pos:x, 1:pos:y, 1:ori, 1:e_maj, 1:e_min, 1:point:xs, 1:point:ys
              |-20190807_140000_A_0_stat1.csv
                  |-dist_McFc, dir, walk, wing_l, wing_r, theta
                  |-v_len, v_dir, vs, vf, av, acc, d_pos, d_dir
                  |-frame, dist_McFc, rel_dir
                  |-rel_pos_h|c|t:x|y, rel_polar_h|c|t:r|t, rel_pos_hh|ht|th:x|y, rel_polar_hh|ht|th:r|t
                  |-$dist_MhFh=rel_polar_hh:r $dist_MhFt=rel_polar_ht:r
                  |-frame, on_edge, we_l, we_r, we_ipsi, phi_m, dist_ht, ht_span, t_span
              |-20190807_140000_A_0_stat2.csv
   |-data
       |-all
       |-center
       |-good
"""
import os
import cv2
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from fpt_util import get_meta, distance, correct_angle, load_dataframe, save_dataframe, save_dict, get_video_in_dir, \
    load_dict, lim_dir, angle_diff, angle_diff_a, calc_bouts, lim_dir_a, correct_wing
from fpt0_util import id_assign_dict
from fpt_consts import DURATION_LIMIT, FEMALE_AVG_LEN_L, FEMALE_AVG_LEN_H, DIST_TO_CENTER_THRESHOLD_FEMALE, \
    DIST_TO_CENTER_THRESHOLD_MALE, DIST_TO_FLY_FAR, FLY_NUM

NO_CENTERED_INFO = False  # test

STATE_FORWARD = 0
STATE_SIDE_WALK = 1
STATE_STATIC = 2
STATE_BACKWARD = 3

SPEED_STATIC_MAX = 0.3  # (mm/s) for determine static(speed range 0-10)
ANGLE_SIDE = 30  # (degree) for determine backward and side walk
# for detect on edge
ANGLE_EDGE_MIN = 10  # (degree) wing angle on the edge
ANGLE_WE_MIN = 45  # wing extension
ANGLE_WE_MIN_30 = 30  # wing extension
ANGLE_WE_MAX = 120  # wing extension
T_HEAD_TAIL_CLOSE_DIST = 1 if FLY_NUM > 1 else 0.1

# OVERLAP_CONTINUE = 8  # frames (NOTE: FPS dependent)
# SIDE_WALK_CONTINUE = 5  # smooth (NOTE: FPS dependent)
# COURT_CONTINUE = 8  # smooth (NOTE: FPS dependent)

OVERLAP_CONTINUE_T = 0.12  # NOTE: unequal tail distribution
SIDE_WALK_CONTINUE_T = 0.08
COURT_CONTINUE_T = 0.12

COURT_N_CONTINUE = 2  # smooth (NOTE: FPS dependent)
CIR_CONTINUE = 2  # (frame) determine circling start & end (NOTE: FPS dependent)
# for detect copulation
COP_DURATION_MIN = 60  # seconds
# for detect circling
# CIR_CONTINUE2 = 10  # (frame) connect two adjacent circling segments (FPS dependent)
CIR_DURATION_MIN = 0.4  # (s)(0.4s) min frames
CIR_SIDE_RATIO_MIN = 0.5  # (percent) for determine circling
# TODO: CIR_OVERLAP_RATIO_MIN
# CIR_WE_RATIO_MIN = 0.3  # (percent) min wing extension for circling (low due to detection error)
# CIR_SUM_MOVE_DIST_MIN = 2.6  #4  # (mm) min move dist for circling  (FPS dependent (noise))
# CIR_SUM_ANGLE_CHANGE_MIN = 8  # (degree) min angle change for circling  (FPS dependent)

CIR_RECHECK_SPEED = 2
CIR_RECHECK_AV = 20
CIR_MIN_X_RANGE = 0.7  # mm
CIR_MIN_DIR_STD = 12  # (NOTE: FPS dependent)
# CIR_RECHECK_SIDE_RATIO = 0.8
# CIR_RECHECK_SUM_ANGLE_CHANGE = 25
# CIR_RECHECK_SUM_MOVE_DIST_MIN = 8  # (FPS dependent)
# CIR_RECHECK_MOVE_DIST_MIN = 2.5  # (FPS dependent)
CIR_RECHECK_DIST_MAX = 5  # (NOTE: body size dependent)
# CIR_RECHECK_DIST_MIN = 1.7
if FLY_NUM < 2:
    CIR_RECHECK_DIST_MAX = 100

def walk_state(speed, theta):
    state = STATE_STATIC
    if speed > SPEED_STATIC_MAX:
        if theta > 180 - ANGLE_SIDE:
            state = STATE_BACKWARD
        elif theta > ANGLE_SIDE:
            state = STATE_SIDE_WALK
        else:
            state = STATE_FORWARD
    return state

def walk_state_a(speed, theta):
    sta = speed <= SPEED_STATIC_MAX
    state = sta * STATE_STATIC
    backward = theta > 180 - ANGLE_SIDE
    state[~sta & backward] = STATE_BACKWARD
    state[~sta & ~backward & (theta > ANGLE_SIDE)] = STATE_SIDE_WALK
    return state

def angle_points(p1, c, p2):
    return dir_diff((p2[0] - c[0], p2[1] - c[1]), (p1[0] - c[0], p1[1] - c[1]))

def distance_array(p1, p2):
    return np.sqrt(((p1 - p2)**2).sum())

def vec_len(v):
    return np.sqrt((v**2).sum())

def vec_angle(v):
    theta = np.arctan2(v[1], v[0])
    return np.rad2deg(theta)

def dir_diff(body_dir, wing_dir):
    theta = np.arctan2(body_dir[1], body_dir[0]) - np.arctan2(wing_dir[1], wing_dir[0])
    if theta > np.pi:
        theta -= 2 * np.pi
    if theta < -np.pi:
        theta += 2 * np.pi
    return np.rad2deg(theta)

def angle_to_vector(angle):
    d = np.deg2rad(angle)
    return np.array([np.cos(d), np.sin(d)])

def angle_to_vector_l(angle):
    d = np.deg2rad(angle)
    return np.cos(d), np.sin(d)

def vector_to_angle(v):
    return np.rad2deg(np.arctan2(v[1], v[0]))

def get_centered_info(pos1, dir1, pos2):
    if NO_CENTERED_INFO:
        return 0, 0, 0, 0
    dirv = angle_to_vector(dir1)
    v12 = pos2 - pos1

    # theta = np.deg2rad(lim_dir(90 - dir1)) #np.arctan2(dirv[0], dirv[1])
    sin_theta = dirv[0] #np.sin(theta)
    cos_theta = dirv[1] #np.cos(theta)
    rotx = (v12[0] * cos_theta) - (v12[1] * sin_theta)
    roty = (v12[0] * sin_theta) + (v12[1] * cos_theta)

    lenv12 = vec_len(v12)
    t = np.cross(v12, dirv)
    if t > 0:
        phi = np.pi / 2 - np.arccos(np.dot(v12, dirv) / lenv12)
    else:
        phi = np.pi / 2 + np.arccos(np.dot(v12, dirv) / lenv12)
    # phi1 = lim_dir(np.rad2deg(phi))
    # phi2 = lim_dir(np.rad2deg(np.pi/2 + (np.arctan2(v12[1], v12[0]) - np.deg2rad(dir1))))
    # if abs(phi1 - phi2) > 0.01:
    #     print("%.3f %.3f" % (phi1, phi2))
    return rotx, roty, lenv12, np.rad2deg(phi)

def load_feat_csv(feat_file):  # NOTE: must use fpt output -- feat.csv
    meta = load_dict(feat_file.replace("feat.csv", "meta.txt"))
    featc_df = pd.read_csv(feat_file, nrows=DURATION_LIMIT * int(meta["FPS"] + 0.5) * 2)
    if np.isnan(featc_df["2:pos:x"][0]):
        # NOTE: fpt_1 output, 1 fly per row
        featc_df.sort_values("frame", inplace=True)
        featc = featc_df.to_dict(orient="records")
        featc = id_assign_dict(featc)
    else:
        featc = featc_df.to_dict(orient="records")
    return featc, featc_df

def calib_feat(feat_file):
    featc, featc_df = load_feat_csv(feat_file)
    camera = feat_file.split("_")
    calib_info_pickle = None
    if len(camera) > 3:
        camera = camera[-3]
        if camera in ["1", "A"]:
            calib_info_pickle = "calib/calib_DH1207_info.pickle"
        elif camera in ["2", "B"]:
            calib_info_pickle = "calib/calib_PG1207_info.pickle"
    if not calib_info_pickle:
        meta = get_meta(feat_file.replace("feat.csv", "meta.txt"))
        meta["calib"] = False
        save_dict(feat_file.replace("feat.csv", "calib_meta.txt"), meta)
        # save_dataframe(featc_df, feat_file.replace("feat.csv", "calib_feat.pickle"))
        return featc, meta
    del featc_df
    calib_info = pickle.load(open(calib_info_pickle, "rb"))
    mtx, dist, newcameramtx, sz = calib_info

    meta = get_meta(feat_file.replace("feat.csv", "meta.txt"))
    meta["calib"] = True
    roi = meta["ROI"]["roi"]
    x0, y0 = roi[0][0], roi[0][1]
    frames = len(featc)
    points = [roi[0], [roi[1][0], roi[0][1]], [roi[0][0], roi[1][1]], roi[1]]
    u_points = cv2.undistortPoints(np.array([points]).astype(float), mtx, dist, P=newcameramtx)[0]
    u_roi = [[min(u_points[0][0], u_points[2][0]), min(u_points[0][1], u_points[1][1])],
             [max(u_points[1][0], u_points[3][0]), max(u_points[2][1], u_points[3][1])]]
    ux0, uy0 = u_roi[0][0], u_roi[0][1]
    meta["ROI"]["roi"] = u_roi
    save_dict(feat_file.replace("feat.csv", "calib_meta.txt"), meta)
    for fly in [1, 2]:
        for i in tqdm(range(frames), "loop 0_%d calib feat" % fly):
            tc = featc[i]
            points = [[tc["%d:pos:x" % fly] + x0, tc["%d:pos:y" % fly] + y0]]
            u_points = cv2.undistortPoints(np.array([points]), mtx, dist, P=newcameramtx)
            tc["%d:pos:x" % fly], tc["%d:pos:y" % fly] = u_points[0][0][0] - ux0, u_points[0][0][1] - uy0,
            xs = [float(tt) + x0 for tt in tc["%d:point:xs" % fly].split()]
            ys = [float(tt) + y0 for tt in tc["%d:point:ys" % fly].split()]
            points = np.array(list(zip(xs, ys)))
            u_points = cv2.undistortPoints(np.array([points]), mtx, dist, P=newcameramtx)
            u_xs = u_points[0][:, 0] - ux0
            u_ys = u_points[0][:, 1] - uy0
            tc["%d:point:xs" % fly] = " ".join(["%.2f"%x for x in u_xs])
            tc["%d:point:ys" % fly] = " ".join(["%.2f"%x for x in u_ys])
    # save_dataframe(pd.DataFrame(featc), feat_file.replace("feat.csv", "calib_feat.pickle"))
    calib_video_frame(feat_file, calib_info, u_roi, roi)
    return featc, meta

def calib_video_frame(feat_file, calib_info, u_roi, roi, frame=0):
    mtx, dist, newcameramtx, sz = calib_info
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, sz, 5)
    video_file = get_video_in_dir(os.path.dirname(os.path.dirname(feat_file)))
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    img_calib = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    img_calib = img_calib[int(u_roi[0][1]):int(u_roi[1][1]), int(u_roi[0][0]):int(u_roi[1][0])]
    img = img[int(roi[0][1]):int(roi[1][1]), int(roi[0][0]):int(roi[1][0])]
    cap.release()
    cv2.imwrite(feat_file.replace("feat.csv", "calib_on.png"), img_calib)
    cv2.imwrite(feat_file.replace("feat.csv", "calib_off.png"), img)

"""
|-dist_McFc, dir, walk, wing_l, wing_r, theta
|-v_len, v_dir, vs, vf, av, acc, d_pos, d_dir
|-frame, dist_McFc, rel_dir
|-rel_pos_h|c|t:x|y, rel_polar_h|c|t:r|t, rel_pos_hh|ht|th:x|y, rel_polar_hh|ht|th:r|t
|-$dist_MhFh=rel_polar_hh:r $dist_MhFt=rel_polar_ht:r
|-frame, on_edge, we_l, we_r, we_ipsi, phi_m, dist_ht, ht_span, t_span
"""
def calc_frame_stat(feat_file):
    featc, meta = calib_feat(feat_file)
    scale = meta["FEAT_SCALE"]
    roi = meta["ROI"]["roi"]
    max_y = roi[1][1] - roi[0][1]
    center = (roi[1][0] - roi[0][0]) / scale / 2, max_y / scale / 2
    def parse_and_scale(l):
        return [float(tt)/scale for tt in l.split()]
    def parse_and_scale_y(l):
        return [(max_y - float(tt))/scale for tt in l.split()]  # NOTE: flip y
    stats = [[], [], []]  # NOTE: common, fly1, fly2

    frames = len(featc)
    fps = meta["FPS"]
    points_b = [None, None, None]
    wings_b = [None, None, None]
    for i in tqdm(range(frames), "loop 1 calc_frame_stat"):
        tc = featc[i]
        tc["time"] = i/fps
        row = [tc, {"frame": i}, {"frame": i}]
        for fly in (1, 2):
            row[fly]["pos:x"] = tc["%d:pos:x"%fly] / scale
            row[fly]["pos:y"] = (max_y - tc["%d:pos:y"%fly]) / scale  # NOTE: flip y
            row[fly]["pos"] = np.array([row[fly]["pos:x"], row[fly]["pos:y"]])
            row[fly]["e_maj"] = tc["%d:e_maj"%fly] / scale
            row[fly]["e_min"] = tc["%d:e_min"%fly] / scale
            row[fly]["area"] = row[fly]["e_maj"] * row[fly]["e_min"] * np.pi / 4 #tc["%d:area"%fly]
            row[fly]["point:xs"] = parse_and_scale(tc["%d:point:xs"%fly])
            row[fly]["point:ys"] = parse_and_scale_y(tc["%d:point:ys"%fly])
        stats[0].append(row[0])
        stats[1].append(row[1])
        stats[2].append(row[2])

    # delta = int(fps/30)
    for i in tqdm(range(frames), "loop 2 calc_frame_stat"):
        row = stats[0][i], stats[1][i], stats[2][i]
        # tbi = max(0, i - delta)
        # tfi = min(frames - 1, i + delta)
        # step = tfi - tbi
        # row_b = stats[0][tbi], stats[1][tbi], stats[2][tbi]
        # row_f = stats[0][tfi], stats[1][tfi], stats[2][tfi]

        row[0]["dist_McFc"] = distance_array(row[1]["pos"], row[2]["pos"])
        row[0]["overlap"] = int(row[0]["reg_n"] < 2)
        for fly in (1, 2):
            dd = row[fly]  # NOTE: DataFrame random access slow, so use dict
            # pos_c = row[fly]["pos"]
            # pos_b = row_b[fly]["pos"]
            # pos_f = row_f[fly]["pos"]

            xs = dd["point:xs"]
            ys = dd["point:ys"]
            head = xs[0], ys[0]
            thorax = xs[1], ys[1]
            tail = xs[2], ys[2]
            wingl = xs[3], ys[3]
            wingr = xs[4], ys[4]
            dd["dist_ht"] = distance(head, tail)
            dd["dist_c"] = distance(head, center)
            dd["e_maj"] = np.clip(distance(head, tail), FEMALE_AVG_LEN_L, FEMALE_AVG_LEN_H)  # NOTE: use head-tail as body len
            dd["ht_close"] = dd["dist_ht"] < T_HEAD_TAIL_CLOSE_DIST
            if dd["ht_close"] and points_b[fly] is not None:
                # NOTE: head near tail, correct by last frame
                dh = distance(head, points_b[fly][0])
                dt = distance(tail, points_b[fly][2])
                if dh < dt:
                    # NOTE: head is correct, tail use last tail
                    tail = points_b[fly][2]
                    xs[2] = tail[0]
                    ys[2] = tail[1]
                else:
                    head = points_b[fly][0]
                    xs[0] = head[0]
                    ys[0] = head[1]

            body_dir_v = head[0] - tail[0], head[1] - tail[1]
            body_dir = np.rad2deg(np.arctan2(body_dir_v[1], body_dir_v[0]))
            dd["dir"] = body_dir

            wing_l = angle_points(tail, thorax, wingl)
            wing_r = angle_points(tail, thorax, wingr)
            if points_b[fly] is not None:
                if abs(wing_l) > 100 and abs(wings_b[fly][0] - wing_l) > 90:
                    wingl = points_b[fly][3]
                    xs[3], ys[3] = wingl
                    wing_l = angle_points(tail, thorax, wingl)
                if abs(wing_r) > 100 and abs(wings_b[fly][1] - wing_r) > 90:
                    wingr = points_b[fly][4]
                    xs[4], ys[4] = wingr
                    wing_r = angle_points(tail, thorax, wingr)
            dd["wing_l"] = wing_l
            dd["wing_r"] = wing_r

            wings_b[fly] = wing_l, wing_r
            points_b[fly] = head, thorax, tail, wingl, wingr

            # vx = (pos_f[0] - pos_b[0]) / step * fps
            # vy = (pos_f[1] - pos_b[1]) / step * fps
            # v = np.array([vx, vy])
            # v_len = vec_len(v)
            # dd["v_len"] = v_len
            # v_dir = vec_angle(v)
            # dd["v_dir"] = v_dir
            # theta = angle_diff(v_dir, body_dir)
            # dd["theta"] = theta
            # dd["walk"] = walk_state(v_len, abs(theta))
            #
            # theta_r = np.deg2rad(theta)
            # dd["vs"] = v_len * np.sin(theta_r)
            # dd["vf"] = v_len * np.cos(theta_r)

    dirb = [np.nan, np.nan, np.nan]
    for i in tqdm(range(frames), "loop 3 calc_frame_stat"):
        row = [stats[0][i], stats[1][i], stats[2][i]]

        row[0]["rel_dir"] = lim_dir(row[1]["dir"] - row[2]["dir"] % 360)

        # tbi = max(0, i - delta)
        # tfi = min(frames - 1, i + delta)
        # step = tfi - tbi
        for fly in (1, 2):
            # tb = stats[fly][tbi]
            # tf = stats[fly][tfi]
            dd = stats[fly][i]
            dd["d_dir"] = angle_diff(dd["dir"], dirb[fly])
            # dd["av"] = angle_diff(tf["dir"], tb["dir"]) / step * fps
            # dd["acc"] = (tf["v_len"] - tb["v_len"]) / step * fps

            # xs = dd["point:xs"]
            # ys = dd["point:ys"]
            # body_dir = dd["dir"]
            # head = xs[0], ys[0]
            # tail = xs[2], ys[2]
            # pos_c = row[fly]["pos"]
            # row_o = row[3 - fly]
            # pos_o = row_o["pos"]
            # head_o = np.array([row_o["point:xs"][0], row_o["point:ys"][0]])
            # tail_o = np.array([row_o["point:xs"][2], row_o["point:ys"][2]])
            # # 1:rel_pos_h|c|t:x|y, 1:rel_polar_h|c|t:r|t, 1:rel_pos_hh|ht|th:x|y, 1:rel_polar_hh|ht|th:r|t
            # rel_x, rel_y, rel_r, rel_phi = get_centered_info(pos_c, body_dir, pos_o)
            # if i==0 and fly==1:
            #     print(pos_c, body_dir, pos_o)
            #     print(rel_x, rel_y, rel_r, rel_phi)
            # rel_c_h = get_centered_info(pos_c, body_dir, head_o)  # c-h
            # rel_c_t = get_centered_info(pos_c, body_dir, tail_o)  # c-t
            # rel_h_h = get_centered_info(head, body_dir, head_o)  # h-h
            # rel_h_t = get_centered_info(head, body_dir, tail_o)  # h-t
            # # rel_t_h = get_centered_info(tail, body_dir, head_o)  # h-t
            # dd["rel_pos:x"], dd["rel_pos:y"] = rel_x, rel_y
            # dd["rel_polar:r"], dd["rel_polar:t"] = rel_r, rel_phi
            # dd["rel_pos_h:x"], dd["rel_pos_h:y"] = rel_c_h[0], rel_c_h[1]
            # dd["rel_polar_h:r"], dd["rel_polar_h:t"] = rel_c_h[2], rel_c_h[3]
            # dd["rel_pos_t:x"], dd["rel_pos_t:y"] = rel_c_t[0], rel_c_t[1]
            # dd["rel_polar_t:r"], dd["rel_polar_t:t"] = rel_c_t[2], rel_c_t[3]
            # # dd["rel_pos_hh:x"], dd["rel_pos_hh:y"] = rel_h_h[0], rel_h_h[1]
            # dd["rel_polar_hh:r"], dd["rel_polar_hh:t"] = rel_h_h[2], rel_h_h[3]
            # # dd["rel_pos_ht:x"], dd["rel_pos_ht:y"] = rel_h_t[0], rel_h_t[1]
            # dd["rel_polar_ht:r"], dd["rel_polar_ht:t"] = rel_h_t[2], rel_h_t[3]
            # # dd["rel_pos_th:x"], dd["rel_pos_th:y"] = rel_t_h[0], rel_t_h[1]
            # # dd["rel_polar_th:r"], dd["rel_polar_th:t"] = rel_t_h[2], rel_t_h[3]

            wing_l = dd["wing_l"]
            wing_r = dd["wing_r"]
            dd["on_edge"] = int(wing_l * wing_r > 0 and abs(wing_l) > ANGLE_EDGE_MIN and abs(wing_r) > ANGLE_EDGE_MIN)
            # sh = min(1, t["2:e_sh"] / 2 / t["dist_MhFh"])
            # st = min(1, t["2:e_sh"] / 2 / t["dist_MhFt"])
            # t["1:h_span"] = np.rad2deg(np.arctan(sh)) * 2
            # t["1:t_span"] = np.rad2deg(np.arctan(st)) * 2
            we_l, we_r, we_ipsi = 0, 0, 0
            we_ls, we_rs = 0, 0
            phi_la = abs(wing_l)
            phi_ra = abs(wing_r)
            phi_m = max(phi_la, phi_ra)
            if not dd["on_edge"] and not dd["ht_close"]:
                we_l = ANGLE_WE_MAX > phi_la > ANGLE_WE_MIN
                we_ls = ANGLE_WE_MAX > phi_la > ANGLE_WE_MIN_30
                we_r = ANGLE_WE_MAX > phi_ra > ANGLE_WE_MIN
                we_rs = ANGLE_WE_MAX > phi_ra > ANGLE_WE_MIN_30
                # if we_l and not we_r:
                #     we_ipsi = 1 if dd["rel_pos_h:x"] < 0 else -1
                # elif we_r and not we_l:
                #     we_ipsi = 1 if dd["rel_pos_h:x"] > 0 else -1
            dd["we_l"] = int(we_l)
            dd["we_r"] = int(we_r)
            dd["we_lr"] = int(we_l and we_r)
            dd["wing_m"] = phi_m
            dd["we_ipsi"] = we_ipsi
            dd["court"] = int(we_l or we_r)
            dd["court_30"] = int(we_ls or we_rs)
            # dd["ht_span"] = abs(lim_dir(dd["rel_polar_ht:t"] - dd["rel_polar_hh:t"]))
            dirb[fly] = dd["dir"]
    dfs = []
    for i in range(3):
        df = pd.DataFrame(stats[i])
        dfs.append(df)
    calc_center_info(dfs)
    for fly in (1, 2):
        dd = dfs[fly]
        idx_wl = dd["we_l"] & (~dd["we_r"])
        idx_wr = dd["we_r"] & (~dd["we_l"])
        idx_l = dd["rel_pos_h:x"] < 0
        dfs[fly]["we_ipsi"][idx_wl & idx_l] = 1
        dfs[fly]["we_ipsi"][idx_wl & ~idx_l] = -1
        dfs[fly]["we_ipsi"][idx_wr & idx_l] = -1
        dfs[fly]["we_ipsi"][idx_wr & ~idx_l] = 1
    return calc_v(dfs, meta["FPS"]), meta

def get_centered_info_l(center, body_dir, part):
    center_x, center_y = center
    part_x, part_y = part
    d = np.deg2rad(body_dir)
    dirv_x, dirv_y = np.cos(d), np.sin(d)
    vx, vy = part_x - center_x, part_y - center_y

    rotx = (vx * dirv_y) - (vy * dirv_x)
    roty = (vx * dirv_x) + (vy * dirv_y)

    lenv = np.sqrt((vx * vx) + (vy * vy))
    phi = lim_dir_a(np.rad2deg((np.arctan2(vy, vx) - d)))
    # rotx1, roty1, lenv1, phi1 = get_centered_info(np.array([center_x[0], center_y[0]]), body_dir[0], np.array([part_x[0], part_y[0]]))
    return rotx, roty, lenv, phi

def calc_center_info(dfs):
    print("calc_center_info ...")
    head = [None] * 3
    center = [None] * 3
    tail = [None] * 3
    body_dir = [None] * 3
    for fly in (1, 2):
        dd = dfs[fly]
        xs = dd["point:xs"]
        ys = dd["point:ys"]
        pos = dd["pos"]
        head[fly] = xs.apply(lambda x: x[0]), ys.apply(lambda x: x[0])
        tail[fly] = xs.apply(lambda x: x[2]), ys.apply(lambda x: x[2])
        center[fly] = pos.apply(lambda x: x[0]), pos.apply(lambda x: x[1])
        body_dir[fly] = dd["dir"]
    for fly in (1, 2):
        # 1:rel_pos_h|c|t:x|y, 1:rel_polar_h|c|t:r|t, 1:rel_pos_hh|ht|th:x|y, 1:rel_polar_hh|ht|th:r|t
        rel_x, rel_y, rel_r, rel_phi = get_centered_info_l(center[fly], body_dir[fly], center[3-fly])
        rel_c_h = get_centered_info_l(center[fly], body_dir[fly], head[3-fly])  # c-h
        rel_c_t = get_centered_info_l(center[fly], body_dir[fly], tail[3-fly])  # c-t
        rel_h_h = get_centered_info_l(head[fly], body_dir[fly], head[3-fly])  # h-h
        rel_h_t = get_centered_info_l(head[fly], body_dir[fly], tail[3-fly])  # h-t
        # rel_t_h = get_centered_info(tail, body_dir, head[3-fly])  # h-t
        dfs[fly]["rel_pos:x"], dfs[fly]["rel_pos:y"] = rel_x, rel_y
        dfs[fly]["rel_polar:r"], dfs[fly]["rel_polar:t"] = rel_r, rel_phi
        dfs[fly]["rel_pos_h:x"], dfs[fly]["rel_pos_h:y"] = rel_c_h[0], rel_c_h[1]
        dfs[fly]["rel_polar_h:r"], dfs[fly]["rel_polar_h:t"] = rel_c_h[2], rel_c_h[3]
        dfs[fly]["rel_pos_t:x"], dfs[fly]["rel_pos_t:y"] = rel_c_t[0], rel_c_t[1]
        dfs[fly]["rel_polar_t:r"], dfs[fly]["rel_polar_t:t"] = rel_c_t[2], rel_c_t[3]
        # dfs[fly]["rel_pos_hh:x"], dfs[fly]["rel_pos_hh:y"] = rel_h_h[0], rel_h_h[1]
        dfs[fly]["rel_polar_hh:r"], dfs[fly]["rel_polar_hh:t"] = rel_h_h[2], rel_h_h[3]
        # dfs[fly]["rel_pos_ht:x"], dfs[fly]["rel_pos_ht:y"] = rel_h_t[0], rel_h_t[1]
        dfs[fly]["rel_polar_ht:r"], dfs[fly]["rel_polar_ht:t"] = rel_h_t[2], rel_h_t[3]
        # dfs[fly]["rel_pos_th:x"], dfs[fly]["rel_pos_th:y"] = rel_t_h[0], rel_t_h[1]
        # dfs[fly]["rel_polar_th:r"], dfs[fly]["rel_polar_th:t"] = rel_t_h[2], rel_t_h[3]

def calc_v(dfs, fps, post_id_correct=False):
    print("fps=", fps)
    d = int(fps/30.0 + 0.5)
    fps_scale = fps / (d*2)
    for fly in (1, 2):
        df = dfs[fly]
        i = df.index
        dfb = df.reindex(i - d, method="nearest")
        dff = df.reindex(i + d, method="nearest")
        xb, yb = np.array(dfb["pos:x"]), np.array(dfb["pos:y"])
        xf, yf = np.array(dff["pos:x"]), np.array(dff["pos:y"])
        vx, vy = xf - xb, yf - yb

        v_len = np.sqrt(vx**2 + vy**2) * fps_scale
        v_dir = np.rad2deg(np.arctan2(vy, vx))
        theta = angle_diff_a(v_dir, np.array(df["dir"]))
        df["theta"] = theta
        df["av"] = angle_diff_a(np.array(dff["dir"]), np.array(dfb["dir"])) * fps_scale
        df["v_len"] = v_len
        df["v_dir"] = v_dir
        df["walk"] = walk_state_a(v_len, np.fabs(theta))

        if post_id_correct:
            theta_r = np.deg2rad(theta)
            df["vs"] = v_len * np.sin(theta_r)
            df["vf"] = v_len * np.cos(theta_r)
            accx, accy = (vx[d:] - vx[:-d]), (vy[d:] - vy[:-d])
            acc_len = np.sqrt(accx**2, accy**2) * fps_scale
            df["acc"] = np.hstack([[np.nan]*d, (v_len[d:] - v_len[:-d]) * fps_scale])
            df["acc_dir"] = np.hstack([[np.nan]*d, np.rad2deg(np.arctan2(accy, accx))])
            df["acc_len"] = np.hstack([[np.nan]*d, acc_len])
    return dfs

def get_row_of_dict(d, k):
    return np.array([dd[k] for dd in d])

def detect_behavior_and_correct_id(dfs, fps):
    # stat4
    #     frame, 1:walk, 1:court, 1:copulate, 1:circle
    dfs0 = dfs[0]
    behs = [{}, {}, {}]
    frames = len(dfs0)
    cop_frame_min_frame = COP_DURATION_MIN * fps
    cir_frame_min_frame = CIR_DURATION_MIN * fps
    overlap = dfs0["overlap"]
    overlap_s = smooth_sequence(overlap, OVERLAP_CONTINUE_T*fps, 0, 1)
    copulate_s = smooth_sequence(overlap_s, cop_frame_min_frame, 1, 0)  # NOTE: keep overlap longer than 1min
    behs[0]["copulate"] = copulate_s

    court_s_l = []
    for fly in (1, 2):
        dfs1 = dfs[fly]
        walk = dfs1["walk"] == STATE_SIDE_WALK
        sidewalk_s = smooth_sequence(walk, SIDE_WALK_CONTINUE_T*fps)
        court_s = smooth_sequence(dfs1["court"] == 1, COURT_N_CONTINUE, True, False)  # shrink court
        court_s = smooth_sequence(court_s, COURT_CONTINUE_T*fps)
        court_s_l.append(court_s)
        behs[fly]["sidewalk_s"] = sidewalk_s
        behs[fly]["court_s"] = court_s
        # NOTE: cir conditions:
        #   in side-walking
        #   in courtship
        #   not in copulation
        #   not overlap
        court_s_30 = smooth_sequence(dfs1["court_30"] == 1, COURT_N_CONTINUE, True, False)  # shrink court
        court_s_30 = smooth_sequence(court_s_30, COURT_CONTINUE_T*fps)
        overlap_s = smooth_sequence(overlap, OVERLAP_CONTINUE_T*fps, 1, 0)  # shrink overlap
        circle_s1 = smooth_sequence(sidewalk_s & court_s_30 & ~overlap_s, CIR_CONTINUE)  #
        circle_s = smooth_sequence(circle_s1, cir_frame_min_frame, True, False)  # NOTE: keep side walk longer than 0.4s

        behs[fly]["court_s_30"] = court_s_30
        behs[fly]["sidewalk_s"] = sidewalk_s
        behs[fly]["circle_s1"] = circle_s1
        behs[fly]["circle"] = circle_s

    court_as_male = np.zeros(frames, dtype=int)
    court_as_male[court_s_l[0]] = 1
    court_as_male[court_s_l[1]] = 2
    court_as_male[court_s_l[0] & court_s_l[1]] = 0
    behs[0]["court_as_male"] = court_as_male
    behs[0]["court_infer_male"] = infer_male(court_as_male, overlap, frames)

    if FLY_NUM > 1:
        correct_id(dfs, behs)  # [WJL] NOTE: use court_infer_male to correct identity
    calc_v(dfs, fps, True)  # NOTE: cale v again after correction

    dist_McFc = dfs0["dist_McFc"]
    for fly in (1, 2):
        # NOTE: recheck circle
        #   male:rel_pos:y > 0 TODO
        #   #accumulated motion,rotation > ?
        #   v,av > 2.6,50
        #   inter-fly distance < 5
        #   side ratio > 0.5
        #   dir std > 12
        dfs1 = dfs[fly]
        circle_s = behs[fly]["circle"]
        cir_bouts = calc_bouts(circle_s, True)
        v_len = dfs1["v_len"]
        walk = dfs1["walk"]
        av = dfs1["av"]
        pos_x = dfs1["pos:x"]
        pos_y = dfs1["pos:y"]
        body_dir = dfs1["dir"]
        for s, e in cir_bouts:
            l = e - s
            if l < CIR_DURATION_MIN * fps:
                circle_s[s:e] = False
                # print("%d %d length=%.2f" % (s, e, l))
                continue
            side_r = np.count_nonzero(walk[s:e]) / l
            if side_r < CIR_SIDE_RATIO_MIN:
                circle_s[s:e] = False
                # print("%d %d side_r=%.2f" % (s, e, side_r))
                continue
            v1 = np.mean(v_len[s:e])
            if v1 < CIR_RECHECK_SPEED:
                # print("%d %d v1=%.2f" % (s, e, v1))
                circle_s[s:e] = False
                continue
            av1 = np.mean(np.abs(av[s:e]))
            if av1 < CIR_RECHECK_AV:
                circle_s[s:e] = False
                # print("%d %d av1=%.2f" % (s, e, av1))
                continue
            xs, ys = pos_x[s:e], pos_y[s:e],
            x_lim = np.max(xs) - np.min(xs)
            y_lim = np.max(ys) - np.min(ys)
            if x_lim < CIR_MIN_X_RANGE and y_lim < CIR_MIN_X_RANGE:
                circle_s[s:e] = False
                # print("%d %d x_lim=%.2f y_lim=%.2f" % (s, e, x_lim, y_lim))
                continue
            dir_std = np.std(correct_angle(np.array(body_dir[s:e])))
            if dir_std < CIR_MIN_DIR_STD:
                circle_s[s:e] = False
                # print("%d %d dir_std=%.2f" % (s, e, dir_std))
                continue
            dist1 = np.mean(dist_McFc[s:e])
            if dist1 > CIR_RECHECK_DIST_MAX:
                circle_s[s:e] = False
                # print("%d %d dist1=%.2f" % (s, e, dist1))
                continue
        behs[fly]["circle"] = circle_s

    for i in range(3):
        for k in behs[i].keys():
            dfs[i][k] = behs[i][k]
    return dfs

# def correct_id(stats, behs):
#     frames = len(stats[0])
#     last_male = 1
#     count = 0
#     print("\ncorrect_id loop...")
#     for i in tqdm(range(frames)):
#         male = behs[0]["court_infer_male"][i]
#         if male != 0:
#             last_male = male
#         if last_male == 2:
#             stats[1][i], stats[2][i] = stats[2][i], stats[1][i]
#             for k in behs[1].keys():
#                 behs[1][k][i], behs[2][k][i] = behs[2][k][i], behs[1][k][i]
#             count += 1
#     print("\ncorrect_id [%d] frames" % count)
#     return stats, behs

def correct_id(dfs, behs):
    frames = len(dfs[0])
    for last_male in behs[0]["court_infer_male"]:
        if last_male > 0:
            break
    i_l = []
    for i in tqdm(range(frames), "loop 4 correct_id"):
        male = behs[0]["court_infer_male"][i]
        if male != 0:
            last_male = male
        if last_male == 2:
            i_l.append(i)

    dfs[1].iloc[i_l], dfs[2].iloc[i_l] = dfs[2].iloc[i_l], dfs[1].iloc[i_l]
    for i in i_l:
        for k in behs[1].keys():
            behs[1][k][i], behs[2][k][i] = behs[2][k][i], behs[1][k][i]
    print("\ncorrect_id [%d] frames" % len(i_l))
    return dfs, behs

def smooth_sequence(a, inter, remove_v=False, fill_v=True):
    # NOTE: replace continuous "remove_v" shorter than "inter" by "fill_v"
    c = 0
    ret = []
    for i, v in enumerate(a):
        condition = v == remove_v
        if condition:
            c += 1
        else:
            if c <= inter:
                ret.extend([fill_v] * c)
            else:
                ret.extend([remove_v] * c)
            c = 0
            ret.append(v)
    if c > 0:
        if c <= inter:
            ret.extend([fill_v] * c)
        else:
            ret.extend([remove_v] * c)
    return np.array(ret)

def infer_male(court_as_male, overlap, frames):
    # NOTE: infer male by extend court_as_male to non-overlap ranges
    # NOTE: overlap--no_court--overlap not corrected
    court_infer_male = np.zeros(frames, dtype=int)
    p = 0
    last_m = 0
    end = frames - 1
    for i, ov in enumerate(overlap):
        male = court_as_male[i]
        if male == 0:
            if ov:
                last_m = 0
                p = 0
            else:
                if last_m:
                    court_infer_male[i] = last_m
                else:
                    p += 1
        if male != 0 or i == end:
            last_m = male
            court_infer_male[i] = last_m
            if p > 0:
                for pp in range(p):
                    court_infer_male[i - pp - 1] = last_m
                p = 0
    return court_infer_male

def test_smooth_inter(seq):
    import matplotlib.pyplot as plt
    r = []
    b = []
    for inter in range(100):
        sidewalk_s = smooth_sequence(seq, inter)
        r.append(np.count_nonzero(sidewalk_s)/len(sidewalk_s))
        bouts = len(calc_bouts(sidewalk_s, True))
        b.append(bouts)
    plt.title("ratio")
    plt.plot(r)
    plt.show()
    plt.title("bouts")
    plt.plot(b)
    plt.show()
    # court_s = smooth_sequence(df[1]["court"] == 1, COURT_CONTINUE)

def revise_stat(stat):
    df = load_dataframe(stat)
    df["wing_m"] = np.maximum(df["wing_l"].abs(), df["wing_r"].abs())
    df["we_ipsi"] = -df["we_ipsi"]
    save_dataframe(df, stat)
    return df

ENGAGED_DIST = 6
ENGAGED_CONTINUE_T = 0.1
FMOTION_VLEN = 3
FMOTION_CONTINUE_T = 0.1
AC_CONTINUE_T = 0.1
AC_DURATION_T = 0.3
WE_N_CONTINUE_T = 0.05
WE_CONTINUE_T = 0.1
FAB_CONTINUE_T = 0.1
def shrink_and_smooth(s, fps, continue_t=0.1, n_continue_t=0.05):
    s = smooth_sequence(s, n_continue_t*fps, True, False)
    s = smooth_sequence(s, continue_t*fps)
    return s

INTERACT_V = 10
def detect_interaction(cir_meta, dfs, path=None):
    # NOTE: cir_bouts1, copulate
    # NOTE: engaged, center, we_on_l, we_on_r, we_l, we_r, ac, acp, fmotion, fabl, fabr
    """
    cir: circling
    we_l: left wing extension
        not on edge
        120 > wing_l > 45
    we_l_startp: left wing start
        not on edge
        wing_l > 0
        wing_l - wing_l(t-0.2) > 30
    we_on_l: wing extension on left
        we_l or we_r
        2:rel_pos:x < -0.5
    wl_g_wr: left wing greater than right wing
        wing_l > wing_r
    engage: close to each other
        dist_MF < 6
    fmotion: female motion (inaccurate)
        v_F > 3
    fabl: female abdomen bending on left (inaccurate when overlapping)
        dist_MF < 6
        dtc_F < 6.5
        (wing_l-wing_r) > 10
    fabl_far:
        fabl
        2:rel_polar_h:r" > 2.7
    ac: attempt copulation
        overlap > 60s
    acp: attempt copulation peak
        ac & vf_M > 5
    """
    if cir_meta.get("interact_v") == INTERACT_V:
        return
    fps = cir_meta["FPS"]
    dfs0 = dfs[0]

    center_s = shrink_and_smooth((dfs[1]["dist_c"] < DIST_TO_CENTER_THRESHOLD_MALE) & (dfs[2]["dist_c"] < DIST_TO_CENTER_THRESHOLD_FEMALE), fps)
    cir_meta["center_bouts"] = calc_bouts(center_s)

    engaged_s = shrink_and_smooth(dfs0["dist_McFc"] < ENGAGED_DIST, fps)
    cir_meta["engaged_bouts"] = calc_bouts(engaged_s)

    c_wing_l = correct_wing(-dfs[1]["wing_l"])
    c_wing_r = correct_wing(dfs[1]["wing_r"])
    cir_meta["wl_g_wr"] = calc_bouts(c_wing_l > c_wing_r)
    wls, wlp = detect_we_start(c_wing_l, np.array(dfs[1]["on_edge"]))
    cir_meta["we_l_start"] = wls
    cir_meta["we_l_startp"] = wlp

    courtl_s = dfs[1]["court_s"] & (dfs[2]["rel_pos:x"] < -0.5)
    courtr_s = dfs[1]["court_s"] & (dfs[2]["rel_pos:x"] > 0.5)
    cir_meta["we_on_l_bouts"] = calc_bouts(shrink_and_smooth(courtl_s, fps))  # courtl_bouts
    cir_meta["we_on_r_bouts"] = calc_bouts(shrink_and_smooth(courtr_s, fps))

    wl_s = shrink_and_smooth(dfs[1]["we_l"] == 1, fps)
    cir_meta["we_l_bouts"] = calc_bouts(wl_s)  # wl_bouts
    wr_s = shrink_and_smooth(dfs[1]["we_r"] == 1, fps)
    cir_meta["we_r_bouts"] = calc_bouts(wr_s)

    fmotion_s = shrink_and_smooth(dfs[2]["v_len"] > FMOTION_VLEN, fps)
    cir_meta["fmotion_bouts"] = calc_bouts(fmotion_s)

    wing_difr = dfs[2]["wing_l"] + dfs[2]["wing_r"]  # NOTE wing_l < 0
    fabl_s = (dfs[2]["dist_c"] < DIST_TO_CENTER_THRESHOLD_FEMALE) & (wing_difr > 10) & (dfs0["dist_McFc"] < ENGAGED_DIST)
    cir_meta["fabl_bouts"] = calc_bouts(shrink_and_smooth(fabl_s, fps))
    fabr_s = (dfs[2]["dist_c"] < DIST_TO_CENTER_THRESHOLD_FEMALE) & (wing_difr < -10) & (dfs0["dist_McFc"] < ENGAGED_DIST)
    cir_meta["fabr_bouts"] = calc_bouts(shrink_and_smooth(fabr_s, fps))
    fabl_far_s = fabl_s & (dfs[2]["rel_polar_h:r"] > DIST_TO_FLY_FAR)
    cir_meta["fabl_far_bouts"] = calc_bouts(shrink_and_smooth(fabl_far_s, fps))
    fabr_far_s = fabr_s & (dfs[2]["rel_polar_h:r"] > DIST_TO_FLY_FAR)
    cir_meta["fabr_far_bouts"] = calc_bouts(shrink_and_smooth(fabr_far_s, fps))

    ac_s = shrink_and_smooth(dfs0["overlap"] == 1, fps)
    ac_bouts = calc_bouts(ac_s)
    acp_bouts = []
    for s, e in ac_bouts:
        acp = dfs[1]["vf"][s:e] > 5
        acp_b = calc_bouts(acp, True)
        if acp_b:
            acp_bouts.extend([[pb[0] + s, pb[1] + s] for pb in acp_b])

    cir_meta["ac_bouts"] = ac_bouts
    cir_meta["acp_bouts"] = acp_bouts
    cir_meta["interact_v"] = INTERACT_V
    path and save_dict(path, cir_meta)

def detect_we_start(wing_r, on_edge, fps=66):
    dt, da = 0.2, 30
    df = int(dt * fps)
    dw = wing_r[df:] - wing_r[:-df]
    b = (dw > da) & (wing_r[df:] > 0) & (wing_r[:-df] > 0) & ~on_edge[df:] & ~on_edge[:-df]
    ret = calc_bouts(b)
    wls_l = []
    wlp_l = []
    for s, e in ret:
        if e - s > 1:
            w = wing_r[s:s+df]
            p = s + int(np.argmax(w[1:] - w[:-1]))
        else:
            p = s
        wls_l.append((p-33, p+33))
        wlp_l.append((p, p+1))
    return wls_l, wlp_l
    # return [(s, s + df) for s, e in ret]

if __name__ == '__main__':
    import sys
    calib_feat(sys.argv[1])
    # stats = calc_frame_stat(sys.argv[1])
    # get_row_of_dict(stats[0], "overlap").dump("st_overlap")
    # get_row_of_dict(stats[1], "walk").dump("st_walk")
    # get_row_of_dict(stats[1], "court").dump("st_court")
    # test_smooth_inter(np.load("st_overlap") == 1)
    # sidewalk_s = smooth_sequence(np.load("st_walk") == STATE_SIDE_WALK, SIDE_WALK_CONTINUE)
    # court_s = smooth_sequence(np.load("st_court") == 1, COURT_CONTINUE)
    # test_smooth_inter(sidewalk_s & court_s & ~np.load("st_overlap"))
    # df = revise_stat(sys.argv[1])
#     tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1
# ,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
# ,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
#     r = smooth_sequence(tt, 2, 0, 1)
#     print(r, len(tt), len(r))
    # print(calc_bouts(tt, 1))
    # mm = [0,0,0,0,1,1,2,2,0,0,0,0,0,0,2,1,1]
    # oo = [0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0]
    # print(infer_male(mm, oo, len(mm)))
