# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from matplotlib.widgets import Button, TextBox, Slider, CheckButtons, RadioButtons
from fpt_consts import *
from fpt_util import load_dict, save_dict, save_dataframe, parse_float_list

LIM_RECT = [95, 95, 140, 140]

def load_feat(feat_file):
    df = pd.read_csv(feat_file)#, nrows=10000)
    df.sort_values("frame", inplace=True)
    df = df.reset_index(drop=True)
    return df

def vec_len(v):
    return np.sqrt((v**2).sum())

def vec_angle(v):
    theta = np.arctan2(v[1], v[0])
    return np.rad2deg(theta)


class FixPoseUI(object):
    def __init__(self, name, figsize, cap, meta_name, feat_name):
        self.points_l = [[0, [None, None, None, None, None]], ]
        self.cur_points = None
        self.points_patch = [None, None, None, None, None]
        self.hint_patch = None
        self.cur_idx = 0
        self.input_int = 0
        self.frame = 0
        self.clicked = False
        self.is_lim = False

        self.cap = cap
        self.cap_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.meta_name = meta_name
        self.feat_name = feat_name
        self.meta_info = load_dict(meta_name)
        if self.meta_info.get("fix_pose") is not None:
            self.points_l = self.meta_info.get("fix_pose")
        self.feat_df = load_feat(feat_name)
        self.total_frame = self.meta_info["total_frame"]
        # self.feat_scale = self.meta_info["FEAT_SCALE"]
        self.roi = np.array(self.meta_info["ROI"]["roi"]).astype(int)
        self.roi_size = (self.roi[1][0] - self.roi[0][0], self.roi[1][1] - self.roi[0][1])
        # self.roi_size_scale = (self.roi_size[0] / self.feat_scale, self.roi_size[1] / self.feat_scale)

        fig, self.cap_ax = plt.subplots(figsize=figsize, num=name)
        plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.15)#, hspace=0.06, wspace=0.06)
        self.slider_ax = plt.axes([0.1, 0.05, 0.75, 0.03])
        self.slider = Slider(self.slider_ax, "", valmin=0, valmax=self.total_frame - 1, valfmt="%d", valinit=0)
        self.slider.on_changed(self.on_slider)
        fig.canvas.mpl_connect('key_press_event', self.onkey)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        fig.canvas.mpl_connect("close_event", self.onclose)
        self.btn_lim = Button(plt.axes([0.88, 0.3, 0.1, 0.05]), "lim")
        self.btn_lim.on_clicked(self.on_click_lim)
        self.btn_ins = Button(plt.axes([0.88, 0.5, 0.1, 0.05]), "Ins")
        self.btn_ins.on_clicked(self.on_click_ins)

    def show(self):
        self.plot_one_frame()
        plt.show()

    def onclose(self, event):
        if self.clicked:
            self.save_result()

    def save_result(self):
        self.meta_info["fix_pose"] = self.points_l
        save_dict(self.meta_name, self.meta_info)

        last_frame, last_points = 0, None
        info = []
        for frame, points in self.points_l:
            info.append([last_frame, frame, last_points])
            last_frame, last_points = frame, points
        info.append([last_frame, len(self.feat_df) - 1, last_points])

        for start, end, points in info:
            if start == end:
                continue
            points = np.array(points)
            if np.isnan(points).any():
                print("invalid:", points)
                return
            v_dir = points[2] - points[0]
            v_hor = points[4] - points[3]
            e_maj = vec_len(v_dir)
            e_min = vec_len(v_hor)
            self.feat_df.loc[start: end, "2:area"] = round(e_maj * e_min, 2)
            self.feat_df.loc[start: end, "2:pos:x"] = round(points[1][0], 2)
            self.feat_df.loc[start: end, "2:pos:y"] = round(points[1][1], 2)
            self.feat_df.loc[start: end, "2:ori"] = round(vec_angle(v_dir), 2)
            self.feat_df.loc[start: end, "2:e_maj"] = round(e_maj, 2)
            self.feat_df.loc[start: end, "2:e_min"] = round(e_min, 2)
            self.feat_df.loc[start: end, "2:point:xs"] = " ".join([("%.2f" % i) for i, j in points])
            self.feat_df.loc[start: end, "2:point:ys"] = " ".join([("%.2f" % j) for i, j in points])

        self.feat_df["reg_n"] = 2
        save_dataframe(self.feat_df, self.feat_name)

    def on_click_ins(self, event):
        points = [self.frame, [None, None, None, None, None]]
        self.points_l.append(points)
        self.points_l.sort(key=lambda x: x[0])
        self.plot_points()

    def get_cur_points(self):
        last_p = self.points_l[0]
        for p in self.points_l:
            if p[0] > self.frame:
                break
            last_p = p
        return last_p

    def on_click_lim(self, event):
        self.is_lim = not self.is_lim
        self.plot_one_frame()

    def onclick(self, event):
        if event.xdata and event.ydata and event.inaxes == self.cap_ax:
            self.clicked = True
            self.cur_points[self.cur_idx] = [event.xdata, event.ydata]
            self.cur_idx = (self.cur_idx + 1) % 5
            self.plot_points()

    def onkey(self, event):
        print(event.key)
        if event.key == "left":
            self.frame -= 1
        elif event.key == "right":
            self.frame += 1
        elif event.key == "h":
            self.frame += 10
        elif event.key == "j" or event.key == "down":
            self.frame += 100
        elif event.key == "k":
            self.frame += 1000
        elif event.key == "l":
            self.frame += 10000
        elif event.key == "y":
            self.frame -= 10
        elif event.key == "u" or event.key == "up":
            self.frame -= 100
        elif event.key == "i":
            self.frame -= 1000
        elif event.key == "o":
            self.frame -= 10000
        elif event.key == "enter":
            self.frame = self.input_int
            self.input_int = 0
        elif event.key in [*"1234567890"]:
            self.input_int = int(event.key) + self.input_int * 10
            print("input: %d" % self.input_int)
            return
        elif event.key in ["f1", "f2", "f3", "f4", "f5"]:
            self.cur_idx = int(event.key[1]) - 1
            self.plot_points()
            return
        else:
            return
        if self.frame >= self.total_frame:
            self.frame = self.total_frame - 1
        if self.frame < 0:
            self.frame = 0
        self.slider.set_val(self.frame)
        
    def on_slider(self, val):
        self.frame = int(val)
        self.plot_one_frame()

    def plot_one_frame(self):
        self.cap_ax.cla()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
        ret, img = self.cap.read()
        if ret:
            img_gray = img[:, :, 1][self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]]
            self.cap_ax.imshow(img_gray, cmap=plt.cm.gray, norm=NoNorm())#, extent=(0, self.roi_size_scale[0], 0, self.roi_size_scale[1]))

        if self.frame < len(self.feat_df):
            track = self.feat_df.iloc[self.frame]
            self.cap_ax.scatter([track["1:pos:x"]], [track["1:pos:y"]], c="b", marker="o")
            xs, ys = parse_float_list(track["1:point:xs"]), parse_float_list(track["1:point:ys"])
            self.cap_ax.scatter(xs, ys, s=35, c="ywmkg", marker="x")
            self.cap_ax.plot(xs[:3], ys[:3], linewidth=1, c="w")
            self.cap_ax.plot(xs[1:4:2], ys[1:4:2], linewidth=1, c="w")
            self.cap_ax.plot(xs[1:5:3], ys[1:5:3], linewidth=1, c="w")
        if self.is_lim:
            self.cap_ax.set_xlim(LIM_RECT[0], LIM_RECT[2])
            self.cap_ax.set_ylim(LIM_RECT[3], LIM_RECT[1])

        t_sec = self.frame / self.cap_fps
        self.slider_ax.set_xlabel("%02d:%02.2f" % (t_sec / 60, t_sec % 60))
        print("#%d" % self.frame)
        self.plot_points()

    def remove_all_patch(self):
        for i, pa in enumerate(self.points_patch):
            if pa:
                pa.remove()
                self.points_patch[i] = None
        if self.hint_patch:
            self.hint_patch.remove()
            self.hint_patch = None

    def plot_points(self):
        start, self.cur_points = self.get_cur_points()
        self.remove_all_patch()
        for i, p in enumerate(self.cur_points):
            if p is not None:
                pa = plt.Circle(p, 0.2, edgecolor="yrmkg"[i], fill=False)
                self.points_patch[i] = pa
                self.cap_ax.add_patch(pa)
        p = self.cur_points[self.cur_idx]
        if p is not None:
            self.hint_patch = self.cap_ax.add_patch(plt.Circle(p, 3, edgecolor="w", fill=False))
        self.cap_ax.set_title("start: %d" % start)
        plt.draw()

if __name__ == '__main__':
    import sys
    video_folder = sys.argv[1]

    video_name = os.path.basename(video_folder)
    cap = cv2.VideoCapture(os.path.join(video_folder, video_name + ".avi"))

    for i in range(1,4):
        pair_folder = os.path.join(video_folder, str(i))
        if not os.path.exists(pair_folder):
            break
        meta = os.path.join(pair_folder, "%s_%d_meta.txt" % (video_name, i))
        feat = os.path.join(pair_folder, "%s_%d_feat.csv" % (video_name, i))
        df = load_feat(feat)
        # df["reg_n"] = 2
        # save_dataframe(df, feat)
        ui_exp = FixPoseUI("fix_pose %d" % i, (4, 4), cap, meta, feat)
        ui_exp.show()
    cap.release()
