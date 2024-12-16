import os
import sys
import cv2
from threading import Timer, Thread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from matplotlib.widgets import Slider
from fpt_util import load_dict, load_dfs, load_dataframe

NEED_CONSOLE = False
PLOT_SIMPLE = False
PLOT_FEAT_LENGTH = 100
PLOT_FEAT_ROWS = 15
PARENT = None  #r"D:\exp\video_test\20190902_150503_A"
class ViewStat(object):
    @staticmethod
    def is_video_exist(stat0_file, meta_file):
        meta = load_dict(meta_file)
        parent = os.path.dirname(stat0_file)
        pp = PARENT or os.path.dirname(parent)
        video_file = os.path.join(pp, os.path.basename(pp) + ".avi")
        return os.path.exists(video_file)

    def __init__(self, stat0_file, meta_file):
        self.frame = 0
        self.caption = []
        self.input_int = 0
        self.cur_play_frame = -1
        self.timer = None
        self.in_update = False
        self.last_x_range = None

        self.meta = load_dict(meta_file)
        parent = os.path.dirname(meta_file)
        pp = PARENT or os.path.dirname(parent)
        video_file = os.path.join(PARENT or os.path.dirname(parent), os.path.basename(pp) + ".avi")
        if PLOT_SIMPLE:
            dfs = load_dfs(stat0_file)
            self.dfs = dfs, dfs, dfs
        else:
            self.dfs = load_dfs(stat0_file)

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap:
            print("open video failed! %s" % video_file)
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap_size = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.feat_scale = self.meta["FEAT_SCALE"]
        self.roi = np.array(self.meta["ROI"]["roi"]).astype(int)
        self.roi_size = (self.roi[1][0] - self.roi[0][0], self.roi[1][1] - self.roi[0][1])
        self.roi_size_scale = (self.roi_size[0] / self.feat_scale, self.roi_size[1] / self.feat_scale)

        self.fig, self.cap_ax = plt.subplots(figsize=(6, 5), num="+/-: next/previous circling")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.2)#, hspace=0.06, wspace=0.06)
        self.slider_ax = plt.axes([0.1, 0.05, 0.78, 0.03])
        self.slider = Slider(self.slider_ax, "", valmin=0, valmax=self.total_frame - 1, valfmt="%d", valinit=0)
        self.slider.on_changed(self.on_slider)
        self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        self.fig.canvas.mpl_connect("close_event", self.onclose)
        self.set_window_pos(600, 600)

        fig0, ax0 = plt.subplots(PLOT_FEAT_ROWS, 1, sharex=True, figsize=(6, 5), num="stat0")
        plt.subplots_adjust(hspace=0, top=1, bottom=0.05, right=0.99)
        self.set_window_pos(600, 0)
        fig0.canvas.mpl_connect("button_press_event", self.on_click_stat)
        fig0.canvas.mpl_connect("key_press_event", self.onkey)
        self.fig.canvas.mpl_connect("close_event", self.onclose)
        self.axes = [ax0]
        self.figs = [fig0]

        if not PLOT_SIMPLE:
            fig1, ax1 = plt.subplots(2*PLOT_FEAT_ROWS, 1, sharex=True, figsize=(6, 12), num="stat1")
            plt.subplots_adjust(hspace=0, top=1, bottom=0.05, right=0.99)
            self.set_window_pos(0, 0)
            fig1.canvas.mpl_connect("button_press_event", self.on_click_stat)
            fig0.canvas.mpl_connect("key_press_event", self.onkey)
            self.fig.canvas.mpl_connect("close_event", self.onclose)
            self.axes.append(ax1)
            self.figs.append(fig1)

    def set_window_pos(self, x, y):
        mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(x, y)#wm_geometry("+%d+%d" % (x+500, y+200))

    def plot_one_frame(self):
        # video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
        ret, img = self.cap.read()
        if ret:
            img_gray = img[:, :, 1][self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]]
            self.cap_ax.cla()
            self.cap_ax.imshow(img_gray, cmap=plt.cm.gray, norm=NoNorm(), extent=(0, self.roi_size_scale[0], 0, self.roi_size_scale[1]))
            #self.cap_ax.invert_yaxis()
            track = None, self.dfs[1].iloc[self.frame], self.dfs[2].iloc[self.frame]
            for i in (1, 2):
                pre = ("%d:" % i) if PLOT_SIMPLE else ""
                c = "b" if i == 1 else "r"
                self.cap_ax.scatter([track[i][pre + "pos:x"]], [track[i][pre + "pos:y"]], c=c, marker="o")
                xs, ys = track[i][pre + "point:xs"], track[i][pre + "point:ys"]
                self.cap_ax.scatter(xs, ys, s=35, c="ywmkg", marker="x" if i == 0 else "o")

                self.cap_ax.plot(xs[:3], ys[:3], linewidth=1, c="w")
                self.cap_ax.plot(xs[1:4:2], ys[1:4:2], linewidth=1, c="w")
                self.cap_ax.plot(xs[1:5:3], ys[1:5:3], linewidth=1, c="w")
                self.plot_body_box(track[i][pre + "pos:x"], track[i][pre + "pos:y"], track[i][pre + "dir"], track[i][pre + "e_maj"]/2, track[i][pre + "e_min"]/2, c)
        t_sec = self.frame / self.cap_fps
        self.slider_ax.set_xlabel("%02d:%02.2f" % (t_sec / 60, t_sec % 60))
        print("#%d" % self.frame)

    def plot_body_box(self, x, y, d, h_maj, h_min, c):
        return
        print(x, y, d, h_maj, h_min)
        t = np.deg2rad(d)
        dx1, dy1 = h_maj*np.cos(t), h_maj*np.sin(t)
        dx2, dy2 = -h_min*np.sin(t), h_min*np.cos(t)
        self.cap_ax.plot([x+dx1, x+dx2, x-dx1, x-dx2, x+dx1], [y+dy1, y+dy2, y-dy1, y-dy2, y+dy1], linewidth=0.5, c=c)

    stat0_keys = ["court_as_male", "court_infer_male", "overlap", "copulate",
                  "dist_McFc", "rel_dir", ]  #, "reg_n"
    stat1_keys = ["court", "circle", "circle_s1", "sidewalk_s", "court_s_30", "walk", "we_ipsi", "on_edge",
                  "dir", "theta", "v_dir",
                  "v_len", "vf", "vs", "av", "acc_len", "wing_l", #"acc_dir", #"acc",
                  # "wing_m", "wing_l", "we_l", "wing_r", "we_r",
                  # "ht_span", "dist_ht",
                  "rel_pos:x", "rel_pos:y", "rel_polar:r", "rel_polar:t", "dist_c",
                  "rel_pos_h:x", "rel_pos_h:y", "rel_polar_h:r", "rel_polar_h:t",
                  "rel_pos_t:x", "rel_pos_t:y", "rel_polar_t:r", "rel_polar_t:t"
                  # "rel_polar_t:r", "rel_polar_t:t",
                  # "rel_polar_hh:r", "rel_polar_ht:r"
                  # "court_s_30", "court_s",
                  # "e_maj", "area", "pos:x", "acc"
                  ]
    feat_keys = [stat0_keys, stat1_keys, stat1_keys]
    def plot_feat(self):
        if PLOT_SIMPLE:
            return self.plot_feat_simple()

        half = PLOT_FEAT_LENGTH/2
        center = np.clip(self.frame, half, self.total_frame - half)
        x_range = (int(center - half), int(center + half))
        for idx, feat in enumerate([self.dfs[0], self.dfs[1]]): # TODO
            f = feat.iloc[self.frame]
            if idx == 1:
                f2 = self.dfs[2].iloc[self.frame]
            keys = ViewStat.feat_keys[idx]#feat.keys()
            i = 0
            for k in keys:
                ax = self.axes[idx][i]
                ax.cla()
                if feat.get(k) is not None:
                    leg, = ax.plot(feat[k][x_range[0]:x_range[1]], lw=0.6, label="%s %.2f" % (k, f[k]))
                    ax.scatter([self.frame], [f[k]], color="b", marker=".", s=1)
                    if idx == 1:
                        ax.plot(self.dfs[2][k][x_range[0]:x_range[1]], lw=0.4, label="%s %.2f" % (k, f2[k]), c="r")
                        ax.scatter([self.frame], [f2[k]], color="r", marker=".", s=1)
                    ax.legend(loc="upper right", fontsize="x-small")
                i += 1
                if i >= 2*PLOT_FEAT_ROWS:
                    break
            self.figs[idx].canvas.draw()

    stat_keys_simple = ["1:circle", "1:circle_s1", "1:sidewalk_s", "1:court_s_30", "0:overlap", "1:walk", #"0:dist_McFc", "0:court_as_male", "0:court_infer_male", "1:we_ipsi"
                        "1:court_s", "1:on_edge", "0:copulate", # "1:wing_l", "1:wing_r",
                        "1:dir", "1:d_dir", "1:v_len", "1:dist_ht", "0:court_as_male"]
    def plot_feat_simple(self):
        in_range = False
        if self.last_x_range is not None:
            if self.last_x_range[0] + 10 < self.frame < self.last_x_range[1] - 10:
                in_range = True
        if in_range:
            x_range = self.last_x_range
        else:
            half = PLOT_FEAT_LENGTH/2
            center = np.clip(self.frame, half, self.total_frame - half)
            x_range = (int(center - half), int(center + half))
        i = 0
        for k in ViewStat.stat_keys_simple:
            ax = self.axes[0][i]
            df = self.dfs[0][k]
            v = df[self.frame:self.frame + 5]
            ax.cla()
            ax.grid(True, linewidth=0.2)
            ax.plot(df[x_range[0]:x_range[1]], lw=0.6, label="%s %.2f" % (k, v[self.frame]))
            ax.scatter(range(self.frame, self.frame + 5), [v], color="b", marker=".", s=1)
            # if k[0] == "1":
            #     ax.plot(self.stat2[k][x_range[0]:x_range[1]], lw=0.1, label="%s %.2f" % (k, f2[k]), c="r")
            #     ax.scatter([self.frame], [f2[k]], color="r", marker=".", s=1)
            ax.legend(loc="upper left", fontsize="x-small")
            i += 1
            if i >= PLOT_FEAT_ROWS:
                break

        self.figs[0].canvas.draw()
        self.last_x_range = x_range

    def plot_one_stat(self, stat, keys, f, axes, x_range, color):
        i = 0
        for k in keys:
            # if feat[k].dtype in (int, float, np.int64, np.float64) and k not in ("frame", "time"):
            ax = axes[i]
            # ax.cla()
            leg, = ax.plot(stat[k][x_range[0]:x_range[1]], lw=0.3, label="%s %.2f" % (k, f[k]), c=color)
            # ax.legend([leg], ["%s %.2f" % (k, f[k])], loc="upper right")
            ax.scatter([self.frame], [f[k]], color=color, marker=".", s=1)
            i += 1
            if i >= PLOT_FEAT_ROWS:
                break

    def on_click_stat(self, event):
        if event.xdata:
            self.frame = int(event.xdata)
            self.update_plot_frame()
            self.plot_feat()

    def onkey(self, event):
        print(event.key)
        if event.key == "left":
            self.frame -= 1
        elif event.key == "right":
            self.frame += 1
        elif event.key == "up":
            self.frame = int(self.frame - self.cap_fps)
        elif event.key == "down":
            self.frame = int(self.frame + self.cap_fps)
        elif event.key == "a":
            self.frame = self.total_frame - 1
        elif event.key == "z":
            self.frame = 0
        elif event.key == "g":
            self.frame += 1
        elif event.key == "h":
            self.frame += 10
        elif event.key == "j":
            self.frame += 100
        elif event.key == "k":
            self.frame += 1000
        elif event.key == "l":
            self.frame += 10000
        elif event.key == "y":
            self.frame -= 10
        elif event.key == "u":
            self.frame -= 100
        elif event.key == "i":
            self.frame -= 1000
        elif event.key == "o":
            self.frame -= 10000
        elif event.key == "enter":
            self.frame = self.input_int
            self.input_int = 0
            self.plot_feat()
        elif event.key == "c":
            s = "#%s %s" % (self.frame, self.input_int)
            self.caption.append(s)
            print("caption:", s)
            self.input_int = 0
        elif event.key in list([*"1234567890"]):
            self.input_int = int(event.key) + self.input_int * 10
            print("input: %d" % self.input_int)
            return
        elif event.key == " ":
            self.plot_feat()
            return
        elif event.key == "+" or event.key == "-":
            if self.cur_play_frame >= 0:
                self.cur_play_frame = -1
                self.timer.cancel()
                self.frame = self.cur_play_frame_range[0 if event.key == "-" else 1]
                self.update_plot_frame()
            else:
                if event.key == "+":
                    self.cur_play_frame_range = self.get_next_cir_range()
                else:
                    self.cur_play_frame_range = self.get_last_cir_range()
                self.cur_play_frame = self.cur_play_frame_range[0]
                self.timer and self.timer.cancel()
                self.timer = Timer(0.1, self.play_frames)
                self.timer.start()
            return
        else:
            self.input_int = 0
            return
        if self.frame >= self.total_frame:
            self.frame = self.total_frame - 1
        if self.frame < 0:
            self.frame = 0
        self.update_plot_frame()

    def update_plot_frame(self):
        if self.in_update:
            print("update conflict!")
            return
        self.in_update = True
        self.plot_one_frame()
        self.slider.set_val(self.frame)
        self.fig.canvas.draw()
        self.in_update = False

    def play_frames(self):
        if self.cur_play_frame >= 0 and self.cur_play_frame <= self.cur_play_frame_range[1]:
            print("play_frames %d" % self.cur_play_frame)
            self.frame = self.cur_play_frame
            self.update_plot_frame()
            if self.cur_play_frame >= 0:
                self.timer.cancel()
                self.timer = Timer(0.1, self.play_frames)
                self.timer.start()
                self.cur_play_frame += 1
                return
        self.cur_play_frame = -1

    def get_next_cir_range(self):
        if self.frame >= self.total_frame - 1:
            return
        c = self.dfs[1]["1:circle" if PLOT_SIMPLE else "circle"]
        for frame in range(self.frame, self.total_frame):
            if c[frame]:
                s = frame
                break
        for frame in range(s, self.total_frame):
            if c[frame] < 1:
                e = frame
                break
        return s, e

    def get_last_cir_range(self):
        f = self.frame
        if f <= 1:
            return
        c = self.dfs[1]["1:circle" if PLOT_SIMPLE else "circle"]
        if c[self.frame] or c[self.frame - 1]:
            for frame in range(0, self.frame):
                if c[self.frame - frame] < 0:
                    f = self.frame - frame - 1
                    break
        for frame in range(0, f):
            if c[f - frame]:
                e = f - frame
                break
        for frame in range(0, e):
            if c[e - frame] < 1:
                s = e - frame
                break
        return s, e

    def onclose(self, val):
        global ex
        ex = True
        plt.close("all")

    def on_slider(self, val):
        self.frame = int(val)
        self.plot_one_frame()

    def show(self):
        self.plot_one_frame()
        self.plot_feat()
        plt.show()

vs = None
ex = False
vm = None
def main(path):
    global vs, vm, PLOT_SIMPLE
    if path.endswith("_stat0.pickle"):
        prefix = path[:path.rfind("_")]
        meta_path = prefix + "_meta.txt"
        if not os.path.exists(meta_path):
            pair = os.path.basename(prefix)
            p = pair.rfind("_")
            # meta_path = "G:/_video_Ort/%s/%s/%s_meta.txt" % (pair[:p], pair[p+1:], pair)
            meta_path = "E:/_video_hrnet_finish/%s/%s/%s_meta.txt" % (pair[:p], pair[p+1:], pair)
            # meta_path = "F:/temp/%s/%s/%s_meta.txt" % (pair[:p], pair[p+1:], pair)
            # meta_path = "G:/_video_part/%s/%s/%s_meta.txt" % (pair[:p], pair[p+1:], pair)
            print(meta_path)
        vs = ViewStat(path, meta_path)
        vm = vs.dfs[1]
        vs.show()
    else:
        vm = load_dataframe(path)
        print("len:", len(vm))
        print(vm.keys())
        PLOT_SIMPLE = False
        return

    # feat_file = sys.argv[1]
    # featc = load_dataframe(feat_file.replace("feat.csv", "featc.csv"))
    # stat0 = pd.concat([featc, load_dataframe(feat_file.replace("feat.csv", "stat0.csv"))], axis=1, sort=False)
    # pass

if __name__ == "__main__":
    if NEED_CONSOLE:
        main_tread = Thread(target=main, args=(sys.argv[1],))
        main_tread.start()
        print(">>100\n>> 1.1000.av\n>> vm[\"frame\"]")
        while not ex:
            cmd = input(">> ")
            try:
                if cmd[0].isdigit():
                    if cmd.isdigit():
                        print(vm.iloc[int(cmd)])
                    else:
                        cmd_t = cmd.split(".")
                        print(vs.dfs[int(cmd_t[0])].iloc[int(cmd_t[1])][cmd_t[2]])
                else:
                    print(eval(cmd))
            except Exception:
                print("trace")
    else:
        main(sys.argv[1])
