import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from matplotlib.widgets import Slider

from fpt_frame_stat import detect_interaction, INTERACT_V
from fpt_plot import plot_hist_by_info, DIST_TO_FLY_FAR, plot_figure
from fpt_util import pair_to_video_path, load_dict, load_dfs, save_dict, dfs_bouts, dfs_bouts_idx, calc_bouts

PLOT_FEAT_LENGTH = 66*10

# NOTE: cop > acp > ac > cir > court(l/r) > fab(l/r) > fm > engage
BEH_COLOR = {
    "cir_bouts1": "b",
    "copulate": "r",
    "engaged_bouts": "c",
    #"court_bouts": "darkblue", # NOTE: court=wl|wr
    "fmotion_bouts": "g",
    "ac_bouts": "k",
    "acp_bouts": "y",
    "wl_bouts": "purple",
    "wr_bouts": "orange",
    "fabl_bouts": "gold",
    "fabr_bouts": "m",
    "left": "m",
    "right": "gold",

    "Flicking1": "r", "Approach1": "g", "Turning1": "b", "Fencing1": "y", "Threat1": "c", "Lunging1": "m",
    "Flicking2": "r", "Approach2": "g", "Turning2": "b", "Fencing2": "y", "Threat2": "c", "Lunging2": "m",
}

def plot_interaction(cir_meta, dfs):
    # r_fab_alpha
    plt.figure("r_fab_alpha")
    ax = plt.gca()
    plot_figure(ax, dfs, 1, "r_alpha_fab", cir_meta=cir_meta)
    # ac_trigger

class ViewEtho(object):
    def __init__(self, cir_meta_file):
        self.frame = 0
        self.input_int = 0
        self.left_bouts = None
        self.right_bouts = None

        self.meta = load_dict(cir_meta_file)
        self.fps = self.meta["FPS"]
        parent = os.path.dirname(cir_meta_file)
        self.pair = os.path.basename(parent)
        video_file = os.path.join(os.path.dirname(parent), self.meta["file"])
        if not os.path.exists(video_file):
            video_file, video_name, pair_no, _ = pair_to_video_path(self.pair)
            self.meta_o = load_dict(os.path.join(os.path.dirname(video_file), pair_no, self.pair + "_meta.txt"))
        else:
            self.meta_o = load_dict(cir_meta_file.replace("_cir_meta.txt", "_meta.txt"))
        self.dfs = load_dfs(parent)
        if self.meta.get("interact_v") != INTERACT_V and not self.meta.get("Approach1"):
            detect_interaction(self.meta, self.dfs)
            save_dict(cir_meta_file, self.meta)

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap:
            print("open video failed! %s" % video_file)
        self.total_frame = len(self.dfs[0])#self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap_size = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.feat_scale = self.meta["FEAT_SCALE"]
        self.roi = np.array(self.meta_o["ROI"]["roi"]).astype(int)
        self.roi_size = (self.roi[1][0] - self.roi[0][0], self.roi[1][1] - self.roi[0][1])
        self.roi_size_scale = (self.roi_size[0]/ self.feat_scale, self.roi_size[1] / self.feat_scale)

        self.cap_fig, self.cap_ax = plt.subplots(figsize=(6, 5), num="video")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.23)
        self.slider_ax = plt.axes([0.1, 0.11, 0.78, 0.03])
        self.slider = Slider(self.slider_ax, "", valmin=0, valmax=self.total_frame - 1, valfmt="%d", valinit=0)
        self.etho_all_ax = plt.axes([0.1, 0.05, 0.78, 0.06])
        self.slider.on_changed(self.on_slider)
        self.cap_fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.cap_fig.canvas.mpl_connect("close_event", self.onclose)
        self.set_window_pos(0, 400)

        self.etho_fig, self.etho_ax = plt.subplots(figsize=(12, 2), num="etho")
        plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.15)
        self.etho_fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.etho_fig.canvas.mpl_connect('button_press_event', self.onclick_etho)
        self.etho_fig.canvas.mpl_connect("close_event", self.onclose)
        self.set_window_pos(0, 1000)

        self.plot_etho(self.etho_all_ax, True)
        self.etho_all_ax.set_yticklabels([])

        # plot_interaction(self.meta, self.dfs)
        # self.save_etho(os.path.join(parent, "etho.png"))

    def set_window_pos(self, x, y):
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+%d+%d" % (x, y))
        
    def plot_body_box(self, x, y, d, h_maj, h_min, c):
        print(x, y, d, h_maj, h_min)
        t = np.deg2rad(d)
        dx1, dy1 = h_maj*np.cos(t), h_maj*np.sin(t)
        dx2, dy2 = -h_min*np.sin(t), h_min*np.cos(t)
        self.cap_ax.plot([x+dx1, x+dx2, x-dx1, x-dx2, x+dx1], [y+dy1, y+dy2, y-dy1, y-dy2, y+dy1], linewidth=0.5, c=c)

    def plot_one_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
        ret, img = self.cap.read()
        if ret:
            img_gray = img[:, :, 1][self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]]
            self.cap_ax.cla()
            self.cap_ax.imshow(img_gray, cmap=plt.cm.gray, norm=NoNorm(), extent=(0, self.roi_size_scale[0], 0, self.roi_size_scale[1]))
            track = None, self.dfs[1].iloc[self.frame], self.dfs[2].iloc[self.frame]
            for i in (1, 2):
                c = "b" if i == 1 else "r"
                self.cap_ax.scatter([track[i]["pos:x"]], [track[i]["pos:y"]], c=c, marker="o")
                xs, ys = track[i]["point:xs"], track[i]["point:ys"]
                self.cap_ax.scatter(xs, ys, s=20, c="ywmkg", marker="o" if i == 0 else "x")

                self.cap_ax.plot(xs[:3], ys[:3], linewidth=0.5, c="w")
                self.cap_ax.plot(xs[1:4:2], ys[1:4:2], linewidth=0.5, c="w")
                self.cap_ax.plot(xs[1:5:3], ys[1:5:3], linewidth=0.5, c="w")
                self.plot_body_box(track[i]["pos:x"], track[i]["pos:y"], track[i]["dir"], track[i]["e_maj"]/2, track[i]["e_min"]/2, c)
        t_sec = self.frame / self.cap_fps
        self.slider_ax.set_title("%02d:%02.2f" % (t_sec / 60, t_sec % 60))
        print("#%d" % self.frame)
        self.cap_fig.canvas.draw()
        self.plot_etho(self.etho_ax)
        self.etho_fig.canvas.draw()

    def save_etho(self, path):
        plt.figure(figsize=(100, 2))
        ax = plt.gca()
        self.plot_etho(ax, True)
        ticks = np.arange(0, 1801, 10)
        plt.xticks(ticks, ticks)
        # ax.set_xticks()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def plot_etho(self, ax, is_all=False):
        half = PLOT_FEAT_LENGTH/2
        center = np.clip(self.frame, half, self.total_frame - half)
        self.x_range = int(center - half), int(center + half)
        # self.etho_ax[0].cla()
        # self.etho_ax[0].plot(self.dfs[1]["v_len"][x_range[0]:x_range[1]], lw=0.3, c="b")
        # self.etho_ax[0].plot(self.dfs[2]["v_len"][x_range[0]:x_range[1]], lw=0.3, c="r")
        # self.etho_ax[0].set_ylabel("v", rotation=0)

        if not self.left_bouts:
            self.left_bouts = calc_bouts(self.dfs[2]["rel_pos:x"] < -0.5)
            self.right_bouts = calc_bouts(self.dfs[2]["rel_pos:x"] > 0.5)

        ax.cla()
        idx = 0

        if self.meta["Approach1"]:
            for fly in ("1", "2"):
                for beh in ["Flicking", "Approach", "Turning", "Fencing", "Threat", "Lunging"]:
                    self.plot_etho_bouts(ax, idx, beh + fly, is_all)
                idx += 1

            ax.set_yticks(np.arange(0.5, idx + 1, 1))
            ax.set_yticklabels(["fly1", "fly2"])
        else:
            self.plot_etho_bouts(ax, idx, "fmotion_bouts", is_all)
            idx += 1
            self.plot_etho_bouts(ax, idx, "fabl_bouts", is_all)
            self.plot_etho_bouts(ax, idx, "fabr_bouts", is_all)
            idx += 1
            self.plot_etho_bouts(ax, idx, "left", is_all, self.left_bouts)
            self.plot_etho_bouts(ax, idx, "right", is_all, self.right_bouts)
            idx += 1
            self.plot_etho_bouts(ax, idx, "wl_bouts", is_all)
            self.plot_etho_bouts(ax, idx, "wr_bouts", is_all)
            idx += 1
            self.plot_etho_bouts(ax, idx, "engaged_bouts", is_all)
            self.plot_etho_bouts(ax, idx, "ac_bouts", is_all)
            ret = self.plot_etho_bouts(ax, idx, "acp_bouts", is_all)
            if is_all:
                self.acp_bar = ret
            idx += 1
            self.plot_etho_bouts(ax, idx, "copulate", is_all)
            self.plot_etho_bouts(ax, idx, "cir_bouts1", is_all)
            idx += 1

            ax.set_yticks(np.arange(0.5, idx + 1, 1))
            ax.set_yticklabels(["fmotion", "fab", "left", "wing", "engage", "cir"])

        cur_t = self.frame/self.fps
        ax.plot([cur_t, cur_t], [0, idx], "r")
        ax.set_ylim([0, idx])
        if is_all:
            ax.set_xlim([0, self.total_frame / self.fps])
        else:
            ax.set_xlim([self.x_range[0] / self.fps, self.x_range[1] / self.fps])

    def plot_etho_bouts(self, ax, idx, beh, is_all, bouts=None):
        if not bouts:
            bouts = self.meta[beh]
        if is_all:
            bar_x = [(b[0]/self.fps, (b[1] - b[0])/self.fps) for b in bouts]
        else:
            bar_x = [(b[0] / self.fps, (b[1] - b[0]) / self.fps) for b in bouts if
                     (self.x_range[0] <= b[1] <= self.x_range[1] or self.x_range[0] <= b[0] <= self.x_range[1]) or
                     (b[0] <= self.x_range[0] and b[1] >= self.x_range[1])]
        ax.broken_barh(bar_x, (idx, 1), facecolors=BEH_COLOR.get(beh, "b"))
        return bar_x

    def onclick_etho(self, event):
        if event.xdata:
            self.frame = int(event.xdata * self.fps)
            self.slider.set_val(self.frame)

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
        elif event.key == "n":
            cur_t = self.frame / self.fps
            for acp in self.acp_bar:
                if acp[0] > cur_t:
                    self.frame = int(acp[0] * self.fps)
                    break
        elif event.key in list([*"1234567890"]):
            self.input_int = int(event.key) + self.input_int * 10
            print("input: %d" % self.input_int)
            return
        else:
            self.input_int = 0
            return
        if self.frame >= self.total_frame:
            self.frame = self.total_frame - 1
        if self.frame < 0:
            self.frame = 0
        self.slider.set_val(self.frame)

    def on_slider(self, val):
        self.frame = int(val)
        self.plot_one_frame()

    def show(self):
        self.plot_one_frame()
        plt.show()

    def onclose(self, val):
        plt.close("all")


if __name__ == "__main__":
    # "D:\exp\data6\geno_data\CS22\20200425_133052_A_0\20200425_133052_A_0_cir_meta.txt"
    # "D:\exp\data7\geno_data\CS22\20200602_144311_A_11\20200602_144311_A_11_cir_meta.txt"
    ViewEtho(sys.argv[1]).show()


