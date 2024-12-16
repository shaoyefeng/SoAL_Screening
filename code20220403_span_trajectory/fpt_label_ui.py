# -*- coding: utf-8 -*-
import cv2
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from fpt2_region_seg import RegionSeg
from fpt_util import load_dict, save_dict

MAX_FRAME = 400
LABEL_STEP = 66*3#90
FLY = 0
DONT_FLIP = False

# NOTE: predict result is identical with .feat, but not .pickle (head_tail_close corrected)
class LabelUI(object):
    def __init__(self, name, figsize, video, roi, roi_i, bg_filename, root, fly=FLY, frame_per_label=LABEL_STEP, point_d=None, real_d=None, head_d=None):
        self.OUTPUT_IMG = name == "head" and not real_d
        self.name = name
        self._video = video
        self.fly = fly
        self.clicked = False
        self.cap = cv2.VideoCapture(self._video)
        self.img_bg = cv2.imread(bg_filename, cv2.IMREAD_GRAYSCALE)
        self.fig, self.axes = plt.subplots(1, 2, num=name, figsize=figsize)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_roi)

        self.roi = roi
        self.roi_i = roi_i
        self.frame_per_label = frame_per_label

        self.rs = RegionSeg(video, 0, None)
        self.rs.init_bg()
        self.frames = [self.frame_per_label, 0]
        self.imgs = [self.read_frame(self.cap, f) for f in self.frames]
        self.root = os.path.join(root, "label")
        not os.path.exists(self.root) and os.mkdir(self.root)
        self.point_d = point_d or {}
        self.real_d = real_d or {}
        self.head_d = head_d or {}
        self.flip_d = {}
        self.refresh_fig()

    def show(self):
        plt.show()

    def on_close(self, event):
        if self.clicked:
            log = open(os.path.join(self.root, "%s.csv"%self.name), "w")
            log.write("Slice,Frame,Area,Max,Mean,X,Y\n")
            log.writelines(["%d,%d,0,0,0,%f,%f\n" % (int(k/self.frame_per_label), k, v[0], v[1]) for k, v in self.point_d.items()])
            log.close()

    def on_click(self, event):
        if event.xdata and event.ydata:
            p = [event.xdata, event.ydata]
            if event.inaxes == self.axes[0]:
                self.clicked = True
                is_flip = self.get_flip(self.imgs[0], self.frames[0])
                p0 = self.imgs[0].shape[1] - p[0] - 1 if is_flip else p[0]
                p1 = self.imgs[0].shape[0] - p[1] - 1 if is_flip else p[1]
                self.point_d[self.frames[0]] = p0, p1
                for i in range(2):
                    self.frames[i] += self.frame_per_label
                self.imgs = [self.read_frame(self.cap, f) for f in self.frames]
            elif event.inaxes == self.axes[1]:
                self.clicked = True
                is_flip = self.get_flip(self.imgs[1], self.frames[1])
                p0 = self.imgs[1].shape[1] - p[0] - 1 if is_flip else p[0]
                p1 = self.imgs[1].shape[0] - p[1] - 1 if is_flip else p[1]
                self.point_d[self.frames[1]] = p0, p1
            self.refresh_fig()
            if self.OUTPUT_IMG:
                cv2.imwrite(os.path.join(self.root, "img%d.jpg"%self.frames[1]), self.imgs[1])

    def on_key_roi(self, event):
        print(event.key)
        if event.key == "left":
            for i in range(2):
                self.frames[i] -= self.frame_per_label
        elif event.key == "right":
            for i in range(2):
                self.frames[i] += self.frame_per_label
        elif event.key == "up":
            for i in range(2):
                self.frames[i] -= self.frame_per_label*30
        elif event.key == "down":
            for i in range(2):
                self.frames[i] += self.frame_per_label*30
        elif event.key == "end":
            for i in range(2):
                self.frames[i] = self.frame_per_label * len(self.point_d) - i
        elif event.key == " ":
            ypk = pred_one_img(self.imgs[0])
            self.axes[0].scatter(ypk[0,:,0], ypk[1,:,0], color="rgbyc")
            plt.draw()
            return

        self.imgs = [self.read_frame(self.cap, f) for f in self.frames]
        self.refresh_fig()
        # if event.key == "right":
        #     cv2.imwrite(os.path.join(self.root, "img%d.jpg"%self.frames[1]), self.imgs[1])

    def refresh_fig(self):
        for i in range(2):
            ax = self.axes[i]
            ax.cla()
            if self.imgs[i] is not None:
                max_x = self.imgs[i].shape[1] - 1
                max_y = self.imgs[i].shape[0] - 1
                is_flip = self.get_flip(self.imgs[i], self.frames[i])
                img = self.imgs[i].astype(np.uint8)
                if is_flip:
                    img = np.flip(img)
                ax.imshow(img, cmap=plt.cm.gray, norm=NoNorm())
                p = self.point_d.get(self.frames[i])
                if p:
                    ax.scatter([max_x - p[0] if is_flip else p[0]], [max_y - p[1] if is_flip else p[1]], c="b", marker="+")
                p = self.real_d.get(self.frames[i])
                if p:
                    ax.scatter([max_x - p[0] if is_flip else p[0]], [max_y - p[1] if is_flip else p[1]], c="r", marker="x")
                p = self.head_d.get(self.frames[i])
                if p:
                    ax.scatter([max_x - p[0] if is_flip else p[0]], [max_y - p[1] if is_flip else p[1]], c="g", marker="o")
                ax.set_title("frame:%d (%d) %s" % (self.frames[i], self.frames[i] / self.frame_per_label, "flip" if is_flip else ""))
        plt.draw()

    def get_flip(self, img, frame):
        if DONT_FLIP:
            return False
        ret = self.flip_d.get(frame)
        if ret is not None:
            return ret
        ph = self.head_d.get(frame)
        if ph is not None:
            return ph[1] > img.shape[0] / 2
        return need_flip(img)

    def read_frame(self, cap, seq):
        if seq <= 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, seq)
        ret, img = cap.read()
        if not ret:
            return None
        return self.rs.proc_one_img(img[:, :, 1], self.roi, self.roi_i)[self.fly]

def need_flip(img_center):
    sub_bin = (img_center < GRAY_THRESHOLD + 40).astype(np.uint8)
    # sub_bin = img_center
    y_center = int(sub_bin.shape[0]/2)
    y_u = int(sub_bin.shape[0]/3)
    y_d = sub_bin.shape[0] - y_u
    top = sub_bin[y_u:y_center, :]
    bottom = sub_bin[y_center:y_d, :]
    if np.count_nonzero(top) > np.count_nonzero(bottom):
    # if np.sum(top) < np.sum(bottom):
        return True
    return False

def pred_imgs(parent, frames):
    img_l = [cv2.imread(parent + "/img%d.jpg" % f)[:,:,1] for f in frames]
    from nn_dlc import predict_box
    from fpt_consts import MODEL_FOLDER
    model = predict_box.Model(MODEL_FOLDER)
    ypk, _ = model.pred_img_batch(img_l)
    return ypk


def RMSE(parent, force=False):
    ypk = None
    rmse_d = {}
    for i, part in enumerate(["head", "center", "tail", "wingl", "wingr"]):
        label_f = os.path.join(parent, part + ".csv")
        if os.path.exists(label_f):
            label = pd.read_csv(label_f, nrows=MAX_FRAME)
            frames = label["Frame"]
            if force or "RX" not in label.keys():
                if ypk is None:
                    ypk = pred_imgs(parent, frames)
                point = ypk[:, i, :]
                truth_x, truth_y = point[0], point[1]
                label["RX"] = truth_x
                label["RY"] = truth_y

            label_x, label_y = np.array(label["X"].tolist()), np.array(label["Y"].tolist())
            d = np.sqrt((label_x - label["RX"])**2 + (label_y - label["RY"])**2)
            label.to_csv(label_f, index=False)
            rmse_d[part] = d.mean()
            print("%.4f" % rmse_d[part])

    # save_dict(os.path.join(parent, "RMSE.txt"), rmse_d)

def direction_error(parent):
    part_d = {}
    for part in ["head", "tail"]:
        label_f = os.path.join(parent, part + ".csv")
        if os.path.exists(label_f):
            part_d[part] = pd.read_csv(label_f)
    body_dir_v = part_d["head"]["X"] - part_d["tail"]["X"], part_d["head"]["Y"] - part_d["tail"]["Y"]
    body_dir = np.rad2deg(np.arctan2(body_dir_v[1], body_dir_v[0]))
    body_dir_v = part_d["head"]["RX"] - part_d["tail"]["RX"], part_d["head"]["RY"] - part_d["tail"]["RY"]
    body_dir_r = np.rad2deg(np.arctan2(body_dir_v[1], body_dir_v[0]))
    err = np.abs(body_dir - body_dir_r)
    err_frame = part_d["head"][err > 90]["Frame"].tolist()
    # plt.hist(err, bins=180)
    # plt.show()
    ret = np.mean(err)
    print("%.4f" % ret)
    f = open(os.path.join(parent, "dir_err.txt"), "w")
    f.write("%.4f %s\n" % (ret, err_frame))
    f.close()

PART = "head"
GRAY_THRESHOLD = 0

def main(path):
    global GRAY_THRESHOLD
    parent = os.path.dirname(path)
    label_dir = os.path.join(parent, "label")
    cmd = "r"#"head"
    if len(sys.argv) > 2:
        cmd = sys.argv[2]
    if cmd == "r":
        print(os.path.basename(path))
        RMSE(label_dir)
        direction_error(label_dir)
    else:
        PART = cmd
        meta = load_dict(path.replace("_feat.csv", "_meta.txt"))
        GRAY_THRESHOLD = meta.get("GRAY_THRESHOLD")
        video_file = os.path.join(os.path.dirname(parent), meta["file"])
        roi = meta["ROI"]["roi"]
        roi_i = meta["ROI"]["idx"]
        point_d, real_d, head_d = {}, {}, {}

        if PART != "head":
            part_path = os.path.join(parent, "label", "head.csv")
            if os.path.exists(part_path):
                part_csv = pd.read_csv(part_path)
                head_d = {r["Frame"]: (r["X"], r["Y"]) for i, r in part_csv.iterrows()}

        part_path = os.path.join(parent, "label", PART + ".csv")
        if os.path.exists(part_path):
            part_csv = pd.read_csv(part_path)
            point_d = {r["Frame"]: (r["X"], r["Y"]) for i, r in part_csv.iterrows()}
            if "RX" in part_csv.keys():
                real_d = {r["Frame"]: (r["RX"], r["RY"]) for i, r in part_csv.iterrows()}
        ui = LabelUI(PART, (6, 3), video_file, roi, roi_i, video_file.replace(".avi", ".bmp"), parent, point_d=point_d,
                     real_d=real_d, head_d=head_d)
        ui.show()

if __name__ == '__main__':
    for path in [r"D:\exp\video_test\test_trk\20190302_103111_1\2\20190302_103111_1_2_meta.txt",
                r"D:\exp\video_test\test_trk\20190605_145501_2\3\20190605_145501_2_3_meta.txt",
                r"D:\exp\video_test\test_trk\20190916_162500_2\0\20190916_162500_2_0_meta.txt",
                r"D:\exp\video_test\test_trk\20200424_163847_2\2\20200424_163847_2_2_meta.txt",
                r"D:\exp\video_test\test_trk\20200602_144311_2\11\20200602_144311_2_11_meta.txt",
                r"D:\exp\video_test\test_trk\20190618_144050_1\2\20190618_144050_1_2_meta.txt",
                r"D:\exp\video_test\test_trk\20200707_165506_1\13\20200707_165506_1_13_meta.txt"]:
        main(path)
    # path = sys.argv[1]
    # main(path)
