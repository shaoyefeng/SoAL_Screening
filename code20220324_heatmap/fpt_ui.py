# -*- coding: utf-8 -*-
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from matplotlib.widgets import Button, TextBox, Slider, CheckButtons, RadioButtons

from fpt_consts import *
from fpt_bg import calc_bg, sub_img, remove_bg2


def read_frame(cap, seq):
    cap.set(cv2.CAP_PROP_POS_FRAMES, seq)
    ret, img = cap.read()
    if not ret:
        return None
    if FIX_VIDEO_SIZE:
        return cv2.resize(img[:, :, 1], FIX_VIDEO_SIZE)  # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img[:, :, 1]

def point_dist(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def replace_text_00(t):
    return t.replace("\x00", "")

def replace_text_box_00(t):
    t.set_val(t.text.replace("\x00", ""))

def make_text_box(coord, label="", initial="", on_submit=None):
    txt_box = TextBox(plt.axes(coord), label, initial=initial)
    on_submit and txt_box.on_submit(on_submit)
    return txt_box


class InputExpInfoUI(object):

    def __init__(self, name, figsize, cap, info):
        self.two_point_len = DEFAULT_TWO_POINT_LEN
        self.point1, self.point2 = None, None
        self.img = None
        self.FEAT_SCALE = 0
        self.AREA_RANGE = 0

        self.fig, self.ax = plt.subplots(num=name, figsize=figsize)
        plt.subplots_adjust(top=0.95, bottom=0.3)

        self.cap = cap
        self.exp_info = info
        self.total_frame = info["total_frame"]
        self.width = info["width"]
        self.height = info["height"]

        # all widgets must have a variable
        self.txt_exp_date = make_text_box([0.25, 0.2, 0.1, 0.05], "exp_date", info.get("exp_date", day2str(datetime.now())), lambda e: None)
        self.txt_female_date = make_text_box([0.5, 0.2, 0.1, 0.05], "female_date", info.get("female_date", day2str(datetime.now() + timedelta(days=-6))))
        self.txt_temperature = make_text_box([0.75, 0.2, 0.1, 0.05], "temperature", str(info.get("temperature", 31)))
        self.txt_box = make_text_box([0.75, 0.12, 0.1, 0.05], "two_point_len(mm)", str(self.two_point_len), self.on_txt_len)
        self.txt_feat_scale = make_text_box([0.25, 0.12, 0.1, 0.05], "FEAT_SCALE")
        self.txt_area_range = make_text_box([0.5, 0.12, 0.1, 0.05], "AREA_RANGE")

        self.btn_random = Button(plt.axes([0.1, 0.04, 0.1, 0.05]), "random")
        self.btn_random.on_clicked(self.on_click_random)
        self.btn_confirm = Button(plt.axes([0.8, 0.04, 0.1, 0.05]), "confirm")
        self.btn_confirm.on_clicked(self.on_click_btn_confirm)
        self.fig.canvas.mpl_connect('button_press_event', self.on_ax_press)

        self.show_random_frame()
        if info.get("FEAT_SCALE"):
            self.FEAT_SCALE = info.get("FEAT_SCALE")
            center = self.width / 2, self.height / 2
            self.point1 = [center[0] - self.two_point_len * self.FEAT_SCALE, center[1]]
            self.point2 = [center[0], center[1]]
        else:
            self.detect_circle(int(self.height / (ARENA_ROWS*2)), int(self.height / (ARENA_ROWS*4)), int(self.height / (ARENA_ROWS*2)))
        self.refresh_fig()

    def show(self):
        plt.show()
        
    def detect_circle(self, min_dist, min_radius, max_radius):
        circles = cv2.HoughCircles(self.img.astype(np.uint8), cv2.HOUGH_GRADIENT, 1, min_dist, #cv2.blur(self.img, (5, 5))
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
        if circles is not None:
            center = self.width/2, self.height/2
            circles = list(circles[0])
            circles.sort(key=lambda t: abs(t[0] - center[0]) + abs(t[1] - center[1]))  # sort by dist to center
            mid_cir = circles[0]
            self.point1 = [mid_cir[0] - mid_cir[2], mid_cir[1]]
            self.point2 = [mid_cir[0] + mid_cir[2], mid_cir[1]]

    def show_random_frame(self):
        frame = np.random.randint(0, self.total_frame)
        self.img = read_frame(self.cap, frame)
        self.ax.set_title("frame:" + str(frame))

    def on_txt_len(self, text):
        self.two_point_len = float(replace_text_00(text))
        self.refresh_fig()

    def on_click_random(self, event):
        self.ax.cla()
        self.show_random_frame()
        self.refresh_fig()

    def on_click_btn_confirm(self, event):
        if self.txt_female_date.text.endswith("X"):
            print("input female date!")
            return
        # self.refresh_fig()
        replace_text_box_00(self.txt_exp_date)
        replace_text_box_00(self.txt_female_date)
        replace_text_box_00(self.txt_temperature)
        replace_text_box_00(self.txt_box)

        self.exp_info["FEAT_SCALE"] = self.FEAT_SCALE
        self.exp_info["AREA_RANGE"] = self.AREA_RANGE
        self.exp_info["temperature"] = self.txt_temperature.text
        self.exp_info["exp_date"] = self.txt_exp_date.text
        self.exp_info["female_date"] = self.txt_female_date.text
        self.exp_info["female_days"] = int(self.txt_female_date.text) if len(self.txt_female_date.text) < 3 else day_str_diff(self.txt_exp_date.text, self.txt_female_date.text)

        print(self.exp_info)
        plt.close()

    def on_ax_press(self, event):
        if event.inaxes == self.ax and event.xdata and event.ydata:
            if self.point1 and self.point2:
                self.point1 = [event.xdata, event.ydata]
                self.point2 = None
            elif self.point1:
                self.point2 = [event.xdata, event.ydata]
            else:
                self.point1 = [event.xdata, event.ydata]
            self.refresh_fig()

    def refresh_fig(self):
        self.ax.cla()
        if self.point1 and self.point2:
            length = point_dist(self.point1, self.point2)
            self.FEAT_SCALE = round(length / self.two_point_len, 2) if self.two_point_len else 0
            self.ax.plot([self.point1[0], self.point2[0]], [self.point1[1], self.point2[1]], "b-o" if self.FEAT_SCALE else "r-o")
        else:
            self.point1 and self.ax.plot([self.point1[0]], [self.point1[1]], "r-o")
            self.FEAT_SCALE = 0
        self.ax.imshow(self.img, cmap=plt.cm.gray, norm=NoNorm())

        self.AREA_RANGE = [int(self.FEAT_SCALE * 3), int(self.FEAT_SCALE * 100)]
        self.txt_feat_scale.set_val(str(self.FEAT_SCALE))
        self.txt_area_range.set_val(str(self.AREA_RANGE))
        plt.draw()


class InputRoiInfoUI(object):
    def __init__(self, name, figsize, cap, info):
        self.rois = []
        self.roi_point1, self.roi_point2 = None, None
        self.circles = []
        self.fly_info_l = []

        self.fig, self.ax = plt.subplots(num=name, figsize=figsize)
        plt.subplots_adjust(top=0.95, bottom=0.3)

        self.cap = cap
        self.roi_info = info
        self.total_frame = info["total_frame"]
        self.width = info["width"]
        self.height = info["height"]
        self.FEAT_SCALE = info["FEAT_SCALE"]
        self.arena_min_dist = int(self.FEAT_SCALE * 10)
        self.arena_radius_min = int(self.FEAT_SCALE * 5)
        self.arena_radius_max = int(self.FEAT_SCALE * 15)
        self.arena_extend = self.FEAT_SCALE / 3
        self.info_to_male_geno = None
        ROI = info.get("ROI")
        if ROI:
            for f in ROI:
                roi = f["roi"]
                self.rois.append(roi)
                x = (roi[1][0] + roi[0][0]) / 2
                y = (roi[1][1] + roi[0][1]) / 2
                r = (roi[1][0] - roi[0][0]) / 2 - self.arena_extend
                self.circles.append([x, y, r, 1])
            # self.fly_info_l = ["_" + f.get("male_geno", "") + "_" + str(f.get("male_days", 0)) for f in ROI]
            # for f in ROI:
            #     if not f.get("info"):
            #         f["info"] = "_Z_0"
            #         f["male_geno"] = "CS"
            #         f["male_date"] = info["exp_date"]
            #         f["male_days"] = "0"
            self.fly_info_l = [f["info"] for f in ROI]
            self.info_to_male_geno = {}
            for f in ROI:
                self.info_to_male_geno[f["info"]] = [f["male_geno"], f["male_date"], f["male_days"]]

        circle_param = "%d,%d,%d,%d" % (self.arena_radius_min, self.arena_radius_max, self.arena_min_dist, self.arena_extend)
        self.txt_box = make_text_box([0.75, 0.12, 0.15, 0.05], "circle_param", circle_param, self.on_submit_shape)
        self.txt_box_fly = make_text_box([0.1, 0.12, 0.5, 0.05], "fly_info", ",".join(self.fly_info_l), self.on_submit_fly_info)
        self.shape_check = CheckButtons(plt.axes([0.1, 0.2, 0.15, 0.05]), ["is_round_roi"], [True])
        self.shape_check.on_clicked(self.on_click_shape)

        self.btn_random = Button(plt.axes([0.1, 0.04, 0.1, 0.05]), "random")
        self.btn_random.on_clicked(self.on_click_random)
        self.btn_clear = Button(plt.axes([0.22, 0.04, 0.1, 0.05]), "clear")
        self.btn_clear.on_clicked(self.on_click_clear)
        self.btn_confirm = Button(plt.axes([0.8, 0.04, 0.1, 0.05]), "confirm")
        self.btn_confirm.on_clicked(self.on_click_btn_confirm)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_roi)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_roi)

        self.show_random_frame()
        self.refresh_fig()
        
    def show(self):
        plt.show()

    def show_random_frame(self):
        frame = np.random.randint(0, self.total_frame)
        self.img = read_frame(self.cap, frame)
        self.ax.set_title("frame:%d" + str(frame))

    def on_click_clear(self, event):
        self.rois = []
        self.roi_point1, self.roi_point2 = None, None
        if self.shape_check.get_status()[0]:
            self.shape_check.set_active(1)
        self.circles = []
        self.refresh_fig()
    
    def on_click_shape(self, name):
        self.refresh_fig()
    
    def on_submit_shape(self, text):
        text = replace_text_00(text)
        try:
            s = text.split(",")
            self.arena_radius_min, self.arena_radius_max, self.arena_min_dist, self.arena_extend = int(s[0]), int(s[1]), int(s[2]), int(s[3])
        except Exception:
            print("input can't split!")
            return
        self.circles = []
        self.refresh_fig()

    def on_submit_fly_info(self, text):
        text = replace_text_00(text)
        try:
            s = text.split(",")
        except Exception:
            print("input can't split!")
            return
        self.fly_info_l = []
        for si in s:
            if si.find("*") > 0:
                a = si.split("*")
                self.fly_info_l.extend([a[0]] * int(a[1]))
            else:
                self.fly_info_l.append(si)
        self.refresh_fig()

    def refresh_fig(self):
        self.ax.cla()
        self.ax.imshow(self.img, cmap=plt.cm.gray, norm=NoNorm())
        if not self.shape_check.get_status()[0]:  # rect
            for p in self.rois:
                self.plot_rect(self.ax, p[0], p[1], "b-o")
            if self.roi_point1 and self.roi_point2:
                self.plot_rect(self.ax, self.roi_point1, self.roi_point2, "r-o")
            else:
                self.roi_point1 and self.ax.plot([self.roi_point1[0]], [self.roi_point1[1]], "r-o")
        else:  # round
            self.rois = []
            if not self.circles:
                circles = cv2.HoughCircles(self.img, cv2.HOUGH_GRADIENT, 1, self.arena_min_dist,
                                           param1=50, param2=30, minRadius=self.arena_radius_min, maxRadius=self.arena_radius_max)
                if circles is not None:
                    for i in circles[0, :]:
                        x, y, r = i[0], i[1], i[2]
                        re = r + self.arena_extend
    
                        self.rois.append(self.get_roi(x, y, re))
                        self.circles.append([x, y, r, 1])
            else:
                for i in self.circles:
                    x, y, r, is_roi = i[0], i[1], i[2], i[3]
                    if is_roi:
                        re = r + self.arena_extend
                        self.rois.append(self.get_roi(x, y, re))
            self.draw_circles()
    
            rois_a = []
            while self.rois:
                min_y = min([rr[0][1] for rr in self.rois])
                row = []
                rois_r = []
                for roi in self.rois:
                    if abs(roi[0][1] - min_y) < 100:
                        row.append(roi)
                    else:
                        rois_r.append(roi)
                row.sort(key=lambda _r: _r[0])
                rois_a.extend(row)
                self.rois = rois_r
            self.rois = rois_a
            self.draw_fly_info()
        print(self.rois)
        plt.draw()

    def draw_fly_info(self):
        if self.fly_info_l:
            for idx, roi in enumerate(self.rois):
                fly_info = self.fly_info_l[idx] if len(self.fly_info_l) > idx else ""
                male_geno, male_date, male_days = self.get_male_geno_day(fly_info)
                self.ax.text((roi[0][0]+roi[1][0])/2, (roi[0][1]+roi[1][1])/2, male_geno + "+" + str(male_days))

    def get_roi(self, x, y, e):
        x1 = max(0, int(x-e))
        y1 = max(0, int(y-e))
        x2 = min(self.width-1, int(x+e))
        y2 = min(self.height-1, int(y+e))
        return [x1, y1], [x2, y2]
    
    def draw_circles(self):
        for i in self.circles:
            x, y, r, is_roi = i[0], i[1], i[2] + self.arena_extend, i[3]
            color = "b" if is_roi else "r"
            circle = plt.Circle((x, y), r, fill=False, color=color)
            self.ax.add_patch(circle)
            circle = plt.Circle((x, y), 2, color=color)
            self.ax.add_patch(circle)
    
    def on_key_roi(self, event):
        if self.shape_check.get_status()[0]:
            return
        if event.key == "enter":
            if self.roi_point1 and self.roi_point2:
                minx, maxx = min(self.roi_point1[0], self.roi_point2[0]), max(self.roi_point1[0], self.roi_point2[0])
                miny, maxy = min(self.roi_point1[1], self.roi_point2[1]), max(self.roi_point1[1], self.roi_point2[1])
                self.rois.append(([minx, miny], [maxx, maxy]))
                self.roi_point1, self.roi_point2 = None, None
                self.refresh_fig()
        elif event.key == "c":
            if self.roi_point1 and self.roi_point2:
                self.roi_point2 = None
                self.refresh_fig()
        elif event.key == "r":
            self.show_random_frame()
            self.rois = []
            self.roi_point1, self.roi_point2 = None, None
            self.refresh_fig()

    def on_click_roi(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata and event.ydata:
            p = [int(event.xdata), int(event.ydata)]
            if not self.shape_check.get_status()[0]:
                if self.roi_point1 and self.roi_point2:
                    self.roi_point1 = p
                    self.roi_point2 = None
                elif self.roi_point1:
                    self.roi_point2 = p
                else:
                    self.roi_point1 = p
            else:
                c = None
                min_dist = 1000
                for i in self.circles:
                    d = point_dist(i, p)
                    if d < min_dist:
                        min_dist = d
                        c = i
                if c:
                    c[3] = 1 - c[3]
                    print("click %s %s" % self.get_roi(c[0], c[1], c[2] + self.arena_extend))
    
            self.refresh_fig()

    def plot_rect(self, ax, p1, p2, c):
        ax.plot([p1[0], p2[0], p2[0], p1[0], p1[0]], [p1[1], p1[1], p2[1], p2[1], p1[1]], c)

    def on_click_random(self, event):
        self.show_random_frame()
        self.refresh_fig()

    def get_male_geno_day(self, fly_info):
        if self.info_to_male_geno:
            return self.info_to_male_geno[fly_info]
        if not fly_info:
            return "", "", 0
        if fly_info[0] == "_":
            infos = fly_info.split("_")  # "_TB>CS_6"
            male_geno = infos[1]
            male_days = int(infos[2])
            male_date = day_add_s(self.roi_info["exp_date"], -male_days)
        else:
            m = re.match(r"^(\w+?)(\d+)$", fly_info).groups()
            male_geno, day1 = code_to_geno(m[0])
            # if male_geno:
            #     male_date = day_add_s(day1, int(m[1]))
            #     male_days = day_str_diff(self.roi_info["exp_date"], male_date)
            # else:
            #     print("warning: geno %s not in map" % m[0])
            #     male_geno = m[0]
            male_days = int(m[1])
            male_date = day_add_s(self.roi_info["exp_date"], -male_days)

        return male_geno, male_date, male_days

    def get_all_roi_info(self):
        prefix = "_".join(self.roi_info["file"].split("_")[:2]) + "_"
        roi_l = []
        for idx, roi in enumerate(self.rois):
            fly_info = self.fly_info_l[idx] if len(self.fly_info_l) > idx else ""
            if fly_info:
                if self.info_to_male_geno:
                    f = self.info_to_male_geno.get(fly_info)
                    if f:
                        male_geno = f[0]
                        male_date = f[1]
                        male_days = f[2]
                    else:
                        male_geno, male_date, male_days = self.get_male_geno_day(fly_info)
                        print("warning: %s not in info_to_male_geno, use consts %s" % (fly_info, male_geno))
                else:
                    male_geno, male_date, male_days = self.get_male_geno_day(fly_info)
            else:
                male_geno = ""
                male_date = ""
                male_days = 0
            roi_l.append({"idx": idx, "roi": roi, "info": fly_info, "male_geno": male_geno, "male_date": male_date, "male_days": male_days})
        if not roi_l:
            roi_l.append({"idx": 0, "roi": [[0, 0], [self.width, self.height]]})
        return roi_l

    def on_click_btn_confirm(self, event):
        self.roi_info["ROI"] = self.get_all_roi_info()
        print(self.roi_info["ROI"])
        plt.close()


class InputBgInfoUI(object):

    def __init__(self, name, figsize, cap, info, bg_filename):
        self.GRAY_THRESHOLD = 150
        self.show_binary = "binary"
        self.bg_filename = bg_filename
        if info.get("GRAY_THRESHOLD"):
            self.GRAY_THRESHOLD = info.get("GRAY_THRESHOLD")
            self.img_bg = cv2.imread(bg_filename, cv2.IMREAD_GRAYSCALE) if os.path.exists(bg_filename) else calc_bg(cap)
        else:
            self.img_bg = cv2.imread(bg_filename, cv2.IMREAD_GRAYSCALE) if os.path.exists(bg_filename) else calc_bg(cap)
        self.fig, self.axes = plt.subplots(4, 4, num=name, figsize=figsize)
        self.axes = self.axes.flatten()
        plt.subplots_adjust(top=0.95, bottom=0.3, left=0.1, hspace=0.01, wspace=0.01)
        
        self.cap = cap
        self.bg_info = info
        self.total_frame = info["total_frame"]
        self.width = info["width"]
        self.height = info["height"]
        self.rois = info["ROI"]
        self.slider = Slider(plt.axes([0.1, 0.12, 0.8, 0.05]), "", valmin=0.0, valmax=255, valinit=self.GRAY_THRESHOLD)
        self.slider.on_changed(self.on_slider)
        self.rad_binary = RadioButtons(plt.axes([0.22, 0.01, 0.1, 0.1]), ["binary", "fg", "origin"], 0, "k")
        self.rad_binary.on_clicked(self.on_click_binary)
        self.btn_random = Button(plt.axes([0.1, 0.04, 0.1, 0.05]), "random")
        self.btn_random.on_clicked(self.on_click_random)
        self.btn_confirm = Button(plt.axes([0.8, 0.04, 0.1, 0.05]), "confirm")
        self.btn_confirm.on_clicked(self.on_click_confirm)
        self.ax_h = plt.axes([0.1, 0.2, 0.8, 0.05])

        self.show_random_frame()
        self.refresh_fig()

    def show(self):
        plt.show()

    def show_random_frame(self):
        self.frame = np.random.randint(0, self.total_frame)
        print(self.frame)
        self.img = read_frame(self.cap, self.frame)

    def on_click_confirm(self, event):
        self.bg_info["GRAY_THRESHOLD"] = self.GRAY_THRESHOLD
        self.bg_info["log"] = self.get_log_str()
        log_csv = open(r"..\log.csv", "a+")
        log_csv.write(self.bg_info["log"] + "\n")

        cv2.imwrite(self.bg_filename, self.img_bg)
        plt.close()
        print(self.bg_info["log"])

    def get_log_str(self):
        return ",".join([str(self.bg_info[x]) for x in ["exp_date", "start", "file", "duration", "temperature", "female_days"]]) + "," + "|".join([i["male_geno"] + "+" + str(i["male_days"]) for i in self.bg_info["ROI"]])

    def on_click_binary(self, val):
        self.show_binary = val
        self.refresh_fig()

    def on_slider(self, val):
        self.GRAY_THRESHOLD = val
        self.refresh_fig()

    def on_click_random(self, event):
        self.show_random_frame()
        self.refresh_fig()

    def draw_gray_hist(self, ax, img_fg):
        ax.cla()
        hist_b = np.array(range(256))
        hist_f = np.zeros((256,))
        img_f = img_fg.flatten().astype(int)
        for i in hist_b:
            hist_f[i] += img_f[img_f == i].size
        ax.bar(hist_b[1:-10], hist_f[1:-10])
        ax.set_xlim(0, 255)

    def refresh_fig(self):
        for i, roi in enumerate(self.rois):
            roi = roi["roi"]
            img = sub_img(self.img, roi)
            img_bg = sub_img(self.img_bg, roi)
            img_fg = remove_bg2(img, img_bg)
            data = img_fg < self.GRAY_THRESHOLD
            if self.show_binary == "fg":
                img = img_fg.astype(int)
            elif self.show_binary == "binary":
                img = data.astype(float)
            ax = self.axes[i]
            ax.cla()
            ax.axis("off")
            ax.imshow(img, cmap=plt.cm.gray, norm=NoNorm())
        for j in range(i, len(self.axes)):
            self.axes[j].axis("off")
        self.draw_gray_hist(self.ax_h, img_fg)
        plt.draw()

    def plot_rect(self, ax, p1, p2, c):
        ax.plot([p1[0], p2[0], p2[0], p1[0], p1[0]], [p1[1], p1[1], p2[1], p2[1], p1[1]], c)
