import os
import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from fpt_bg import sub_img
from fpt_consts import ALL_VIDEO_DIR
from fpt_util import get_meta, pair_to_video_path

last_pair = None
cap = None

def frame_by_pair(pair, frame):
    global last_pair, cap
    _idx = pair.rfind("_")
    video_name, pair_no = pair[:_idx], pair[_idx+1:]
    if cap and last_pair == pair:
        pass
    else:
        for exp_dir in ALL_VIDEO_DIR:
            video_folder = os.path.join(exp_dir, video_name)
            video = os.path.join("E:/", video_folder, video_name + ".avi")
            if os.path.exists(video):
                break
        cap = cv2.VideoCapture(video)
    meta = get_meta(video)
    if not meta:
        return None
    roi = meta["ROI"][int(pair_no)]["roi"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    return sub_img(img, roi)

def show_3_frame(img1, img2, img3, t="*"):
    # cv2.imshow("1", img1)
    # cv2.waitKey(1000)
    #
    # cv2.putText(img2, t, (6, 21), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    # cv2.imshow("1", img2)
    # # cv2.waitKey(-1)
    # cv2.waitKey(3000)
    # cv2.imshow("1", img3)
    # cv2.waitKey(1000)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img1, cmap="Greys_r")
    axes[0].text(6, 21, t)
    axes[1].imshow(img2, cmap="Greys_r")
    axes[2].imshow(img3, cmap="Greys_r")
    plt.show()

def show_frames(pair, frame, t):
    imgs = [frame_by_pair(pair, f) for f in range(frame-6, frame+6)]
    fig, axes = plt.subplots(1, len(imgs), figsize=(len(imgs)*1.5, 2))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0, wspace=0)
    for i, img in enumerate(imgs):
        if img is not None:
            axes[i].axis("off")
            axes[i].imshow(img, cmap="Greys_r")
    axes[6].text(6, 21, t)
    plt.show()

def play_pair_video(pair, s, e, out_video=None):
    if e < s:
        e += s
    video, video_name, pair_no, idx = pair_to_video_path(pair)
    meta = get_meta(video)
    if not meta:
        return None
    roi = meta["ROI"][int(pair_no)]["roi"]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    output_video = None
    if out_video:
        out_size = (300, 300)
        output_video = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"DIVX"), 30, out_size)
    for frame in range(s, e):
        ret, img1 = cap.read()
        img = sub_img(img1[:, :, 1], roi)
        cv2.imshow("1", img)
        cv2.waitKey(50)
        if output_video:
            img = cv2.resize(sub_img(img1, roi), out_size)
            output_video.write(img)
    cap.release()
    if output_video:
        output_video.release()

def csv_to_video(df, out_video, need_turn=False):
    out_size = (300, 300)
    output_video = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"DIVX"), 15, out_size)
    for i, row in df.iterrows():
        pair, s, e = row["pair"], int(row["s"]), int(row["e"])
        if e < s:
            e += s
        t = [int(a) for a in row["turns"].split()] if need_turn else []
        video, video_name, pair_no, idx = pair_to_video_path(pair)
        meta = get_meta(video)
        if not meta:
            continue
        roi = meta["ROI"][int(pair_no)]["roi"]
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for frame in range(s, e):
            ret, img = cap.read()
            img = sub_img(img, roi)
            img = cv2.resize(img, out_size)
            cv2.putText(img, pair, (6, 21), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.putText(img, pair, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            cv2.putText(img, "%d-%d %d" % (s, e, frame), (6, 41), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.putText(img, "%d-%d %d" % (s, e, frame), (5, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            if (frame - s) in t:
                cv2.putText(img, pair, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                for ii in range(10):
                    output_video.write(img)
            output_video.write(img)
        cap.release()
    output_video.release()

def load_txt(txt):
    import numpy as np
    m = np.array([l.split() for l in open(txt).readlines()])
    df = pd.DataFrame(m, columns=["pair", "s", "e", "a"])
    return df

if __name__ == '__main__':
    # argv = ("a "+"20200608_151311_A_1 50619 73 25").split()
    # argv = ("a "+"20190727_151557_1_15 40906 41056").split()
    # argv = ("a "+"20200619_143308_1_0 118535 118685").split()
    # argv = ("a "+"20200616_132131_1_4 82446 82508").split()
    # argv = ("a "+"20200426_134619_2_0 178681 178783").split()
    # argv = ("a "+"20200423_160921_2_8 189043 189112").split()
    # argv = ("a "+"20200423_160921_2_14 10769 10935").split()
    # argv = ("a "+"20190916_162459_1_10 74873 74997").split()
    # argv = sys.argv

    from manuscript_figures import plot_fab_alpha_bout
    for a in ["20190916_162459_1_10_74873,74997",
                "20200423_133926_1_11_7367,7661",
                "20200620_144304_1_15_19042,19193",
                "20200426_134619_2_0_99888,100094",

                # "20200619_143308_1_0_118535,118685",
                # "20200616_132132_2_6_109404,109646",
                # "20200607_153503_1_2_215213,215320",
                # "20200423_133926_2_13_115103,115207",
              ]:
    # for a in ["20200620_144304_1_15_19042, 19193",
    #         "20200620_144304_1_10_158584, 159070",
    #         "20200619_143308_1_1_55579, 55639",
    #         "20200619_143308_1_1_60341, 60397",
    #         "20200616_132132_2_9_127184, 127257",
    #         "20200616_132132_2_6_109404, 109646",
    #         "20200607_153503_1_5_198792, 199165",
    #         "20200607_153503_1_2_215213, 215320",
    #         "20200602_144312_2_5_185245, 185382",
    #         "20200426_134619_2_0_99888, 100094",
    #         "20200424_163846_1_11_11946, 12026",
    #         "20200424_132718_1_5_7759, 7838",
    #         "20200423_160921_2_12_9347, 9482",
    #         "20200423_160921_1_0_220717, 220901",
    #         "20200423_133926_2_13_115103, 115207",
    #         "20200423_133926_1_11_7367, 7661",
    #         "20190605_145501_2_2_12616, 12833",
    #         "20190424_154015_1_1_52496, 52601",
    #         "20190424_143441_1_5_10347, 10406",
    #         "20190424_143441_1_3_22571, 22675",
    #         "20181220_114006_1_4_93729, 94091"]:
        tt = a.split("_")
        pair = "_".join(tt[:-1])
        s, e = tt[-1].split(",")
        # play_pair_video(pair, int(s), int(e), "%s_%s,%s.avi" % (pair, s, e))
        plot_fab_alpha_bout(pair, int(s), int(e))
    exit(0)

    if len(argv) > 3:
        play_pair_video(argv[1], int(argv[2]), int(argv[3]), "%s_%s,%s.avi" % (argv[1], argv[2], argv[3]))
    elif len(argv) > 2:
        img = frame_by_pair(argv[1], int(argv[2]))
        plt.imshow(img, cmap="Greys_r")
        plt.show()
    elif len(argv) > 1:
        if argv[1].endswith(".csv"):
            df = pd.read_csv(argv[1])
            csv_to_video(df, argv[1].replace(".csv", ".avi"), argv[1].endswith("turn.csv"))
        elif argv[1].endswith(".txt"):
            df = load_txt(argv[1])
            csv_to_video(df, argv[1].replace(".txt", ".avi"))
    else:
        df0 = pd.read_csv("img/heat_xy.csv")
        print(df0["weights"].mean(), (df0["weights"]>0).mean())
        df0=df0[df0["weights"]>0]
        df = df0.to_dict("index")
        last_frame = -2
        for i in range(len(df)):
            if df[i]["frame"] != last_frame + 1:
                pair, frame = df[i]["pair"], df[i]["frame"]
                # print(pair, frame, df[i]["weights"])
                # if df[i]["weights"] < -1:  # test
                if True:
                    show_frames(pair, frame, str(df[i]["weights"]))
                    print("%d/%d" % (i, len(df)))
            last_frame = df[i]["frame"]


