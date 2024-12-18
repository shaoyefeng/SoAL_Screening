# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
from fpt_consts import DURATION_LIMIT, PRINT_FRAME, PRINT_FRAME1
from fpt0_util import load_dict, save_dict, to_feat_s, get_feat_header


class IdAssign(object):
    def __init__(self, video, task_id, task_q):
        # redirect_output("log/ia%d_%s.log" % (task_id, os.path.basename(video)))
        self._video = video
        self._task_id = task_id
        self._task_q = task_q
        self._meta = load_dict(video.replace(".avi", "_meta.txt"))
        self._total_frame = min(DURATION_LIMIT * self._meta["FPS"], self._meta["total_frame"]) or self._meta["total_frame"]
        self._max_frame = self._total_frame - 1
        roi_l = self._meta.get("ROI")
        self._max_fly = self._max_frame * len(roi_l) * 2
        self._roi = [r["roi"] for r in roi_l]
        self._feat_l = []
        self._feat_f_l = []
        for roi_i in range(len(roi_l)):
            self._feat_f_l.append(self.init_mata_feat(roi_i))
            self._feat_l.append([])
        self._feat_header = get_feat_header()

    def init_mata_feat(self, roi_i):
        parent = os.path.dirname(self._video)
        base = os.path.basename(self._video)
        parent = os.path.join(parent, str(roi_i))
        not os.path.exists(parent) and os.mkdir(parent)
        feat = os.path.join(parent, base.replace(".avi", "_%d_feat.csv" % roi_i))
        meta = os.path.join(parent, base.replace(".avi", "_%d_meta.txt" % roi_i))
        meta_d = self._meta.copy()
        meta_d["ROI"] = meta_d["ROI"][roi_i]
        save_dict(meta, meta_d)
        return feat

    @staticmethod
    def process(video, task_id, q):
        inst = IdAssign(video, task_id, q)
        inst.proc_video()

    def proc_video(self):
        roi_num = len(self._roi)
        print("[id_assign]#%d ---start: %dframes %drois %s" % (self._task_id, self._total_frame, roi_num, datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")))
        fly_info = []
        last_fly_info = []
        for roi_i in range(roi_num):
            last_fly_info.append([])
            fly_info.append([])
        start_ts = time.time()
        last_ts = start_ts
        frame = 0
        i = 0
        fs = [open(f, "w") for f in self._feat_f_l]
        for f in fs:
            f.write(",".join(self._feat_header) + "\n")
        while True:
            pack = self._task_q.get()
            if pack is None:
                for f in fs:
                    f.close()
                break
            roi_i, frame, reg, reg_n, points = pack
            # fly_info[roi_i].append((frame, reg, reg_n, points))  # NOTE: diff
            if True: #len(fly_info[roi_i]) >= FLY_NUM:
                # id_assign(fly_info[roi_i], last_fly_info[roi_i])

                # last_fly_info[roi_i] = fly_info_c
                # fly_info[roi_i] = []
                fs[roi_i].write(to_feat_s([(frame, reg, reg_n, points)]) + "\n")
                i += 1
                # if frame >= self._max_frame:  # FIXME: closed before finish!!!
                #     fs[roi_i].close()
                #     break
                if i >= self._max_fly:  # NOTE: cant stop if FRAME_STEP > 1
                    for f in fs:
                        f.close()
                    break

                if frame % PRINT_FRAME == PRINT_FRAME1 and roi_i == 0:
                    ts = time.time()
                    # d_ts = ts - last_ts # NOTE: not correct
                    # last_ts = ts
                    print("[id_assign]#%d (%d/%d) %drois a%.2fframe/s %d%%" % (self._task_id, frame, self._total_frame, roi_num, frame/(ts-start_ts), frame*100.0/self._total_frame))

        end_ts = time.time()
        d = end_ts - start_ts
        # print("[PPPPP] id_assign_q: %f %f/%d" % (d, prof_t, prof_c))
        # for roi_i, feat in enumerate(self._feat_l):
        #     df = DataFrame()
        #     total_frame = len(self._feat_l[roi_i])
        #     for info in self._feat_l[roi_i]:
        #         df = df.append([to_feat(info)], ignore_index=True)
        #     df.columns = self._feat_header
        #     df.loc[0, "frames"] = total_frame
        #     df.loc[total_frame - 1, "frames"] = -total_frame
        #     df.to_csv(self._feat_f_l[roi_i])
        # print("[PPPPP] id_assign: %f/%d, write_df: %f" % (prof_t, prof_c, time.time() - s2))
        finish_s = "#%d (%d(%d)/%.2fs=%.2fframe/s)\n" % (self._task_id, frame, self._total_frame, d, frame / d) + datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")
        print("[id_assign]---finish: %s %s" % (finish_s, self._video))

        state_file = os.path.join(os.path.dirname(self._video), ".state")
        f = open(state_file, "w")
        f.write("finish\n" + finish_s)
        f.close()

# if __name__ == '__main__':
#     # import sys
#     # id_assign_file(sys.argv[1])
#     a=[(38819, (236.5, (25.012317657470703, 167.6109619140625), 65.20472717285156, 31.37689971923828, 11.00249195098877,
#               (447.0123176574707, 664.6109619140625)), 2, np.array([17.04197623, 25.01387041, 28.1936708, 34.21586891,
#                                                                  37.7951487, 156.41019352, 167.80515589, 183.26776582,
#                                                                  184.78891952, 187.34245069, 1., 1.,
#                                                                  1., 1., 1.])), (38819, (
#     199.0, (5311.3203125, 3923.333740234375), 35.29325866699219, 12904.91796875, 183.7032928466797,
#     (5733.3203125, 4420.333740234375)), 2, np.array([5.32771447e+03, 5.33852098e+03, 5.32101180e+03, 5.29724867e+03,
#                                                   5.33033673e+03, 3.92034396e+03, 3.91924875e+03, 3.91947227e+03,
#                                                   3.94153885e+03, 3.93313650e+03, 1.00000000e+00, 1.00000000e+00,
#                                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00]))]
#     b=[(38818, (
#     243.5, (25.88311767578125, 169.1397705078125), 65.61027526855469, 32.48033142089844, 11.378422737121582,
#     (447.88311767578125, 666.1397705078125)), 2, np.array([18.4285512, 25.84881275, 28.71365597, 34.52003523,
#                                                         38.59498158, 155.93487085, 168.96424884, 183.52074931,
#                                                         185.76435907, 188.25690992, 1., 1.,
#                                                         1., 1., 1.])), (38818, (
#     199.5, (60.53291702270508, 206.46481323242188), 35.002655029296875, 42.7540168762207, 13.27085018157959,
#     (482.5329170227051, 703.4648132324219)), 2, np.array([44.93519598, 55.98675723, 61.13757273, 77.60173838,
#                                                        74.83378891, 194.59008385, 203.2156059, 215.54660774,
#                                                        211.93484508, 206.44650227, 1., 1.,
#                                                        1., 1., 1.]))]
#     print(id_assign(a, b))

