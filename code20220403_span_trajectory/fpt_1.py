# -*- coding: utf-8 -*-

import os
# import time
# import sys
# from datetime import datetime
# from multiprocessing import Process, Queue
# from fpt1_preprocess import preprocess
# process import times (config=1,1,1,n): fpt_1, PredictParts, RegionSeg, RegionSeg.process_sub*n, IdAssign
# from fpt2_region_seg_1 import RegionSeg  # NOTE: diff
# from fpt3_predict_parts_1 import PredictParts  # NOTE: diff
from fpt_consts import IMG_QUEUE_CAPACITY, PRED_PROCESS_NUM, VIDEO_TODO_DIR, MAX_TASK

# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = "false"

TEST_TASKS = """
20190902_150503_A
20190902_174522_B
20190903_144331_A
20190903_144332_B
20190903_162013_B
20190904_144842_A
""".split()
# ["20191101_165013_B"]

def fpt_main():
    print("start...")
    from multiprocessing import Process, Queue
    from fpt3_predict_parts_1 import PredictParts
    pred_q_l = []
    for i in range(PRED_PROCESS_NUM):
        pred_q = Queue(maxsize=IMG_QUEUE_CAPACITY)
        pred_p = Process(target=PredictParts.process, args=(pred_q, i))
        pred_p.start()
        pred_q_l.append(pred_q)

    from fpt2_region_seg_1 import RegionSeg
    from datetime import datetime
    import time
    last_task_n = 0
    task_id = 0
    task_finish = 0
    total_task = 0
    while True:
        print("[fpt] check dir... (%d tasks running) (%d/%d tasks finished)" % (last_task_n, task_finish, total_task))
        task_n = 0
        task_finish = 0
        total_task = 0
        for video_dir in os.listdir(VIDEO_TODO_DIR):
            video_dir_path = os.path.join(VIDEO_TODO_DIR, video_dir)
            if os.path.isdir(video_dir_path):
                state_file = os.path.join(video_dir_path, ".state")
                total_task += 1
                if not os.path.exists(state_file):
                    print("write init file", state_file)
                    f = open(state_file, "w")
                    f.write("init\n")
                    f.close()
                if os.path.exists(state_file):
                    try:
                        f = open(state_file, "r")
                    except Exception:
                        continue
                    state = f.readline().replace("\n", "")
                    if state == "init":
                        if last_task_n >= MAX_TASK or task_n >= MAX_TASK:
                            f.close()
                            continue
                        print("[init task]: %s" % video_dir)
                        video = os.path.join(video_dir_path, video_dir + ".avi")
                        if not os.path.exists(video):
                            print("error!!!")
                            # video = os.path.join(video_dir_path, video_dir + "_t.avi")
                            # video = preprocess(video)
                        pred_q = pred_q_l[task_id % PRED_PROCESS_NUM]
                        pred_q.put([task_id, video])

                        rs_p = Process(target=RegionSeg.process, args=(video, task_id, pred_q))
                        rs_p.start()

                        f.close()
                        f = open(state_file, "w")
                        f.write("running\n" + "%d %s\n" % (task_id, datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")))

                        task_id += 1
                        task_n += 1
                        # break
                    elif state == "running":
                        task_n += 1
                    elif state.startswith("finish"):
                        task_finish += 1

                    f.close()

        last_task_n = task_n
        if task_finish >= total_task:
            break
        time.sleep(15)
    for pred_q in pred_q_l:
        pred_q.put([None, None])
    print("all finished")

# def copy_finish():
#     import shutil
#     for video_dir in os.listdir(VIDEO_TODO_DIR):
#         video_dir_path = os.path.join(VIDEO_TODO_DIR, video_dir)
#         if os.path.isdir(video_dir_path):
#             state_file = os.path.join(video_dir_path, ".state")
#             if os.path.exists(state_file):
#                 f = open(state_file, "r")
#                 state = f.readline().replace("\n", "")
#                 if state == "finish":
#                     info = str(f.readline())[:-1]
#                     f.close()
#                     print("%s(%s) finish, move to finish dir" % (video_dir, info))
#                     # os.rename(video_dir_path, video_dir_path + "_finish")
#                     shutil.move(video_dir_path, os.path.join(VIDEO_FINISH_DIR, video_dir))

def write_init_file(file_l):
    for file in file_l:
        f = open(file, "w")
        f.write("init\n")
        f.close()

if __name__ == '__main__':
    file_l = []
    import sys, os
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if len(path) <= 3:
            if path != "all":
                MAX_TASK = int(path)
            for video_dir in os.listdir(VIDEO_TODO_DIR):
                video_dir_path = os.path.join(VIDEO_TODO_DIR, video_dir)
                if os.path.isdir(video_dir_path):
                    file_l.append(os.path.join(video_dir_path, ".state"))
            # else:
            #     tasks = int(path)
            #     for tt in range(tasks):
            #         file_l.append(os.path.join(VIDEO_TODO_DIR, TEST_TASKS[tt], ".state"))  # TEST
        else:
            # if path == "finish":
            #     copy_finish()
            # else:
                if not os.path.isdir(path):
                    path = os.path.dirname(path)
                file_l.append(os.path.join(path, ".state"))
    write_init_file(file_l)
    fpt_main()

"""
1. preprocess
    frame_align (optional)
    input_meta_info
        input_exp_info
        input_roi_info
        input_bg_info

-input: frame
2. region_seg
    background subtraction
    region segmentation
    center_img
3. predict_parts
4. id_assign
-output: feat

directory structure:

 #: file property
 {}: input item
 *: copy previous

exp(D:\exp)
    |-log.csv
    |   |-exp_date, start, file, duration, temperature, female_days, male_geno_days
    |-exp.xlsx
    |-20190807_140000_A
        |-.state (init|running|finish)
        |-20190807_140000_A.avi (0_frame_align)
            |-#Duration, #FPS
        |-20190807_140000_A_bg.bmp (0_calc_bg)
        |-20190807_140000_A.log
        |-20190807_140000_A_meta.txt (0_input_meta_info)
            |-file, total_frame, width, height, camera, start, end, duration, FPS
            |-VERSION, ROUND_ARENA, MODEL_FOLDER
            |-{FEAT_SCALE}, {AREA_RANGE}, {temperature}, {exp_date}, {female_date}, female_days
            |-{GRAY_THRESHOLD}
            |-ROI
                |-idx, {roi(2x2)}, {info}, {male_geno}, {male_date}, male_days
                |-...
            |-log
        |-0
            |-20190807_140000_A_0_feat.csv
               |-frame, time, reg_n, 1:area, 1:pos:x, 1:pos:y, 1:ori, 1:e_maj, 1:e_min, 1:point:xs, 1:point:ys...
               |-...
            |-20190807_140000_A_0_meta.txt
                |-*INFO
                |-*ROI
                    |-idx, {roi(2x2)}, {info}, {male_geno}, {male_date}, male_days
    |-data
        |-all
        |-center
        |-good
"""
