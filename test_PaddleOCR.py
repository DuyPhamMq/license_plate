import base64
import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from PaddleOCR.tools.infer.include_align import align_image, check_format_plate, check_format_plate_append, check_plate_sqare, mode_rec
from PaddleOCR.tools.infer.predict_det import TextDetector
import time
import cv2
import numpy as np
from PIL import Image

# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# os.environ["FLAGS_allocator_strategy"] = 'auto_growth'


sys.path.insert(0, "./License_Plate/PaddleOCR")
import argparse
import datetime

# debug log
from inspect import currentframe, getframeinfo

import tools.infer.predict_cls as predict_cls
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read_gif, get_image_file_list

####inclue
from tools.infer.include_align import *
from tools.infer.predict_rec import TextRecognizer
from tools.infer.predict_system import TextSystem
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image

debug_log = False


def Debug_log(cf, filename, name=None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f"================ [{ct}] file {filename} , line : {cf.f_lineno} {name}")


logger = get_logger()


class PaddleOcR_Reg:
    def __init__(
        self,
        IS_VISUALIZE=False,
        MODE_REC=True,
        MODE_DET=True,
        MODE_SYS=False,
        threshold=0.9,
    ):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.IS_VISUALIZE = IS_VISUALIZE
        self.MODE_REC = MODE_REC  # mode recognition
        self.MODE_DET = MODE_DET  # mode detect text
        self.MODE_SYS = MODE_SYS  # mode both recognition and detect text
        self.args = utility.parse_args()
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        if self.MODE_REC:
            self.args.rec_model_dir = "./model_ocr/best_model_4_4"
            self.args.rec_image_shape = "3, 48, 320"
            self.args.rec_char_dict_path = "./model_ocr/licence_plate_dict.txt"
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.threshold = 0.9  # check reg plate smaller then exit
        #############load model paddle#############
        if self.MODE_SYS:
            print("run MODE_SYS")
            text_sys = TextSystem(self.args)
        if self.MODE_REC:
            print("run MODE_REC")
            self.text_recognizer = TextRecognizer(self.args)
            # 3, 48, 320
            hight_ini = 80
            weigh_ini = 240
            img_ini = out_img = np.zeros((hight_ini, weigh_ini, 3), np.uint8)
            rec_res_ini, _ = self.text_recognizer([img_ini])
            print("run MODE_REC done")
        if self.MODE_DET:
            print("run MODE_DET")
            self.text_detector = TextDetector(self.args)
        print("load model done 1")
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)

    def reg_plate(self, img_raw, box_kpss):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        image_decode = img_raw.copy()
        count_plate = 0
        results = []
        for box_kps in box_kpss:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            id = box_kps[2]
            kps_int = box_kps[1]

            kps = [
                [kps_int[0], kps_int[1]],
                [kps_int[2], kps_int[3]],
                # [kps_int[4], kps_int[5]],
                [kps_int[6], kps_int[7]],
                [kps_int[8], kps_int[9]],
                
            ]
            x1, y1, x2, y2 = box_kps[0]
            # count_kp = 0
            bbox = box_kps[0]
            kpt = kps

            pnt0 = np.maximum(kpt[0],bbox[:2])
            pnt1 = np.array([np.minimum(kpt[1][0],bbox[2]), np.maximum(kpt[1][1],bbox[1])])
            pnt2 = np.minimum(kpt[3],bbox[2:4])
            pnt3 = np.array([np.maximum(kpt[2][0],bbox[0]), np.minimum(kpt[2][1],bbox[3])])
            points_norm = np.concatenate(([pnt0], [pnt1], [pnt2], [pnt3]))
            print(points_norm)
            coordinate_dict = {
                "top-right": (points_norm[1][0]-x1, points_norm[1][1]-y1),
                "bottom-right": (points_norm[2][0]-x1, points_norm[2][1]-y1),
                "top-left": (points_norm[0][0]-x1, points_norm[0][1]-y1),
                "bottom-left": (points_norm[3][0]-x1, points_norm[3][1]-y1),
            }
            print(coordinate_dict)
            Debug_log(
                currentframe(),
                getframeinfo(currentframe()).filename,
                image_decode.shape,
            )
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, box_kps[0])

            img_crop_lp = image_decode[y1:(y2), x1:x2]
            
            img_copy = img_crop_lp.copy()

            Debug_log(
                currentframe(), getframeinfo(currentframe()).filename, img_copy.shape
            )
            # align image
            cropped_img = align_image(img_copy, coordinate_dict)

            ####################detect and recognize#############################
            img_list = []
            # detect and recognize in horizontal number plate
            check_plate_sqare_, img_list = check_plate_sqare(cropped_img)
            if check_plate_sqare_ is not None:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                img = check_plate_sqare_
            else:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                img = cropped_img
                img_list = [cropped_img]
            starttime = time.time()
            if (
                self.MODE_SYS
            ):  # If you combine the plate into 1 line, it will be considered as 1 word sometime 2 word
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                dt_boxes, rec_res = text_sys(img)
                txt = ""
                for text, score in rec_res:
                    txt = str(text) + ":" + str(round(score, 2)) + "--"
                if txt == "":
                    txt = "Null"
                image_decode = cv2.putText(
                    image_decode,
                    txt,
                    (x1 - 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                mode_sys(dt_boxes, rec_res, img, args, count_plate)
            if self.MODE_REC:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                print('wait text_recognizer')
                rec_res, _ = self.text_recognizer(img_list)
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                txt_result, check_acc, arv_acc = mode_rec(rec_res, self.threshold)
                if check_acc: #samller threshold
                    continue
                print("txt_result", txt_result)

                result_format_check = check_format_plate(txt_result)
                # result_format_check = True

                if result_format_check == False:
                    Debug_log(
                        currentframe(),
                        getframeinfo(currentframe()).filename,
                        "CHECK FORMAT PLATE FALSE",
                    )
                    Debug_log(
                        currentframe(),
                        getframeinfo(currentframe()).filename,
                        txt_result,
                    )
                txt_result = check_format_plate_append(
                    txt_result
                )  # return lp + '#' or None
                if txt_result == None:
                    continue

            if self.MODE_DET:
                print("wait det")
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                dt_boxes, _ = self.text_detector(img)

            elapse = time.time() - starttime
            # print('time inferen det and rec : ', elapse)
            #####################################################################
            count_plate += 1
            results.append([id, txt_result, [x1, y1, x2, y2], [], [], []])
            if self.IS_VISUALIZE:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                print("IS_VISUALIZE or SAVE_VIDEO")
                cv2.rectangle(image_decode, (x1, y1), (x2, y2), (255, 0, 0), 2)
                for kp in kps:
                    cv2.circle(image_decode, kp, 9, (255, 255, 0), -1)
        print("check", results)
        return results, cropped_img


if __name__ == "__main__":
    start_time = time.time()
    img_paths = "./image_crop/*.jpeg"
    reg_plate = PaddleOcR_Reg()
    for i in range(0, 10):
        path = "./image_crop/test.jpg"
        frame = cv2.imread(path)
        box_kpss = [
            # [
            #     [0, 516, 150,640],
            #     [4, 533, 123, 541, 64, 575, 130, 622, 9, 609],
            #     11,
            #     0.9968008995056152,
            # ],
            # [
            #     [363, 345, 417, 394],
            #     [363, 354, 412, 345, 415, 385, 366, 392, 415, 385],
            #     11,
            #     0.9968008995056152,
            # ]
            [
                [0, 0, frame.shape[1], frame.shape[0]],
                [0, 0, frame.shape[1], 0, 0, 0, 0, frame.shape[0], frame.shape[1], frame.shape[0]],
                11,
                0.9968008995056152,
            ]
        ]
        results = reg_plate.reg_plate(frame, box_kpss)
        print("result:", results)
    end_time = time.time()
    run_time = end_time - start_time

    print("Thời gian chạy của chương trình:", run_time, "giây")
        # for result in results:
        #     id = result[0]
        #     txt_result = str(id) + "-" + result[1]
        #     x1, y1, x2, y2 = result[2]
        #     cropped_img_height, cropped_img_width, _ = result[3].shape
        #     x_crop = result[4][0][0]
        #     y_crop = result[4][0][1]
        #     frame[
        #         y_crop : y_crop + cropped_img_height,
        #         x_crop : x_crop + cropped_img_width,
        #     ] = result[3]
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #     cv2.rectangle(
        #         frame,
        #         (x_crop, y_crop),
        #         (x_crop + cropped_img_width, y_crop + cropped_img_height),
        #         (147, 125, 255),
        #         2,
        #     )
        #     cv2.line(
        #         frame,
        #         (x_crop, y_crop + int(cropped_img_height / 2)),
        #         (x_crop + cropped_img_width, y_crop + int(cropped_img_height / 2)),
        #         (147, 125, 255),
        #         2,
        #     )
        #     cx = x1
        #     cy = y1 - 12
        #     cv2.putText(
        #         frame,
        #         txt_result,
        #         (cx, cy),
        #         cv2.FONT_HERSHEY_DUPLEX,
        #         0.7,
        #         (255, 255, 255),
        #         2,
        #         cv2.LINE_AA,
        #     )

        #     window_name = "frame"
        # # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        # # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # # cv2.imshow(window_name, frame)
        # # cv2.waitKey(0)

        #     print(
        #         "*********************************************************************************", txt_result
        #     )
