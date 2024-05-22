import datetime
import sys
from sort.sort import Sort
from ultralytics import YOLO
# debug log
from inspect import currentframe, getframeinfo

# SORT
import skimage

sys.path.insert(0, "./sort")
from sort import *

debug_log = False


def Debug_log(cf, filename, name=None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f"================ [{ct}] file {filename} , line : {cf.f_lineno} {name}")


Debug_log(currentframe(), getframeinfo(currentframe()).filename)
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

Debug_log(currentframe(), getframeinfo(currentframe()).filename)
sys.path.insert(0, "./License_Plate/retinaface")
from retinaface.data_retinaface import cfg_mnet, cfg_re50
from retinaface.layers_retinaface.functions.prior_box import PriorBox
from retinaface.models_retinaface.retinaface import RetinaFace
from retinaface.utils_retinaface.box_utils import decode, decode_landm
from retinaface.utils_retinaface.nms.py_cpu_nms import py_cpu_nms
from test_PaddleOCR import PaddleOcR_Reg
Debug_log(currentframe(), getframeinfo(currentframe()).filename)
SHOW_IMG = False

parser = argparse.ArgumentParser(description="Retinaface")

parser.add_argument(
    "-m",
    "--trained_model",
    default="./retinaface/weights_retinaface/data_extend_tri_tu_Minh29_9_final/mobilenet0.25_epoch_235.pth",
    type=str,
    help="Trained state_dict file path to open",
)
parser.add_argument(
    "--network", default="mobile0.25", help="Backbone network mobile0.25 or resnet50"
)
parser.add_argument(
    "--cpu", action="store_true", default=True, help="Use cpu inference"
)
parser.add_argument(
    "--confidence_threshold", default=0.02, type=float, help="confidence_threshold"
)
parser.add_argument("--top_k", default=5000, type=int, help="top_k")
parser.add_argument("--nms_threshold", default=0.4, type=float, help="nms_threshold")
parser.add_argument("--keep_top_k", default=750, type=int, help="keep_top_k")
parser.add_argument(
    "-s",
    "--save_image",
    action="store_true",
    default=True,
    help="show detection results",
)
parser.add_argument(
    "--vis_thres", default=0.7, type=float, help="visualization_threshold"
)
args = parser.parse_args()

class model_yolo:
    def __init__(self, pretrained_model_path):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.device = torch.device("cpu")
        self.model = YOLO(pretrained_model_path)
        # state_dict = torch.load(pretrained_model_path, map_location=self.device)
        # self.model.load_state_dict(state_dict["model"])
        # self.model.to(self.device)

        sort_max_age = 30
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        self.sort_tracker = Sort(
            max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh
        )
    def find_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        a11 = y1 - y2
        a12 = x2 - x1
        b1 = (x1*y2 - x2*y1)

        a21 = (y3 - y4)
        a22 = (x4 - x3)
        b2 = (x3*y4 - x4*y3)
        A = np.array([[a11, a12],
                    [a21, a22]])

        B = -np.array([b1,b2])

        intersection = np.linalg.solve(A, B)
        x, y = intersection[0], intersection[1]
        
        return x, y        

    def detect_pose(self, image_path, file_img=True, check_sort=True):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        if file_img:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            img_raw = image_path

        # img_raw = cv2.resize(img_raw, (1280, 720)) 

        img_raw_heigh, img_raw_width, img_raw_depth = img_raw.shape
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        # img -= (104, 117, 123)
        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img).unsqueeze(0)
        # img = img.to(self.device)
        # scale = scale.to(self.device)
        tic = time.time()

        results = self.model(img, conf=0.6, device=self.device, classes=0)
        
        # boxes = (results[0].boxes.xyxy.cpu().numpy().astype(int))[0]
        # print('box', boxes)
        plate_point = []
        plate_boxes = []
        for result in results:
            keypoints = result.keypoints
            for i in range(len(keypoints)):
                keypoints_i = keypoints.data[i]
                keypoints_list = keypoints_i.tolist()
                orig_img = result.orig_img
                if len(keypoints_list) >= 4:
                    x1, y1 = keypoints_list[0]
                    x2, y2 = keypoints_list[1]
                    x3, y3 = keypoints_list[2]
                    x4, y4 = keypoints_list[3]
                    xc, yc = self.find_intersection((x1, y1, x3, y3), (x2, y2, x4, y4))
                    plate_point.append([x1, y1, x2, y2, xc, yc, x4, y4, x3, y3])
            boxes = result.boxes
            for i in range(len(boxes)):
                box_i = boxes.data[i]
                box_list = box_i.tolist()
                orig_img = result.orig_img
                if len(box_list) >= 4:
                    x_tl, y_tl , x_br, y_br, confident = box_list[:5]
                    plate_boxes.append([x_tl, y_tl, x_br, y_br, confident])
                    
        if len(plate_boxes)!=0:
            dets = np.concatenate((plate_boxes, plate_point), axis=1)
            print('dets', dets)
            results_post = []
            if check_sort:
                # tracking
                dets_to_sort = np.empty((0, 7))
                for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    conf = float(b[4])

                    # print('type(conf) : ', type(conf))
                    for i in range(0, len(b)):
                        if b[i] < 0:
                            b[i] = 0
                    b = list(map(int, b))
                    b_kps = b[5:]
                    dets_to_sort = np.vstack(
                        (dets_to_sort, np.array([b[0], b[1], b[2], b[3], conf, 0, b_kps]))
                    )

                tracked_dets = self.sort_tracker.update(dets_to_sort)
                # print('tracked_dets : ', tracked_dets)

                for det in tracked_dets:
                    conf = det[-1]
                    det = list(map(int, det))
                    for i in range(0, 4):
                        if det[i] < 0:
                            det[i] = 0
                    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                    id = det[8]
                    box_kps = det[9:-1]
                    Debug_log(
                        currentframe(), getframeinfo(currentframe()).filename, len(box_kps)
                    )
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, det)
                    Debug_log(
                        currentframe(), getframeinfo(currentframe()).filename, box_kps
                    )

                    text = str(id) + "_" + str(conf)

                    results_post.append([[x1, y1, x2, y2], box_kps, id, conf])
                    if SHOW_IMG:
                        cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cx = x1
                        cy = y1 + 12
                        cv2.putText(
                            img_raw,
                            text,
                            (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.5,
                            (255, 255, 255),
                        )

                        # landms
                        cv2.circle(img_raw, (box_kps[0], box_kps[1]), 1, (0, 0, 255), 4)
                        cv2.circle(img_raw, (box_kps[2], box_kps[3]), 1, (0, 255, 255), 4)
                        cv2.circle(img_raw, (box_kps[4], box_kps[5]), 1, (255, 0, 255), 4)
                        cv2.circle(img_raw, (box_kps[6], box_kps[7]), 1, (0, 255, 0), 4)
                        cv2.circle(img_raw, (box_kps[8], box_kps[9]), 1, (255, 0, 0), 4)
                        cv2.imshow("img", img_raw)
                        cv2.waitKey(0)
            else:
                for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    conf = float(b[4])

                    # print('type(conf) : ', type(conf))
                    for i in range(0, len(b)):
                        if b[i] < 0:
                            b[i] = 0
                    b = list(map(int, b))
                    b_kps = b[5:]

                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    results_post.append([[x1, y1, x2, y2], b_kps, None, conf])
            print('results_post', results_post)
        else:
            results_post=[]
        return results_post

# debug_log = True

# from PIL import Image, ImageDraw, ImageFont
# import os
# import cv2
# import time

# if __name__ == "__main__":
#     pretrained_model_path = "./model_yolo/train_5_4/weights/best.pt"
#     model_yolo = model_yolo(pretrained_model_path)
#     filename_video = "video/video1.mp4"
#     video_name = os.path.basename(filename_video).split('.')[0]
#     # cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.6.203:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')
#     cap = cv2.VideoCapture(filename_video)

#     output_video_width = 1920
#     output_video_height = 1080
#     output_frame_rate = 10.0

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     output_video = cv2.VideoWriter('output_video.mp4', fourcc, output_frame_rate, (output_video_width, output_video_height))

#     frame_interval = 2 * int(output_frame_rate)
#     frame_counter = 0

#     font_path = "font/MQ_License_Plate_font.ttf" 
#     font_size = 40
#     font = ImageFont.truetype(font_path, font_size)

#     output_folder = "check_reg_follow_frame/video1"
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     frame_index = 0
#     while True:
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, (1920, 1080))
#         if not ret:
#             break
#         frame_index += 1
#         frame_counter += 1
#         if frame_counter < frame_interval:
#             continue
#         if frame_counter >= frame_interval:
#             frame_counter = 0
#         t1 = time.time()
#         dets_plate = model_yolo.detect_pose(
#             frame, file_img=False, check_sort=False
#         )  # [[x1, y1, x2, y2], box_kps, id, conf]
#         print("det_plate:", dets_plate)
#         if len(dets_plate) == 0:
#             continue
#         reg_plate = PaddleOcR_Reg()
#         for det in dets_plate:
#             box = det[0]
#             text_plate, crop_img = reg_plate.reg_plate(frame, [det]) 
#             frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             draw = ImageDraw.Draw(frame_pil)
#             print("text_plate",text_plate)
#             if text_plate:
#                 draw.text((box[0], box[1] - 30), f'{text_plate[0][1]}', fill=(0, 255, 0), font=font)
#                 frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

#                 image_name = f"{video_name}_frame_{frame_index}_{text_plate[0][1]}.jpg"
#                 cv2.imwrite(os.path.join(output_folder, image_name), crop_img)

#         t2 = time.time()
#         print("**********************************************************************")
#         frame_resized = cv2.resize(frame, (output_video_width//2, output_video_height//2))
#         output_video.write(frame_resized) 
#         cv2.imshow('Video', frame_resized)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     output_video.release()
#     cv2.destroyAllWindows()

from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import time

if __name__ == "__main__":
    pretrained_model_path = "./model_yolo/train_8_4/weights/best.pt"
    model_yolo = model_yolo(pretrained_model_path)

    input_image_folder = "test_hb/plate-bn/04-03-2024"  
    output_folder = "output_folder_test_hb/plate-bn/04-03-2024"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    font_path = "font/MQ_License_Plate_font.ttf" 
    font_size = 40
    font = ImageFont.truetype(font_path, font_size)

    for image_name in os.listdir(input_image_folder):
        image_path = os.path.join(input_image_folder, image_name)
        
        image = Image.open(image_path)
        # import matplotlib.pyplot as plt
        # plt.subplot(1, 2, 2)
        # plt.imshow(image)
        # plt.title('ROI')
        # plt.show()
        image_np = np.array(image)
        
        dets_plate = model_yolo.detect_pose(image_np, file_img=False, check_sort=False)
        
        draw = ImageDraw.Draw(image)
        reg_plate = PaddleOcR_Reg()
        for det in dets_plate:
            box = det[0]
            text_plate, crop_img = reg_plate.reg_plate(image_np, [det]) 
            if text_plate:
                draw.text((box[0], box[1] - 30), text_plate[0][1], fill=(0, 255, 0), font=font)
        
        output_image_path = os.path.join(output_folder, image_name)
        image.save(output_image_path)
