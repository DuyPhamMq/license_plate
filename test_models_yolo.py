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
    "--vis_thres", default=0.5, type=float, help="visualization_threshold"
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

        # img_raw_heigh, img_raw_width, img_raw_depth = img_raw.shape
        # img = np.float32(img_raw)

        # im_height, im_width, _ = img.shape
        # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        # img -= (104, 117, 123)
        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img).unsqueeze(0)
        # img = img.to(self.device)
        # scale = scale.to(self.device)
        tic = time.time()

        results = self.model(img_raw, conf = args.vis_thres , device=self.device, classes=0)
        # boxes = (results[0].boxes.xyxy.cpu().numpy().astype(int))[0]
        # print('box', boxes)
        results_posts = []
        for result in results:
            plate_point = []
            plate_boxes = []
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
                print("***************box_list***************", box_list)
                orig_img = result.orig_img
                if len(box_list) >= 4:
                    x_tl, y_tl , x_br, y_br, confident = box_list[:5]
                    plate_boxes.append([x_tl, y_tl, x_br, y_br, confident])
            if len(plate_boxes) == 0:
                results_posts.append([])
                continue
        
            dets = np.concatenate((plate_boxes, plate_point), axis=1)
            print('dets', dets)
            results_post = []
            if check_sort:
                # tracking
                dets_to_sort = np.empty((0, 16))
                for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    conf = float(b[4])

                    # print('type(conf) : ', type(conf))
                    
                    for i in range(0, len(b)):
                        if b[i] < 0:
                            b[i] = 0
                    b = list(map(int, b))
                    # b_kps = [b[5], b[6], b[7], b[8], b[11], b[12], b[13], b[14]]
                    b_kps = b[5:]
                    print("b_kps:", *b_kps)

                    print("dets_to_sort", dets_to_sort)

                    print("b[0], b[1], b[2], b[3], conf, 0, b_kps:", b[0], b[1], b[2], b[3], conf, 0, b_kps)   

                    dets_to_sort = np.vstack((dets_to_sort, np.array([b[0], b[1], b[2], b[3], conf, 0, *b_kps])))
                    print("dets_to_sort********************************", dets_to_sort)


                tracked_dets = self.sort_tracker.update(dets_to_sort)
                print('tracked_dets : ', tracked_dets)

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
            results_posts.append(results_post)
        return results_posts

# debug_log = True
class model_retina:
    def __init__(self):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.cfg = None
        if args.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif args.network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        self.device = "cpu"  # torch.device("cpu" if args.cpu else "cuda")
        self.retinaPlateNet = RetinaFace(cfg=self.cfg, phase="test")
        self.retinaPlateNet = self.load_model(
            self.retinaPlateNet, args.trained_model, args.cpu
        )
        self.retinaPlateNet.eval()
        print("Finished loading model!")
        cudnn.benchmark = True

        self.retinaPlateNet = self.retinaPlateNet.to(self.device)
        self.resize = 1

        # load model sort
        # sort
        sort_max_age = 30
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        self.sort_tracker = Sort(
            max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh
        )  # {plug into parser}

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print(f"Missing keys:{len(missing_keys)}")
        print(f"Unused checkpoint keys:{len(unused_pretrained_keys)}")
        print(f"Used keys:{len(used_pretrained_keys)}")
        assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
        return True

    def remove_prefix(self, state_dict, prefix):
        """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
        print(f"remove prefix '{prefix}'")
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model, pretrained_path, load_to_cpu):
        print(f"Loading pretrained model from {pretrained_path}")
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        if load_to_cpu:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage
            )
        else:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            # device = 1#torch.cuda.current_device()
            print("retina load device : ", self.device)
            Debug_log(
                currentframe(), getframeinfo(currentframe()).filename, self.device
            )
            pretrained_dict = torch.load(
                pretrained_path,
                map_location=lambda storage, loc: storage.cuda(self.device),
            )
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(
                pretrained_dict["state_dict"], "module."
            )
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, "module.")
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        return model

    def detect_plate(self, image_path, file_img=True, check_sort=True):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        if file_img:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            img_raw = image_path
        img_raw_heigh, img_raw_width, img_raw_depth = img_raw.shape
        # print(img_raw.shape)
        # cv2.imshow('img_raw', img_raw)
        # cv2.waitKey(0)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = self.retinaPlateNet(img)  # forward pass

        # print('retinaPlateNet forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))

        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[: args.keep_top_k, :]
        landms = landms[: args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        print('det',dets)
        results = []
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
                print("b_kps:", b_kps)
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

                results.append([[x1, y1, x2, y2], box_kps, id, conf])
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
                results.append([[x1, y1, x2, y2], b_kps, None, conf])
        return results
def statistics(plates):
    # plate_texts = np.array([plate[1] for plate in plates])

    consecutive_plates = []
    current_id = None
    current_consecutive = []
    for plate in plates:
        id = plate[0]
        if id != current_id:
            if current_consecutive:
                consecutive_plates.append(current_consecutive)
                current_consecutive = []
            current_id = id
        current_consecutive.append(plate)
    for consecutive_list in consecutive_plates[0]:
        x = np.array([list(plate[1]) for plate in consecutive_list])
        for row in x.T:
            unique, counts = np.unique(row, return_counts=True, axis=0)
            maxpos = counts.argmax()
            print(unique[maxpos])


def mergerLP(list_lp_old):
    """
    list_lp_old: a list results = [id, txt_result, [x1, y1, x2, y2], cropped_img, kps, img_raw] of the same number plate
    """
    list_lp = []
    index_frame = len(list_lp_old) // 2
    frame = list_lp_old[index_frame][-1]
    box_kps = list_lp_old[index_frame][2]

    for i in range(len(list_lp_old)):
        if len(list_lp_old[i][1]) != 9:
            list_lp_old[i][1] = list_lp_old[i][1] + "#" * (9 - len(list_lp_old[i][1]))
        list_lp.append(list(list_lp_old[i][1]))

    lp_ret = ""
    for i in range(9):
        column_values = [row[i] for row in list_lp]
        unique, counts = np.unique(column_values, return_counts=True)
        max_index = np.argmax(counts)
        lp_ret += unique[max_index]

    lp_ret = lp_ret.replace("#", "")
    id = list_lp_old[0][0]

    now = datetime.datetime.now()
    return {
        "id": id,
        "txt": lp_ret,
        "box_kps": box_kps,
        "frame": frame,
        "time": now.strftime("%Y-%m-%d_%H-%M-%S"),
    }  # now.strftime("%Y-%m-%d %H:%M:%S")
    # return {'id': id, 'txt':lp_ret, 'time': now.strftime("%Y-%m-%d %H:%M:%S")}

from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import time

if __name__ == "__main__":
    pretrained_model_path = "./model_yolo/train_19_5/weights/yolov8_pose_plate_50k_19_5.pt"
    model_yolo = model_yolo(pretrained_model_path)
    filename_video = "video/video1.mp4"
    video_name = os.path.basename(filename_video).split('.')[0]
    cap1 = cv2.VideoCapture(filename_video)

    # cap1 = cv2.VideoCapture('rtsp://admin:MQ@123456@192.168.6.207:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1')
    # cap2 = cv2.VideoCapture('rtsp://admin:MQ@123456@192.168.6.205:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1')
    # cap2 = cv2.VideoCapture('rtsp://admin:admin123@192.168.6.203:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')

    output_video_width = 1920
    output_video_height = 1080
    output_frame_rate = 10.0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter('output_video.avi', fourcc, output_frame_rate, (output_video_width, output_video_height))

    frame_interval = 2 * int(output_frame_rate)
    frame_counter = 0

    font_path = "font/MQ_License_Plate_font.ttf" 
    font_size = 40
    font = ImageFont.truetype(font_path, font_size)

    output_folder = "/run/user/1000/gvfs/sftp:host=192.168.6.159,user=mq/home/mq/disk2T/duy/plate_test_mq"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    frame_index = 0
    plates = []
    reg_plate = PaddleOcR_Reg()

    while True:
        ret, frame1 = cap1.read()
        # ret, frame2 = cap2.read()
        
        if not ret:
            break
        frame1 = cv2.resize(frame1, (1920, 1080))
        # frame2 = cv2.resize(frame2, (1920, 1080))
        frame_index += 1
        frame_counter += 1
        if frame_counter < frame_interval:
            continue
        if frame_counter >= frame_interval:
            frame_counter = 0
        t1 = time.time()
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pil_image = Image.fromarray(frame_rgb)
        # modelretina = model_retina()
        # dets_plate = modelretina.detect_plate(
        #     frame, file_img=False, check_sort=False
        # )  # [[x1, y1, x2, y2], box_kps, id, conf]
        dets_plates = model_yolo.detect_pose(
            [frame1, frame1], file_img=False, check_sort=False
        )
        for frame, dets_plate in zip([frame1, frame1], dets_plates):
            if len(dets_plate) == 0:
                # image_name = f"frame_{frame_index}.jpg"
                # cv2.imwrite(os.path.join(output_folder, image_name), frame)
                continue
            for det in dets_plate:
                box = det[0]
                kps = det[1]
                text_plate, crop_img = reg_plate.reg_plate(frame, [det])
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                if text_plate:
                    text_plate[0][3] = crop_img
                    text_plate[0][4] = kps
                    text_plate[0][5] = frame
                    print("*********************************kps*********************************",kps)
                    plates.append(text_plate[0])
                    if len(plates) == 7:
                        text_merLP = mergerLP(plates)
                        plates = []
                        print("*********************************text_merLP*********************************",text_merLP)
                        draw.text((box[0], box[1] - 30), f'{text_merLP["txt"]}', fill=(0, 255, 0), font=font)
                        image_name = "frame_{}_{}.jpg".format(frame_index, text_merLP["txt"])
                    else:
                        draw.text((box[0], box[1] - 30), f'{text_plate[0][1]}', fill=(0, 255, 0), font=font)
                        image_name = "frame_{}_{}.jpg".format(frame_index, text_plate[0][1])
                    # text_merLP = mergerLP(plates)
                    # print("*********************************text_merLP*********************************",text_merLP)
                # if len(plates) == 10:
                #     statistics(plates)
                #     exit()
                # print("text_plate",text_plate)
                # if text_plate:
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                    kp_x1, kp_y1 = kps[0], kps[1]
                    kp_x2, kp_y2 = kps[2], kps[3]
                    kp_x3, kp_y3 = kps[4], kps[5]
                    kp_x4, kp_y4 = kps[6], kps[7]
                    kp_x5, kp_y5 = kps[8], kps[9]
                    points = [(kp_x1, kp_y1), (kp_x2, kp_y2), (kp_x4, kp_y4), (kp_x5, kp_y5)]
                    for point in points:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

                    rect_points = np.array([[kp_x1, kp_y1], [kp_x2, kp_y2], [kp_x5, kp_y5], [kp_x4, kp_y4]], np.int32)
                    rect_points = rect_points.reshape((-1, 1, 2))
                    cv2.polylines(frame, [rect_points], isClosed=True, color=(0, 255, 0), thickness=2)
                    # if text_merLP[0][1] != "34MĐ103269":

                    cv2.imwrite(os.path.join(output_folder, image_name), frame)
                    
                    frame_resized = cv2.resize(frame, (output_video_width//2, output_video_height//2))
                    output_video.write(frame_resized) 
                    cv2.imshow('Video', frame_resized)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            continue
        t2 = time.time()
        print("**********************************************************************")
          

    cap1.release()
    # cap2.release()
    output_video.release()
    cv2.destroyAllWindows()

# from PIL import Image, ImageDraw, ImageFont
# import os
# import numpy as np
# import time
# from test_detect_char import detect_char
# if __name__ == "__main__":
#     pretrained_model_path = "./model_yolo/train_19_4_best/weights/best.pt"
#     model_yolo = model_yolo(pretrained_model_path)

#     input_image_folder = "data_split/test_3"  
#     output_folder = "output_folder_data_split_detect_text/test_3"
#     output_folder_no_plate = "output_folder_no_plate/test_3"  

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     if not os.path.exists(output_folder_no_plate):
#         os.makedirs(output_folder_no_plate)

#     font_path = "font/MQ_License_Plate_font.ttf" 
#     font_size = 40
#     font = ImageFont.truetype(font_path, font_size)

#     count = 0
#     for image_name in os.listdir(input_image_folder):
#         start_time = time.time()
#         image_path = os.path.join(input_image_folder, image_name)
        
#         image = Image.open(image_path)
#         image_np = np.array(image)
        
#         dets_plate = model_yolo.detect_pose(image_np, file_img=False, check_sort=False)
#         print("**************************det_plate**************************",dets_plate)
#         if len(dets_plate[0]) == 0:
#             output_image_path_no_plate = os.path.join(output_folder_no_plate, image_name)
#             image.save(output_image_path_no_plate)
#             count += 1
#             continue

#         draw = ImageDraw.Draw(image)
#         reg_plate = PaddleOcR_Reg()
#         for det in dets_plate:
#             box = det[0]
#             text_plate, crop_img = reg_plate.reg_plate(image_np, [det]) 
#             print("*************************************")
#             print("det_plate:", text_plate)
#             print("*************************************")
#             if text_plate:
#                 draw.text((box[0], box[1] - 30), text_plate[0][1], fill=(0, 255, 0), font=font)
        
#         output_image_path = os.path.join(output_folder, image_name)
#         image.save(output_image_path)

#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"Execution time: {execution_time} seconds")
#         print("Count:", count)
