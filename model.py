import cv2
import torch
import numpy as np
from munkres import Munkres, print_matrix, DISALLOWED

from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.utils import postprocess

import sys
sys.path.append('./reid')
from torchreid.utils import FeatureExtractor

video_path = 'demo_2.mp4'
output_folder = './output/'
detect_exp_file = "./YOLOX/exps/default/yolox_s.py"
ckpt = "./YOLOX/yolox_s.pth"
save_folder = "./result/"

confidence = 0.82
iou_threshold = 0.1
distance_threshold = 20
iou_threshold = 0
edge_threshold = 0.5
frame_step = 2

extractor = FeatureExtractor(
    model_name='mobilenetv2_x1_4',
    model_path='./reid/mobilenetv2_1.4-bc1cc36b.pth',
    device='cuda'
)

def load_yolox_model(exp_file, ckpt):
    exp = get_exp(exp_file)
    exp.test_conf = 0.25
    exp.nmsthre = 0.45
    exp.test_size = (640, 640)

    model = exp.get_model()
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model.eval()
    return model, exp

# YOLOX inference
def detect_inference(model, exp, img):
    img_info = {"id": 0}
    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img
    
    ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
    img_info["ratio"] = ratio

    preproc = ValTransform()
    img, _ = preproc(img, None, exp.test_size)
    img = torch.from_numpy(img).unsqueeze(0).float()

    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=True)

    # Filter out bounding boxes for class 0
    mask = (outputs[0][:, 6] == 0) & (outputs[0][:, 4] > confidence)
    outputs[0] = outputs[0][mask]

    return outputs[0], img_info

def crop_bbox(frame, bbox, info):
    ratio = info["ratio"]
    box = bbox[:4]
    box = box / ratio

    x1, y1, x2, y2 = map(int, box)
    bbox_image = frame[y1:y2, x1:x2, :]
    return bbox_image

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.333, 1.000, 0.000,
        0.300, 0.300, 0.300,
        0.000, 0.000, 1.000,
        1.000, 0.000, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.600, 0.600, 0.600,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        1.000, 0.500, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def visualize_results(img, results, img_info, class_names, save_path, video):
    i = 0
    for result in results:
        box = result[:4]
        score = result[4]
        #class_id = int(result[5])
        
        ratio = img_info["ratio"]
        box = box / ratio
        
        #i += 1
        if score > 0.5:
            x0, y0, x1, y1 = map(int, box)
            color = (_COLORS[result[-1].int()] * 255).astype(np.uint8).tolist()
            #label = f"{class_names[class_id]}: {score:.2f}"
            
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            #cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #cv2.imwrite(save_path, img)
    # add the image to video
    video.write(img)
    #print(f"Visualized results saved to {save_path}")

def calculate_iou(box1, box2):
    # box1, box2: (x1, y1, x2, y2)
    # intersection coordinate
    intersection_x1 = max(box1[0], box2[0])
    intersection_y1 = max(box1[1], box2[1])
    intersection_x2 = min(box1[2], box2[2])
    intersection_y2 = min(box1[3], box2[3])

    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    # bounding box area
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection_area / float(area_box1 + area_box2 - intersection_area)
    
    return iou

# calculate the iou of two bboxes matrix
def calculate_iou_matrix(boxes1, boxes2):
    iou_matrix = torch.zeros((len(boxes1), len(boxes2)))

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i, j] = calculate_iou(box1, box2)

    return iou_matrix

# calculate the center point distance of two bboxes matrix
def calculate_center_distance(boxes1, boxes2):
    # center point
    center1 = torch.stack([(boxes1[:, 0] + boxes1[:, 2]) / 2,
                           (boxes1[:, 1] + boxes1[:, 3]) / 2],
                           dim=1)

    center2 = torch.stack([(boxes2[:, 0] + boxes2[:, 2]) / 2,
                           (boxes2[:, 1] + boxes2[:, 3]) / 2],
                           dim=1)

    distance_matrix = torch.norm(center1[:, None, :] - center2[None, :, :], dim=2)

    return distance_matrix

# using min center_distance to find unmatched one
def find_new_indices(center_distance):
    # find min of each row
    min_values, min_indices = torch.min(center_distance, dim=1)

    # mask[i][j] = True, which is min
    min_mask = torch.zeros_like(center_distance, dtype=torch.bool)
    min_mask[torch.arange(center_distance.size(0)), min_indices] = True

    unmatched_idx = torch.nonzero(~min_mask.any(dim=0)).squeeze(dim=1)

    return unmatched_idx


def main():
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    # load the yolox model
    model, exp = load_yolox_model(detect_exp_file, ckpt)
    # Hungarian Algorithm
    m = Munkres()

    num_people = 0
    video = []
    
    # each frame
    for idx in range(0, frame_count, frame_step):
        vc.set(1, idx)
        ret, frame = vc.read()
        
        height, width, layers = frame.shape
        size = (width, height)

        if ret:
            # detect the people in each frame
            detection, info = detect_inference(model, exp, frame)
            
            # first frame
            if idx == 0:
                bboxes_all = detection      # store the last detection of all people appeared
                num_people = len(detection)

                # add id to each bbox
                box_ids = torch.arange(len(bboxes_all)).unsqueeze(1)
                bboxes_all = torch.cat((bboxes_all, box_ids.float()), dim=1)

                features_all = []   # store the feature of all people appeared
                for i in range(detection.shape[0]):
                    bbox = detection[i]
                    bbox_image = crop_bbox(frame, bbox, info)

                    # calculate Re-ID feature of each crop
                    feature = extractor(bbox_image)
                    features_all.append(feature)
                # turn the features_all to tensor stack
                features_all_tensor = torch.cat(features_all).float()

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                save_video_path = save_folder + 'result_' + video_path
                video = cv2.VideoWriter(save_video_path, fourcc, fps / frame_step, size)

                save_path = '{}{:d}.jpg'.format(save_folder, idx)
                visualize_results(frame, bboxes_all, info, COCO_CLASSES, save_path, video)

            # not first frame
            else:
                bboxes_n = detection      # store the detection of current frame
                # add id to each bbox
                box_ids = torch.arange(len(bboxes_n)).unsqueeze(1)
                bboxes_n = torch.cat((bboxes_n, box_ids.float()), dim=1)
                
                features_n = []   # store the feature of current frame
                for i in range(detection.shape[0]):
                    bbox = detection[i]
                    bbox_image = crop_bbox(frame, bbox, info)

                    # calculate Re-ID feature of each crop
                    feature = extractor(bbox_image)
                    features_n.append(feature)

                #center_distance = calculate_center_distance(bboxes_all, bboxes_n)
                if len(bboxes_all) < len(bboxes_n):
                #if center_distance.shape[0] < center_distance.shape[1]:
                    # if there's no min distance in j column, then j is new person
                    unmatched_indices = find_new_indices(center_distance)
                    for j in unmatched_indices:
                        bboxes_all = torch.cat([bboxes_all, bboxes_n[j].unsqueeze(0)], dim=0)
                        features_all_tensor = torch.cat([features_all_tensor, features_n[j]], dim=0)
                        num_people += 1

                center_distance = calculate_center_distance(bboxes_all, bboxes_n)

                for j in range(center_distance.shape[1]):
                    # detect new person from distance_threshold
                    # if the box is on the edge of frame
                    if torch.all(center_distance[:, j] > distance_threshold):
                        ratio = info["ratio"]
                        box = bboxes_n[j][:4] / ratio
                        if (box[0] < edge_threshold or box[1] < edge_threshold or box[2] > 1280-edge_threshold or box[3] > 720-edge_threshold):
                            bboxes_all = torch.cat([bboxes_all, bboxes_n[j].unsqueeze(0)], dim=0)
                            features_all_tensor = torch.cat([features_all_tensor, features_n[j]], dim=0)
                            num_people += 1

                    # if the box is in the middle of frame
                    elif torch.all(center_distance[:, j] > distance_threshold * 3):
                        bboxes_all = torch.cat([bboxes_all, bboxes_n[j].unsqueeze(0)], dim=0)
                        features_all_tensor = torch.cat([features_all_tensor, features_n[j]], dim=0)
                        num_people += 1

                features_n_tensor = torch.cat(features_n).float()
                # calculate the distances of all bboxes of two frames
                distances = torch.cdist(features_all_tensor, features_n_tensor)
                iou_matrix = calculate_iou_matrix(bboxes_all, bboxes_n)
                center_distance = calculate_center_distance(bboxes_all, bboxes_n)

                # if there's two nonzero iou in j column, test with iou_threshold
                check_zero = torch.nonzero((iou_matrix != 0).sum(dim=0) >= 2).squeeze(dim=1)
                for j in check_zero:
                    for i in range(center_distance.shape[0]):
                        if iou_matrix[i][j] <= iou_threshold:
                            distances[i][j] = 1000

                for i in range(center_distance.shape[0]):
                    for j in range(center_distance.shape[1]):
                        # match the bboxes only when center distance <= distance_threshold
                        if center_distance[i][j] > distance_threshold * 3:
                            distances[i][j] = 1000
                        else:
                            #distances[i][j] = distances[i][j] + (center_distance[i][j] / 200)
                            distances[i][j] = distances[i][j] - iou_matrix[i][j]
 
                # turn to 2D matrix
                distances = distances.cpu().tolist()
                # Hungarian Algorithm to match bboxes
                match_indexes = m.compute(distances)    # [(row, column), (row, column)]
               
                for row, column in match_indexes:
                    # modify the id of matched bbox
                    bboxes_n[column][-1] = row
                    # record the latest bbox of matched person
                    bboxes_all[row] = bboxes_n[column]
                    features_all_tensor[row] = features_n[column]

                save_path = '{}{:d}.jpg'.format(save_folder, idx)
                visualize_results(frame, bboxes_n, info, COCO_CLASSES, save_path, video)
        
        print("\rTracking Process: {}/{}".format(idx+1 , frame_count), end = '')

    print(f" Visualized results saved to {save_video_path}")
    print('count: ', num_people)

    vc.release()
    video.release()

if __name__ == "__main__":
    main()
