import copy
import json

import numpy as np 
from PIL import Image

from utils import check_results, display_results


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    - [xmin, ymin, xmax, ymax]
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union, [xmin, ymin, xmax, ymax]


def hflip(img, bboxes):
    """
    horizontal flip of an image and annotations
    args:
    - img [PIL.Image]: original image
    - bboxes [list[list]]: list of bounding boxes
    return:
    - flipped_img [PIL.Image]: horizontally flipped image
    - flipped_bboxes [list[list]]: horizontally flipped bboxes
    """
    # IMPLEMENT THIS FUNCTION
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    bboxes_cp = np.array(bboxes[:])
    flipped_bboxes = np.array(bboxes[:])
    flipped_bboxes[:,1] = flipped_img.width - bboxes_cp[:,3]
    flipped_bboxes[:,3] = flipped_img.width - bboxes_cp[:,1]
    
    return flipped_img, flipped_bboxes


def resize(img, boxes, size):
    """
    resized image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - size [array]: 1x2 array [width, height]
    returns:
    - resized_img [PIL.Image]: resized image
    - resized_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    resized_image = img.resize(size)
    
    w_r = resized_image.width / img.width
    h_r = resized_image.height / img.height

    resized_boxes = np.array(boxes[:])
    resized_boxes[:,[0,2]] = h_r * resized_boxes[:,[0,2]]
    resized_boxes[:,[1,3]] = w_r * resized_boxes[:,[1,3]]

    return resized_image, resized_boxes


def random_crop(img, boxes, classes, crop_size, min_area=100):
    """
    random cropping of an image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - crop_size [array]: 1x2 array [width, height]
    - min_area [int]: min area of a bbox to be kept in the crop
    returns:
    - cropped_img [PIL.Image]: resized image
    - cropped_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    w, h = img.size
    x1 = np.random.randint(0, w - crop_size[0])
    x2 = x1 + crop_size[0]
    y1 = np.random.randint(0, h - crop_size[1])
    y2 = y1 + crop_size[1]

    cropped_image = img.crop((x1, y1, x2, y2))

    cropped_boxes = []
    cropped_classes = []
    for bb, cl in zip(boxes, classes):
        iou, intersection = calculate_iou(bb, [y1, x1, y2, x2])
        if iou > 0:
            area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
            if area > min_area:
                cropped_boxes.append(
                    [
                        intersection[0] - y1,
                        intersection[1] - x1,
                        intersection[2] - y1,
                        intersection[3] - x1,
                    ]
                )
                cropped_classes.append(cl)

    return cropped_image, cropped_boxes, cropped_classes


if __name__ == "__main__":
    # fix seed to check results
    np.random.seed(48)
    
    # open annotations
    with open("./data/ground_truth.json") as f:
        ground_truth = json.load(f)
    
    # filter annotations and open image
    filename = 'segment-12208410199966712301_4480_000_4500_000_with_camera_labels_79.png'
    gt_boxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_classes = [g['classes'] for g in ground_truth if g['filename'] == filename][0]
    img = Image.open(f'./data/images/{filename}')

    # check horizontal flip, resize and random crop
    # use check_results defined in utils.py for this
    # check horizontal flip
    flipped_img, flipped_bboxes = hflip(img, gt_boxes)
    display_results(img, gt_boxes, flipped_img, flipped_bboxes)
    check_results(flipped_img, flipped_bboxes, aug_type='hflip')

    # check resize
    resized_image, resized_boxes = resize(img, gt_boxes, size=[640, 640])
    display_results(img, gt_boxes, resized_image, resized_boxes)
    check_results(resized_image, resized_boxes, aug_type='resize')

    # check random crop
    cropped_image, cropped_boxes, cropped_classes = random_crop(img, gt_boxes, gt_classes, [512, 512], min_area=100)
    display_results(img, gt_boxes, cropped_image, cropped_boxes)
    check_results(cropped_image, cropped_boxes, aug_type='random_crop', classes=cropped_classes)