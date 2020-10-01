import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import k_means

TRAIN_DIR = './tile_data_56/train/'
VAL_DIR = './tile_data_56/val/'
TRAIN_TXT_DIR = './tile_data_n7/train.txt'


def name_clean():
    for filename in tqdm(os.listdir(TRAIN_DIR)):
        new_name = filename.strip().replace(' ', '')
        new_name = 'TR' + new_name
        os.rename(TRAIN_DIR + filename, TRAIN_DIR + new_name)

    for filename in tqdm(os.listdir(VAL_DIR)):
        new_name = filename.strip().replace(' ', '')
        new_name = 'VA' + new_name
        os.rename(VAL_DIR + filename, VAL_DIR + new_name)


def anchor_cluster(k=9):
    all_boxes_wh = []
    for line in open(TRAIN_TXT_DIR).readlines():
        boxes = parse_line(line)
        all_boxes_wh.extend(boxes)
    boxes_cluster, labels_, inertia_ = k_means(all_boxes_wh, n_clusters=k, )
    boxes_cluster = boxes_cluster[np.array([box[0] * box[1] for box in boxes_cluster]).argsort()]  # sorted by area
    return boxes_cluster


def parse_line(line):
    """
    Given a line from the training/test txt file, return parsed info.
    line format: line_index, img_path, img_width, img_height, [box_info_1 (5 number)], ...
    return:
        line_idx: int64
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
        labels: shape [N]. class index.
        img_width: int.
        img_height: int
    """
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    s = s[4:]
    assert len(
        s) % 5 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_max - x_min, y_max - y_min])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    return boxes


def change_anchor_txt(boxes):
    with open('./yolo_anchors.txt', 'r') as f:
        line = f.readlines()
        del line
        new_line = []
        for box in boxes:
            new_line.append([int(box[0]), int(box[1])])
        strl = ''
        for i in range(boxes.shape[0]):
            strl = strl + str(int(boxes[i][0])) + ',' + str(int(boxes[i][1])) + ',' + ' '
        with open('./yolo_anchors.txt', 'w') as f:
            f.write(strl)
        print(strl)


if __name__ == "__main__":
    name_clean()
    # boxes_clustered = anchor_cluster()
    # # print(boxes_clustered)
    # change_anchor_txt(boxes_clustered)
