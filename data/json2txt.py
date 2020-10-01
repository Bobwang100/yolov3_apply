from pycocotools.coco import COCO
import numpy as np
import tqdm
import argparse
import os

DATA_TYPE = 'train'
if DATA_TYPE == 'train':
    JSON_FILE = './my_data/instances_train2017.json'
    TXT_TYPE = 'train'
    PIC_LIST = [name for name in os.listdir('./my_data/train2017')]
elif DATA_TYPE == 'val':
    JSON_FILE = './my_data/instances_val2017.json'
    TXT_TYPE = 'val'
    PIC_LIST = [name for name in os.listdir('./my_data/val2017')]
else:
    raise ValueError('No type matched!')

def arg_parser():
    parser = argparse.ArgumentParser('code by rbj')
    parser.add_argument('--annotation_path', type=str,
                        default=JSON_FILE )
    #生成的txt文件保存的目录
    parser.add_argument('--save_base_path', type=str, default='./my_data/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    annotation_path = args.annotation_path
    save_base_path = args.save_base_path

    data_source = COCO(annotation_file=annotation_path)
    catIds = data_source.getCatIds()
    categories = data_source.loadCats(catIds)
    categories.sort(key=lambda x: x['id'])
    classes = {}
    coco_labels = {}
    coco_labels_inverse = {}
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)

    img_ids = data_source.getImgIds()

    save_path = save_base_path + TXT_TYPE + '.txt'
    with open(save_path, mode='w') as fp:
        count = -1
        for index, img_id in tqdm.tqdm(enumerate(img_ids), desc='change .json file to .txt file'):

            img_info = data_source.loadImgs(img_id)[0]
            file_name = img_info['file_name'].split('.')[0]
            if file_name + '.jpg' in PIC_LIST:
                print(count, 'pic %s is legal' % (file_name + '.jpg'))
                count += 1
                # if count > 30:
                #     break
                height = img_info['height']
                width = img_info['width']
                annotation_id = data_source.getAnnIds(img_id)
                boxes = np.zeros((0, 5))
                if len(annotation_id) == 0:
                    fp.write('')
                    continue
                annotations = data_source.loadAnns(annotation_id)
                lines = ''
                lines += str(count)
                lines = lines + ' ' + './data/my_data/%s/' % (DATA_TYPE+'2017') + img_info['file_name']
                lines = lines + ' ' + str(width)
                lines = lines + ' ' + str(height)   # width first and then height
                for annotation in annotations:
                    box = annotation['bbox']
                    # some annotations have basically no width / height, skip them
                    if box[2] < 1 or box[3] < 1:
                        continue
                    # top_x,top_y,width,height---->cen_x,cen_y,width,height
                    # box[0] = round((box[0] + box[2] / 2) / width, 6)
                    # box[1] = round((box[1] + box[3] / 2) / height, 6)
                    # box[2] = round(box[2] / width, 6)
                    # box[3] = round(box[3] / height, 6)
                    box[0] = int(box[0])     #  top_x,top_y,width,height----（xmin, ymin, xmax, ymax）
                    box[1] = int(box[1])
                    box[2] = int(box[0] + box[2])
                    box[3] = int(box[1] + box[3])
                    label = coco_labels_inverse[annotation['category_id']]

                    lines = lines + ' ' + str(label)
                    for i in box:
                        lines += ' ' + str(i)
                lines += '\n'
                fp.writelines(lines)
    print('finish')


