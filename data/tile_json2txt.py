import json
import numpy as np
import os
import xml.etree.cElementTree as ET
# from tqdm import tqdm
import argparse

DATA_TYPE = 'val'
ROOT_DIR = 'tile_data_56'
if DATA_TYPE == 'train':
    FILE_DIR = './%s/train/' % ROOT_DIR
    TXT_TYPE = 'train'

elif DATA_TYPE == 'val':
    FILE_DIR = './%s/val/' % ROOT_DIR
    TXT_TYPE = 'val'
else:
    raise ValueError('No type matched!')

PIC_LIST = []
ALL_FILE_LIST = [name for name in os.listdir(FILE_DIR)]
JSON_LIST = [name for name in ALL_FILE_LIST if name.endswith('.json')]
INVALID_JSON = []
json_count, pic_count = 0, 0
for file in JSON_LIST:
    json_count += 1
    if file.replace('.json', '.jpg') in ALL_FILE_LIST:
        PIC_LIST.append(file.replace('.json', '.jpg'))
        pic_count += 1
    elif file.replace('.json', '.png') in ALL_FILE_LIST:
        PIC_LIST.append(file.replace('.json', '.png'))
        pic_count += 1
    else:
        INVALID_JSON.append(file)
        print("%s miss the related picture!" % file)
for file in INVALID_JSON:
    JSON_LIST.remove(file)
print("the num of json and pic is %s ,%s " % (np.shape(JSON_LIST), np.shape(PIC_LIST)))

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_base_path', type=str, default='./%s/' % ROOT_DIR)
    args = parser.parse_args()
    return args


def conver_json2txt():
    print('pic list shape:', np.shape(PIC_LIST))
    print('json list shape:', np.shape(JSON_LIST))
    args = arg_parser()
    save_path = args.save_base_path + TXT_TYPE + '.txt'
    with open(save_path, mode='w') as fp:
        for idx, pic_file in (enumerate(PIC_LIST)):
            with open(FILE_DIR + pic_file[:-4]+'.json') as f:
                json_dict = json.load(f)
                if np.shape(json_dict['shapes'])[0] > 0:
                    lines = ''
                    lines += str(idx)
                    lines += ' ' + './data/%s/%s/' % (ROOT_DIR, DATA_TYPE) + pic_file
                    lines += ' ' + str(json_dict['imageWidth']) + ' ' + str(json_dict['imageHeight'])
                    # print(pic_file, json_dict, np.shape(json_dict['shapes'])[0])
                    for i in range(np.shape(json_dict['shapes'])[0]):
                        x_min = np.min([int(json_dict['shapes'][i]['points'][0][0]),
                                        int(json_dict['shapes'][i]['points'][1][0])])
                        y_min = np.min([int(json_dict['shapes'][i]['points'][0][1]),
                                        int(json_dict['shapes'][i]['points'][1][1])])
                        x_max = np.max([int(json_dict['shapes'][i]['points'][0][0]),
                                        int(json_dict['shapes'][i]['points'][1][0])])
                        y_max = np.max([int(json_dict['shapes'][i]['points'][0][1]),
                                        int(json_dict['shapes'][i]['points'][1][1])])
                        # lines += ' ' + str(json_dict['shapes'][i]['label'])
                        lines += ' ' + str(0)
                        lines += ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max)
                    lines += '\n'
                    fp.writelines(lines)
                else:
                    print('%s have no true box' % pic_file)
        print('finish')


if __name__ == "__main__":
    conver_json2txt()
