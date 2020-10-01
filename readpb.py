import tensorflow as tf
import os
import cv2
from tensorflow.python.platform import gfile

import numpy as np

PIC_RESIZE = (416, 416)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh


def load_img(path):
    img = cv2.imread(path)
    img, resize_ratio, dw, dh = letterbox_resize(img, 416, 416, interp=0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    ret_img = img[np.newaxis, :] / 255.
    boxes_, scores_, labels_ = sess.run([op, op1, op2], feed_dict={input_x: ret_img})
    boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
    boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    print('boxes :', boxes_, '\n', 'scores: ', scores_, '\n', 'labels:', labels_)
    # resized_img = cv2.resize(img, PIC_RESIZE)
    # norm_img = resized_img / 255.0
    # ret_img = norm_img[None, :, :, :]
    # ret_img = norm_img[None, :, :, None]
    # resized_img = skimage.transform.resize(img, (224, 224))[None, :, :, None]        # [1,224,224,3]
    return ret_img


def load_test_data4pb():  # 2 kinds
    test_data_files = []
    imgs = {'qual': [], 'unqual': []}  #
    for k in imgs.keys():
        dir_ = '/media/xn/AA1A74301A73F821/wbw/NGtile/data/' + 'testpb/' + k
        for i, file in enumerate(os.listdir(dir_)):
            if "_" in file:
                print("ATTENTIN ", file, 'passed')
                continue
            try:
                # print('the %d pic is reading' % i, 'pic', file)
                print(file)
                resized_img = load_img(os.path.join(dir_, file))
            except OSError:
                continue
            imgs[k].append(resized_img)
            # if not file in test_data_files:
            test_data_files.append(file)
            # print('the %d pic already readed' % i, 'pic', file, 'test file length is %s' % (len(test_data_files)))
            # if len(imgs[k]) == 5:
            #     break
        # print('the Num IS ', len(imgs[k]))
    aa, bb = [1, 0], [0, 1]
    len_qual, len_unqual = len(imgs['qual']), len(imgs['unqual'])
    qual_y = np.array(len_qual * aa).reshape(len_qual, 2)
    unqual_y = np.array(len_unqual * bb).reshape(len_unqual, 2)
    return imgs['qual'], imgs['unqual'], qual_y, unqual_y, test_data_files


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pb_file_path = './pb/'  # 获取当前代码路径
sess = tf.Session()
with gfile.FastGFile(pb_file_path + 'yolo.pb', 'rb') as f:  # 加载模型
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())
# 需要先复原变量
# print(sess.run('b:0'))
# 1
# 下面三句，是能否复现模型的关键
# 输入

input_x = sess.graph.get_tensor_by_name('input_data:0')  #此处的x一定要和之前保存时输入的名称一致！
# input_x2 = sess.graph.get_tensor_by_name('phase_train:0')  #此处的x一定要和之前保存时输入的名称一致！
# input_y = sess.graph.get_tensor_by_name('Placeholder:1')  #此处的y一定要和之前保存时输入的名称一致！
# op = sess.graph.get_tensor_by_name('resnet_v2_152/logits/BiasAdd:0')  # 此处的op_to_store一定要和之前保存时输出的名称一致！
op = sess.graph.get_tensor_by_name('concat_10:0')  #此处的op_to_store一定要和之前保存时输出的名称一致！
op1 = sess.graph.get_tensor_by_name('concat_11:0')  #此处的op_to_store一定要和之前保存时输出的名称一致！
op2 = sess.graph.get_tensor_by_name('concat_12:0')  #此处的op_to_store一定要和之前保存时输出的名称一致！
# op = sess.graph.get_tensor_by_name('vgg_19/fc8/squeezed:0')  #此处的op_to_store一定要和之前保存时输出的名称一致！
qual_xt, unqual_xt, qual_yt, unqual_yt, test_files = load_test_data4pb()


# err_count = 0
# valid_count = 0
# for i in range(len(xst)):
#     boxes_, scores_, labels_ = sess.run([op, op1, op2], feed_dict={input_x: xst[i]})
#     boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
#     boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
#     # ret_S = sess.run(tf.nn.softmax(op), feed_dict={input_x: xst[i][np.newaxis, :]})
#     # ret_S = [np.exp(ret[0][0]) / (np.exp(ret[0][0]) + np.exp(ret[0][1])),
#     #          np.exp(ret[0][1]) / (np.exp(ret[0][0]) + np.exp(ret[0][1]))]
#     print('pic:', test_files[i], '\n', 'boxes :', boxes_, '\n', 'scores: ', scores_, '\n', 'labels:', labels_)
#     # print('pic:', test_files[i], 'cal_result :', ret, 'soft:', ret_S)
#     # if ret_S[0] > 0.5:
#     #     valid_count += 1
#     #     print('valid_count', valid_count)
#     # if not np.argmax(ret) == np.argmax(yst[i]):
#     #     err_count += 1
#     # print('the error count is %d' % err_count)
#     # print(i, test_files[i], ret)
