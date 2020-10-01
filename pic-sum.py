# import cv2
# import numpy as np


# import os
#
# file_unqual_list = [file for file in os.listdir('./test_tile_result/or-qual/or-unqual')]
# bad_unqual_list = [file for file in os.listdir('./test_tile_result/or-qual/bad-orunqual')]
#
# print(file_unqual_list, '\n', bad_unqual_list)
# unqual_sum = 0
# bad_unqual_sum = 0
#
# for file in os.listdir('./test_tile_result/or-qual/dp-qual'):
#     if file in file_unqual_list:
#         unqual_sum += 1
#         print('unqual:', file)
#     elif file in bad_unqual_list:
#         bad_unqual_sum += 1
#         print('bad:', file)
# print('unqual_sum', unqual_sum)
# print('bad_unqual_sum', bad_unqual_sum)


# def voc_ap(rec, prec, use_07_metric=False):
#     """Compute VOC AP given precision and recall. If use_07_metric is true, uses
#     the VOC 07 11-point method (default:False).
#     """
#     if use_07_metric:
#         # 11 point metric
#         ap = 0.
#         for t in np.arange(0., 1.1, 0.1):
#             if np.sum(rec >= t) == 0:
#                 p = 0
#             else:
#                 p = np.max(prec[rec >= t])
#             ap = ap + p / 11.
#     else:
#         # correct AP calculation
#         # first append sentinel values at the end
#         mrec = np.concatenate(([0.], rec, [1.]))
#         mpre = np.concatenate(([0.], prec, [0.]))
#
#         # compute the precision envelope
#         for i in range(mpre.size - 1, 0, -1):
#             mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
#
#         # to calculate area under PR curve, look for points
#         # where X axis (recall) changes value
#         i = np.where(mrec[1:] != mrec[:-1])[0]
#
#         # and sum (\Delta recall) * prec
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#     return ap
#
#
# rec = np.array([0.1, 0.2, 0.3, 0.3, 0.5])
# pre = np.array([0.5, 0.4, 0.3, 0.3, 0.1])
# voc_ap(rec, pre, True)
# voc_ap(rec, pre, False)
