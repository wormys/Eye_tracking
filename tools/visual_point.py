import torch
import torch.nn as nn
from model.model import IrisLandmarks
import matplotlib.pyplot as plt
import cv2
# import imgaug.augmenters as iaa
import numpy as np
import json
# import geomeas as gm
from tensorboardX import SummaryWriter


def pointAndPointToLine(pt0, pt1):  # 由两点得直线的标准方程 ax+by=c
    x0, y0 = pt0
    x1, y1 = pt1
    return (y1 - y0, x0 - x1, x0 * y1 - y0 * x1)


def lineCrossLine(p1, p2, q1, q2):  # 求两条直线交点

    a0, b0, c0 = pointAndPointToLine(p1, p2)
    a1, b1, c1 = pointAndPointToLine(q1, q2)
    dd = a0 * b1 - a1 * b0
    if abs(dd) < 1e-6: return None
    return ((c0 * b1 - c1 * b0) / dd, (a0 * c1 - a1 * c0) / dd)


# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)
# print("cuDNN version:", torch.backends.cudnn.version())

# gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# net = IrisLandmarks().to(gpu)
#
# net.load_weights("./my_model.pth")
# import os, sys
#
# # 打开文件
# path = "./test_image"
# dirs = os.listdir(path)
# dirs.sort(key=lambda x: int(x[:-4]))
#
# # 输出所有文件和文件夹
# for file in dirs:
#     img = cv2.imread(path + '/' + file)
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     img = cv2.resize(img, (64, 64))
#
#     eye, iris, pupil = net.predict_on_image(img)
#
#     a, b = lineCrossLine((pupil[0][0][0], pupil[0][0][1]), (pupil[0][2][0], pupil[0][2][1]),
#                          (pupil[0][1][0], pupil[0][1][1]), (pupil[0][3][0], pupil[0][3][1]))
#
#     data = {str(file): {"Pupil Center": [float(a), float(b)], "Up Eyelid": [float(eye[0][2][0]), float(eye[0][2][1])],
#                         "Down Eyelid": [float(eye[0][3][0]), float(eye[0][3][1])],
#                         "Left Eyelid": [float(eye[0][0][0]), float(eye[0][0][1])],
#                         "Right Eyelid": [float(eye[0][1][0]), float(eye[0][1][1])],
#                         "Up Pupil": [float(pupil[0][0][0]), float(pupil[0][0][1])],
#                         "Down Pupil": [float(pupil[0][2][0]), float(pupil[0][2][1])],
#                         "Left Pupil": [float(pupil[0][1][0]), float(pupil[0][1][1])],
#                         "Right Pupil": [float(pupil[0][3][0]), float(pupil[0][3][1])]}}
#
#     with open("record.json", "a") as f:
#         json.dump(data, f)
#         f.write('\n')
#
#     plt.imshow(img, zorder=1)
#     x, y = eye[:, :, 0], eye[:, :, 1]
#     plt.scatter(x, y, zorder=2, s=1.0, c='cyan')
#
#     x, y = pupil[:, :, 0], pupil[:, :, 1]
#     plt.scatter(x, y, zorder=2, s=1.0, c='r')
