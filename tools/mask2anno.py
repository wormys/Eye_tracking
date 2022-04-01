import os
import cv2
import json
import numpy as np
import pandas as pd

from configs import config

dataset_pth = config.dataset_pth
cameras = config.cameras
real_dataset_pth = config.real_dataset_pth


def mask2anno(img_path):
    image = cv2.imread(img_path)
    color = [
        ([35, 43, 46], [99, 255, 255])  # 青色
    ]
    #
    # (lower, upper) = color[0]
    for (lower, upper) in color:
        # 创建NumPy数组
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到对应颜色
        mask = cv2.inRange(image, lower, upper)

    mask_output = cv2.bitwise_and(image, image, mask=mask)

    imgray = cv2.cvtColor(mask_output, cv2.COLOR_BGR2GRAY)  # 转灰度图
    _, thresh = cv2.threshold(imgray, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓查找

    pupil_x_position, pupil_y_position, x_axes, y_axes, angle = 0, 0, 0, 0, 0
    if len(contours) != 0 and contours[0].shape[0] >= 5:
        # [0] center, [1] axes, [2] angle
        retval = cv2.fitEllipse(contours[0])  # 拟合椭圆
        ((pupil_x_position, pupil_y_position), (x_axes, y_axes), angle) = retval
        # ellipsis_img = cv2.ellipse(image, retval, (255, 255, 255), thickness=2)  # 画椭圆
        # cv2.imshow("mark_ellipse[1]", ellipsis_img)

        # show
        # cv2.imwrite("../test/output.jpg", ellipsis_img)
        # cv2.waitKey(0)

    return pupil_x_position, pupil_y_position, x_axes, y_axes, angle


def json2csv(json_dir, gaze_data_dir, csv_save_dir):
    col_name = ['img_pth', 'pupil_x_position', 'pupil_y_position', 'gaze_x_degree', 'gaze_y_degree',
                'x_axes', 'y_axes', 'angle']
    preprocess_anno_data = pd.DataFrame(columns=col_name)
    gaze_data = pd.read_csv(gaze_data_dir)
    gaze_data_x = np.array(gaze_data['gaze_x'])
    # gaze_data_x = [gaze_data_x[i] / 2 + 0.5 for i in range(len(gaze_data_x))]
    gaze_data_y = np.array(gaze_data['gaze_y'])
    # gaze_data_y = [gaze_data_y[i] / 2 + 0.5 for i in range(len(gaze_data_y))]

    with open(json_dir) as f:
        anno_data = json.load(f)
    for img_name, anno in anno_data.items():
        img_name = int(img_name.split('.')[0])
        img_pth = os.path.join(real_dataset_pth, 'dataset', '%06d.jpg' % img_name)
        pupil_x_position, pupil_y_position = anno['Pupil Center']
        if pupil_x_position != -1 and pupil_y_position != -1:
            preprocess_anno_data.loc[len(preprocess_anno_data)] = [img_pth, pupil_x_position,
                                                                   pupil_y_position, 0, 0, 0, 0, 0]
    preprocess_anno_data['gaze_x_degree'] = gaze_data_x
    preprocess_anno_data['gaze_y_degree'] = gaze_data_y
    preprocess_anno_data.to_csv(csv_save_dir)




def main():
    # anno_pth = os.path.join(dataset_pth, 'nvgaze_eye_sample.csv')
    # anno_data = pd.read_csv(anno_pth)
    #
    # mask_img_paths = anno_data['regionmask_withoutskin_L_0']
    #
    # col_name = ['img_pth', 'pupil_x_position', 'pupil_y_position', 'gaze_x_degree', 'gaze_y_degree',
    #             'x_axes', 'y_axes', 'angle']
    # preprocess_anno_data = pd.DataFrame(columns=col_name)
    # for idx, mask_img_path in enumerate(mask_img_paths):
    #     mask_img_id = mask_img_path.split('.')[0]
    #     for i in cameras:
    #         current_img_path = os.path.join(dataset_pth, 'dataset', mask_img_id + '_%03d.png' % i)
    #         pupil_x_position, pupil_y_position, x_axes, y_axes, angle = mask2anno(current_img_path)
    #         origin_img_path = current_img_path.replace('maskWithoutSkin', 'img')
    #         gaze_x_degree, gaze_y_degree = anno_data.iloc[idx][['float_gaze_x_degree', 'float_gaze_y_degree']]
    #         gaze_x_degree = (gaze_x_degree + 90) / 180
    #         gaze_y_degree = (gaze_y_degree + 90) / 180
    #         if pupil_x_position != 0 and pupil_y_position != 0:
    #
    #             preprocess_anno_data.loc[len(preprocess_anno_data)] = [origin_img_path, pupil_x_position / 360,
    #                                                                    pupil_y_position / 240, gaze_x_degree,
    #                                                                    gaze_y_degree, x_axes, y_axes, angle]
    #         print('%s finish!' % origin_img_path)
    #
    # preprocess_anno_data.to_csv('../test/anno.csv')

    json_dir = os.path.join(real_dataset_pth, 'anno.json')
    gaze_data_dir = os.path.join(real_dataset_pth, 'gaze_data.csv')
    csv_save_dir = '../test/real_anno.csv'
    json2csv(json_dir, gaze_data_dir, csv_save_dir)
    pass


# 彩色图像进行自适应直方图均衡化
def hisEqulColor(img):
    ## 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 以下代码详细注释见官网：
    # https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


if __name__ == '__main__':
    main()