import torch
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from model.model import IrisLandmarks
from dataset.dataset import SyntheticDataset
from prefetch_generator import BackgroundGenerator
from configs import config
import cv2
from torch.optim import lr_scheduler
from tools.visual_point import lineCrossLine

dataset_pth = config.dataset_pth

workers = config.num_workers
batch_size = config.batch_size
base_lr = config.lr
max_epochs = config.num_epochs
model_dir = config.model_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# DataLoaderX = torch.utils.data_tobedeprecated.DataLoader, accelerate
class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def test(test_loader, model, criterion, epoch):
    model.eval()
    epoch_start_time = time.time()
    losses = []
    gaze_losses, pupil_losses = [], []
    pupil_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            img_data, gaze_x, gaze_y, pupil_x, pupil_y = data

            img_data = img_data.cuda(non_blocking=True)

            gaze_x = torch.from_numpy(np.array(gaze_x)).unsqueeze(1)
            gaze_y = torch.from_numpy(np.array(gaze_y)).unsqueeze(1)

            gaze_data = torch.cat((gaze_x, gaze_y), 1)

            pupil_x = torch.from_numpy(np.array(pupil_x)).unsqueeze(1)
            pupil_y = torch.from_numpy(np.array(pupil_y)).unsqueeze(1)

            pupil_data = torch.cat((pupil_x, pupil_y), 1)
            # print(type(pupil_data[0][0]))

            gaze_data_pred = model(img_data)

            gaze_data_pred = gaze_data_pred.data.cpu()
            # pupil_land_pred = model(img_data)
            #
            # pupil_land_pred = pupil_land_pred.data.cpu()

            # print(type(gaze_data), gaze_data_pred.size(), pupil_land_pred.size())

            # pupil_data_pred = torch.zeros_like(pupil_data)
            # for idx, pupil_lands in enumerate(pupil_land_pred):
            #     # print(pupil_lands.size(), type(pupil_lands))
            #     pupil_data_pred[idx][0] = (pupil_lands[0][0] + pupil_lands[1][0] +
            #                                pupil_lands[2][0] + pupil_lands[3][0]) / 4
            #
            #     pupil_data_pred[idx][1] = (pupil_lands[0][1] + pupil_lands[1][1] +
            #                                pupil_lands[2][1] + pupil_lands[3][1]) / 4
            # print(pupil_data_pred)

            # print(pupil_data_pred.size(), type(pupil_data_pred))
            # print(pupil_data_pred)
            # loss1 = criterion(gaze_data, gaze_data_pred)
            loss2 = criterion(gaze_data_pred, gaze_data)
            if i == 0:
                gaze_pred = gaze_data_pred
            else:
                gaze_pred = torch.cat((gaze_pred, gaze_data_pred))
            loss = loss2

            # print( f"Epoch: {epoch} \t Batch_num: {i + 1} \t Gaze_loss={loss1.data.cpu():.4} \t "
            #        f"Pupil_loss={loss2.data.cpu():.4}")
            #
            losses.append(loss.data.cpu())
            gaze_losses.append(loss2.data.cpu())
            # pupil_losses.append(loss2.data.cpu())

    print(f"Test - Epoch: {epoch} \t AVG_Loss={np.mean(losses):.4} \t AVG_Gaze_loss={np.mean(gaze_losses):.4} \t "
          f"AVG_Pupil_loss={np.mean(pupil_losses):.4} \t Time={time.time() - epoch_start_time:.4} \t")

    return np.mean(losses), gaze_pred.numpy()



def main():
    anno_data = pd.read_csv('./test/real_anno.csv')
    img_pths = np.array(anno_data['img_pth'])
    label_data = anno_data[['gaze_x_degree', 'gaze_y_degree', 'pupil_x_position', 'pupil_y_position']]


    test_dataset = SyntheticDataset(img_pths, label_data)
    test_loader = DataLoaderX(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    device_ids = [0, 1]
    model = torch.nn.DataParallel(IrisLandmarks(), device_ids)
    model.cuda()

    criterion = torch.nn.MSELoss(reduction='mean').cuda()

    # params = []
    # for name, param in model.named_parameters():
    #     # if 'fc' in name:
    #     params.append(param)
    #     print("\t", name)
    #

    saved = torch.load(os.path.join(model_dir, 'checkpoint_gaze_best.pth'))
    print('Loading checkpoint for resnet')
    pre_trained_state_dict = saved
    # load pre_trained_params
    model.load_state_dict(pre_trained_state_dict, strict=False)

    model.eval()

    test_loss, gaze_pred = test(test_loader, model, criterion, 0)
    a = 1
    # print(test_loss)
    # for i in range(len(pupil_center_pred)):
    #     if pupil_center_pred[i][0] > 0 and pupil_center_pred[i][1] > 0:
    #         current_img = cv2.imread(img_pths[i])
    #         current_img = cv2.circle(current_img, (int(640 * pupil_center_pred[i][0]),
    #                                                (int(480 * pupil_center_pred[i][1]))),
    #                                  5, (0, 0, 255), -1)
    #         current_img = cv2.circle(current_img, (int(640 * np.array(label_data['pupil_x_position'])[i]),
    #                                                (int(480 * np.array(label_data['pupil_y_position'])[i]))),
    #                                  5, (255, 0, 0), -1)
    #         cv2.imwrite('./test/real_output/%05d.jpg' % i, current_img)


if __name__ == '__main__':
    main()