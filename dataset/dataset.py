import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
from skimage import data, exposure
from tools.mask2anno import hisEqulColor
import cv2


class SyntheticDataset(Dataset):
    def __init__(self, img_paths, label_data):
        self.img_paths = img_paths
        self.len = len(self.img_paths)

        self.gaze_x = np.array(label_data['gaze_x_degree'])
        self.gaze_y = np.array(label_data['gaze_y_degree'])

        self.pupil_x = np.array(label_data['pupil_x_position'])
        self.pupil_y = np.array(label_data['pupil_y_position'])

        self.img_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):

        img = Image.open(self.img_paths[index]).convert('RGB')

        # self.flag = random.randint(0, 1)
        # if self.flag == 1:
        #     # cv2 2 PIL Image
        #     img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        #
        #     # histogram
        #     img = hisEqulColor(img)
        #
        #     # PIL Image 2 cv2
        #     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return self.img_preprocess(img), self.gaze_x[index], self.gaze_y[index], \
               self.pupil_x[index], self.pupil_y[index]

    def __len__(self):
        return self.len
