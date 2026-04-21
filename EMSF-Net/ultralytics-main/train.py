# -*- coding: utf-8 -*
import torch
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(model=r"D:\A_Good_study\A_Future\No.1_EMSF-Net\Project_Open_Gitub\EMSF-Net\ultralytics-main\ultralytics\cfg\models\11\yolo11s_MSF-Backbone_add_P2_DWConv_C2D_MSFBlock_LDetect .yaml")
    model.train(data=r'',
                epochs=200,
                batch=8,
                optimizer='SGD',
                workers=8,
                device='',
                close_mosaic=10,
                resume=True,
                project=r'',
                name='',
                box=6,  # (float 7) box loss gain
                cls=1,  # (float 0.5) cls loss gain (scale with pixels)
                dfl=2,  # (float 1.5) dfl loss gain
                single_cls=False,
                cache=False,
                amp=True,
                mosaic=0.2,
                mixup=0.1,
                )