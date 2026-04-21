# -*- coding: utf-8 -*-

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'')
    model.predict(source=r'',
                  save=True,
                  show=True,
                  )
