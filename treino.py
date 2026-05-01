import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import os
import time
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
import ultralytics
print(f"Versao do ultralytics: {ultralytics.__version__}")

yolo_custom = None


def print_metricas(model, particao):
    if particao not in ('val', 'test'):
        print('particao invalida')
        return

    metricas = model.val(data=path_config_yaml, verbose=False, split=particao)
    print(f'\n{"-" * 60}')
    print('Metricas particao', particao)
    print('-' * 60)
    print(f'  mAP@0.50         : {metricas.box.map50:0.3f}')
    print(f'  mAP@0.50:0.95    : {metricas.box.map:0.3f}')
    print(f'  Precisao media   : {metricas.box.mp:0.3f}')
    print(f'  Recall medio     : {metricas.box.mr:0.3f}')
    print('-' * 60)
    for i, cls_idx in enumerate(metricas.box.ap_class_index):
        nome = model.names[cls_idx]
        ap   = metricas.box.ap50[i]
        print(f'  {int(cls_idx)} {nome:<15} AP@0.5 = {ap:.4f}')


if __name__ == '__main__':
    # instancia modelo inicial
    #yolo_custom = YOLO("yolov8n.pt")
    yolo_custom = YOLO("yolo26n.pt")

    # dados identificacao do projeto/modelo
    projeto = "transfer_yolo26_ep30"
    nome_modelo = "yolo26_transfer"
    path_config_yaml = 'config.yaml'

    # treino
    results_treino = yolo_custom.train(
        data=path_config_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        device='cuda',
        project=projeto,
        name=nome_modelo,
        exist_ok=True,
        patience=15,
        plots=True,
        amp=True,
        verbose=False,
    )

    best_model_path = str(results_treino.save_dir / 'weights' / 'best.pt')
    yolo_best = YOLO(best_model_path)
