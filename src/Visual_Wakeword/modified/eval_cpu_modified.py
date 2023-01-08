# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import random
from PIL import Image
import numpy as np
from importlib import import_module
from thop import profile, clever_format

device = torch.device('cpu')

MODIFIED_MODELS_DIR = 'modified_models'

# arg parser
parser = argparse.ArgumentParser(description='PyTorch VisualWakeWords evaluation')
parser.add_argument('--weights', default=None, type=str, help='load from checkpoint')
parser.add_argument('--image_folder', default=None, type=str, help='folder containing images')
parser.add_argument('--model_arch',
                    default='mobilenet_gru_front', type=str,
                    choices=['mobilenet_gru_front', 'mobilenet_lstm_front',
                             'mobilenet_fg_last', 'mobilenet_gru_last', 'mobilenet_lstm_last',
                             'mobilenet_fg_fl', 'mobilenet_gru_fl', 'mobilenet_lstm_fl'],
                    help='choose architecture among rpool variants')


if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Model: {args.model_arch}')
    print()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]) 

    module = import_module(f'{MODIFIED_MODELS_DIR}.{args.model_arch}')
    model = module.mobilenetv2_rnnpool(num_classes=2, width_mult=0.35, last_channel=320)
    model = model.to(device)

    # Because model is trained on 4 GPUs (default) with DataParallel, here DataParallel is necessary,
    # or model.load_state_dict will raise error "Unexpected key(s) in state_dict"
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.weights, map_location=device)
    checkpoint_dict = checkpoint['model']
    model_dict = model.state_dict()
    model_dict.update(checkpoint_dict) 
    model.load_state_dict(model_dict)

    model.eval()

    img_path = args.image_folder
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]

    # statistic
    rand_input = torch.randn(1, 3, 224, 224)
    macs, params = clever_format(profile(model, inputs=(rand_input, )), "%.3f")
    print(f'\nMAdds: {macs}, Parameters: {params}\n')

    for i, path in enumerate(sorted(img_list)):
        img = Image.open(path).convert('RGB')
        img = transform_test(img)
        img = img.unsqueeze(0)

        out = model(img)

        print(path)
        print(out)

        # if out[0][0] > 0.15:
        if out[0][0] > out[0][1]:
            print('No person present\n')
        else:
            print('Person present\n')
