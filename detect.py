import time
import numpy as np

import torch
from models.get_model import get_model
from torchvision.transforms import functional as F
import torchvision.transforms
import torch.nn.functional

from PIL import Image
from PIL import ImageFile

import matplotlib as plt
import random

from utils.stats import (
    non_max_suppression,load_classe_names)
from utils.opts import Opt

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True


def load_image(path=''):
    with open(path,'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def resize(image,size,mode='bilinear'):
    return torch.nn.functional.interpolate(image,size=size,mode=mode).squeeze(0)


@torch.no_grad()
def detect(model, input, opt):
    w,h = input.size

    if not opt.no_pad2square:
        if w == h:
            image = input
        else:
            dim_diff = abs(w - h)
            padding_1,padding_2 = dim_diff // 2,dim_diff - dim_diff // 2
            padding = (0,padding_1,0,padding_2) if w > h else (padding_1,0,padding_2,0)
            image = F.pad(input,padding,fill=0,padding_mode='constant')
    else:
        image = input

    image = torchvision.transforms.ToTensor()(image).float().unsqueeze(0)
    image = resize(image,opt.image_size).float().unsqueeze(0)
    if opt.gpu:
        model.to(opt.device)
        image = image.to(opt.device)

    t_start = time.time()
    detections = model.forward(image)
    t_end = time.time()
    detections = non_max_suppression(detections,opt.conf_thresh,opt.nms_thresh)
    print(detections)

    print("inference time:" + str(t_end - t_start))
    class_names = load_classe_names(opt.classname_path)

    image = image.squeeze(0).transpose(0,1).transpose(1,2)
    detection = detections[0]

    fig,ax = plt.pyplot.subplots(1)
    plt.pyplot.axis('off')
    plt.pyplot.gca().xaxis.set_major_locator(plt.ticker.NullLocator())
    plt.pyplot.gca().yaxis.set_major_locator(plt.ticker.NullLocator())
    plt.pyplot.tight_layout(pad=0)

    ax.imshow(image.cpu().numpy())

    unique_labels = detection[:,-1].unique()
    num_cls_preds = len(unique_labels)
    color_map = plt.pyplot.get_cmap('tab20b')
    colors = [color_map(i) for i in np.linspace(0,1,opt.num_classes)]
    bbox_colors = random.sample(colors, num_cls_preds)

    for xmin, ymin, xmax, ymax, conf, cls_conf, cls_pred in detection:
        box_w = xmax - xmin
        box_h = ymax - ymin
        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        bbox = plt.patches.Rectangle((xmin,ymin),box_w,box_h,linewidth=2,edgecolor=color,facecolor="none")
        ax.add_patch(bbox)
        plt.pyplot.text(
            xmin,
            ymin,
            s='{} {:.4f}'.format(class_names[int(cls_pred)],cls_conf),
            color='white',
            verticalalignment='top',
            bbox={'color': color,'pad': 0},
        )
    plt.pyplot.show()


if __name__ == "__main__":
    opt = Opt().parse()
    torch.manual_seed(opt.manual_seed)
    model = get_model(opt)
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    assert opt.model == checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(opt.device)
    model.eval()

    image_path = 'path/to/image/file'
    input_image = load_image(image_path)

    detect(model,input_image,opt)
