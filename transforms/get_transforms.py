import torch
from .transforms import Compose, PadToSquare, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, ColorJitter


def get_train_transforms(opt):
    transforms = []
    if not opt.no_hflip:
        transforms.append(RandomHorizontalFlip(opt.hflip_prob))
    if not opt.no_vflip:
        transforms.append(RandomVerticalFlip(opt.vflip_prob))
    if not opt.no_crop:
        transforms.append(RandomCrop())
    if not opt.no_color_jitter:
        transforms.append(ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))
    if not opt.no_pad2square:
        transforms.append(PadToSquare())
    transforms.append(ToTensor())

    return Compose(transforms)


def get_val_transforms(opt):
    transforms = []
    if not opt.no_pad2square:
        transforms.append(PadToSquare())
    transforms.append(ToTensor())

    return Compose(transforms)


def get_test_transforms(opt):
    transforms = []
    if not opt.no_pad2square:
        transforms.append(PadToSquare())
    transforms.append(ToTensor())

    return Compose(transforms)