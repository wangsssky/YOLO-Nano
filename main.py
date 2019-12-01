import os
import time
import torch
from adabound import AdaBound

from data.get_dataset import get_train_dataset, get_val_dataset, get_test_dataset
from models.get_model import get_model
from transforms.get_transforms import get_train_transforms, get_val_transforms, get_test_transforms

from utils.opts import Opt
from utils.logger import Logger

from train import train
from val import val
from test import test

if __name__ == "__main__":

    opt = Opt().parse()

    ########################################
    #                 Model                #
    ########################################
    torch.manual_seed(opt.manual_seed)
    model = get_model(opt)

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.lr,
            weight_decay=opt.weight_decay)
    elif opt.optimizer == 'AdaBound':
        optimizer = AdaBound(
            model.parameters(),lr=opt.lr,final_lr=0.1,
            weight_decay=opt.weight_decay)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr,
            momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        NotImplementedError("Only Adam and SGD are supported")

    best_mAP = 0


    ########################################
    #              Transforms              #
    ########################################
    if not opt.no_train:
        train_transforms = get_train_transforms(opt)
        train_dataset = get_train_dataset(opt, train_transforms)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_threads,
            collate_fn=train_dataset.collate_fn
        )
        train_logger = Logger(os.path.join(opt.checkpoint_path, 'train.log'))

    if not opt.no_val:
        val_transforms = get_val_transforms(opt)
        val_dataset = get_val_dataset(opt, val_transforms)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_threads,
            collate_fn=val_dataset.collate_fn
        )
        val_logger = Logger(os.path.join(opt.checkpoint_path, 'val.log'))

    if opt.test:
        test_transforms = get_test_transforms(opt)
        test_dataset = get_test_dataset(opt,test_transforms)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_threads,
            collate_fn=test_dataset.collate_fn
        )

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.model == checkpoint['model']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(opt.device)
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_mAP = checkpoint["best_mAP"]


    ########################################
    #           Train, Val, Test           #
    ########################################
    if opt.test:
        test(model,test_dataloader,opt.begin_epoch,opt)
    else:
        for epoch in range(opt.begin_epoch, opt.num_epochs + 1):
            if not opt.no_train:
                print("\n---- Training Model ----")
                train(model,optimizer,train_dataloader,epoch,opt,train_logger, best_mAP=best_mAP)

            if not opt.no_val and (epoch+1) % opt.val_interval == 0:
                print("\n---- Evaluating Model ----")
                best_mAP = val(model,optimizer,val_dataloader,epoch,opt,val_logger,best_mAP=best_mAP)
