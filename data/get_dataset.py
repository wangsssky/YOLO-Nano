import os


def get_train_dataset(opt, transforms=None):
    if opt.dataset == 'coco':
        from data.coco import COCO
        dataset = COCO(
            opt.dataset_path, os.path.join(opt.annotation_path, opt.train_json), subset='train',
            image_size=opt.image_size, multi_scale=(not opt.no_multi_scale), transforms=transforms)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("train dataset [%s] was created" % (dataset.name()))
    return dataset


def get_val_dataset(opt, transforms=None):
    if opt.dataset == 'coco':
        from data.coco import COCO
        dataset = COCO(
            opt.dataset_path, os.path.join(opt.annotation_path, opt.val_json), subset='val',
            image_size=opt.image_size, multi_scale=False, transforms=transforms)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("val dataset [%s] was created" % (dataset.name()))
    return dataset


def get_test_dataset(opt, transforms=None):
    if opt.dataset == 'coco':
        from data.coco import COCO
        dataset = COCO(
            opt.dataset_path, os.path.join(opt.annotation_path, opt.test_json), subset='val',
            image_size=opt.image_size, multi_scale=False, transforms=transforms)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("test dataset [%s] was created" % (dataset.name()))
    return dataset