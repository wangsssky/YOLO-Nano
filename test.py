import time
import numpy as np
import torch
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

from terminaltables import AsciiTable

from utils.stats import (
    non_max_suppression,xywh2xyxy,
    get_batch_statistics,ap_per_class,load_classe_names)


@torch.no_grad()
def test(model,dataloader,epoch,opt):
    labels = []
    sample_matrics = []
    total_time = 0
    pic_num = 0
    first_run = True
    if opt.gpu:
        model.to(opt.device)
    model.eval()

    # warm-up
    if opt.gpu:
        input_shape = (3,416,416)
        dummy_input = Variable(torch.randn(1,*input_shape))
        model.forward(dummy_input.to(opt.device))

    for i,(images,targets) in enumerate(dataloader):
        pic_num += opt.batch_size
        labels += targets[:,1].tolist()
        targets[:,2:] = xywh2xyxy(targets[:,2:])
        targets[:,2:] *= opt.image_size

        if opt.gpu:
            images = Variable(images.to(opt.device))
        t_start = time.time()
        detections = model.forward(images)
        t_end = time.time()
        detections = non_max_suppression(detections,opt.conf_thresh,opt.nms_thresh)

        print("forward time:"+str(t_end - t_start))
        total_time += t_end - t_start

        sample_matrics += get_batch_statistics(detections,targets,iou_threshold=0.5)

    print("Average time:"+str(total_time/pic_num) + "s")
    true_positives,pred_scores,pred_labels = [np.concatenate(x,0) for x in list(zip(*sample_matrics))]
    precision,recall,AP,f1,ap_class = ap_per_class(true_positives,pred_scores,pred_labels,labels)

    metric_table_data = [
        ['Metrics','Value'],['precision',precision.mean()],['recall',recall.mean()],
        ['f1',f1.mean()],['mAP',AP.mean()]]

    metric_table = AsciiTable(
        metric_table_data,
        title='[Epoch {:d}/{:d}'.format(epoch,opt.num_epochs))

    class_names = load_classe_names(opt.classname_path)
    for i,c in enumerate(ap_class):
        metric_table_data += [['AP-{}'.format(class_names[c]),AP[i]]]
    metric_table.table_data = metric_table_data
    print('{}\n'.format(metric_table.table))


