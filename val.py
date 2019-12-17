import os
import numpy as np
import torch
from torch.autograd import Variable
from terminaltables import AsciiTable


from utils.stats import (
    non_max_suppression, xywh2xyxy,
    get_batch_statistics, ap_per_class, load_classe_names)


@torch.no_grad()
def val(model,optimizer,dataloader,epoch,opt,val_logger,best_mAP=0):
    labels = []
    sample_matrics = []
    if opt.gpu:
        model = model.to(opt.device)
    model.eval()

    for i, (images, targets) in enumerate(dataloader):
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= opt.image_size

        batches_done = len(dataloader) * epoch + i
        if opt.gpu:
            images = Variable(images.to(opt.device))

        detections = model.forward(images)
        detections = non_max_suppression(detections, opt.conf_thresh, opt.nms_thresh)
        sample_matrics += get_batch_statistics(detections, targets.cpu(), iou_threshold=0.5)

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_matrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    
    # logging
    metric_table_data = [
        ['Metrics', 'Value'], ['precision', precision.mean()], ['recall', recall.mean()], 
        ['f1', f1.mean()], ['mAP', AP.mean()]]
     
    metric_table = AsciiTable(
            metric_table_data,
            title='[Epoch {:d}/{:d}'.format(epoch, opt.num_epochs))

    class_names = load_classe_names(opt.classname_path)
    for i, c in enumerate(ap_class):
        metric_table_data += [['AP-{}'.format(class_names[c]), AP[i]]]
    metric_table.table_data = metric_table_data
    val_logger.print_and_write('{}\n'.format(metric_table.table))

    if best_mAP < AP.mean():
        save_file_path = os.path.join(opt.checkpoint_path, 'best.pth')
        states = {
            'epoch': epoch + 1,
            'model': opt.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_mAP': best_mAP,
        }
        torch.save(states,save_file_path)
        best_mAP = AP.mean()

    print("current best mAP:" + str(best_mAP))

    return best_mAP
