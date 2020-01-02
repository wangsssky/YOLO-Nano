import onnxruntime as nxrun
import numpy as np
from PIL import Image, ImageOps, ImageFile
import torch
from utils.stats import (to_cpu, non_max_suppression,load_classe_names)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

no_pad2square = False
CATEGORY_NUM = 80

# From https://github.com/eriklindernoren/PyTorch-YOLOv3
class YOLOLayer(object):
    # detection layer
    def __init__(self, anchors, num_classes, img_dim=416):
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.obj_scale = 1
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, img_dim=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        self.img_dim = img_dim

        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes+5, grid_size, grid_size)
            .permute(0,1,3,4,2)
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0]) # center x
        y = torch.sigmoid(prediction[..., 1]) # center y
        w = prediction[..., 2] # width
        h = prediction[..., 3] # Height
        pred_conf = torch.sigmoid(prediction[..., 4]) # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:]) # Cls Pred

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        return output


anchors52 = [[10,13], [16,30], [33,23]] # 52x52
anchors26 = [[30,61], [62,45], [59,119]] # 26x26
anchors13 = [[116,90], [156,198], [373,326]] # 13x13
YOLOLayer52 = YOLOLayer(anchors52,CATEGORY_NUM,img_dim=416)
YOLOLayer26 = YOLOLayer(anchors26,CATEGORY_NUM,img_dim=416)
YOLOLayer13 = YOLOLayer(anchors13,CATEGORY_NUM,img_dim=416)


def load_image(path=''):
    with open(path,'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def preprocess(input):
    w,h = input.size

    if not no_pad2square:
        if w == h:
            image = input
        else:
            dim_diff = abs(w - h)
            padding_1,padding_2 = dim_diff // 2,dim_diff - dim_diff // 2
            padding = (0,padding_1,0,padding_2) if w > h else (padding_1,0,padding_2,0)
            image = ImageOps.expand(input,border=padding,fill=0)  ##left,top,right,bottom
    else:
        image = input

    image = image.resize((416,416),Image.ANTIALIAS)

    input_data = np.array(image).astype(np.float32)
    input_data = input_data/255.0
    input_data = np.transpose(input_data,(2,0,1))
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def postprecess(outputs):
    detections = []
    temp = YOLOLayer52.forward(torch.tensor(outputs[0]),img_dim=416)
    detections.append(temp)
    temp = YOLOLayer26.forward(torch.tensor(outputs[1]),img_dim=416)
    detections.append(temp)
    temp = YOLOLayer13.forward(torch.tensor(outputs[2]),img_dim=416)
    detections.append(temp)

    detections = to_cpu(torch.cat(detections,1))
    return detections


image_path = '/path/to/a.jpg'
input_image = load_image(image_path)
img_input = preprocess(input_image)


sess = nxrun.InferenceSession('./path/to/simplified.onnx')

print("The model expects input shape: ", sess.get_inputs()[0].shape)
print("The shape of the Image is: ", img_input.shape)

input_name = sess.get_inputs()[0].name
output_name = [sess.get_outputs()[0].name, sess.get_outputs()[1].name, sess.get_outputs()[2].name]
outputs = sess.run(output_name,{input_name: img_input})
detections = postprecess(outputs)

detections = non_max_suppression(detections,0.8,0.5)
print(detections)
