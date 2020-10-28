import os
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 载入图像
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # 没有将mask转为RGB, 因为没一个颜色对应不同的实例
        mask = Image.open(mask_path)
        # 把PIL 图像转为numpy 数组
        mask = np.array(mask)
        # 实例解码为不同的颜色
        obj_ids = np.unique(mask)
        # 第一个id 是背景颜色 直接移除
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # 获取边界框给每一个mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.min(pos[0])

            boxes.append([xmin, ymin, xmax, ymax])

        # 把一切都转为torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # 假设所有的实例都不拥挤
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained pre-trained on coco
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 用用户定义的分类来作为新的分类
num_classes = 2  # 1 class (person) + backgroud
# 获得分类输入的number
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 用新的替换 提前训练的
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

"""
 Modifying the model to add a different backbon
"""
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 为classification和retrun 载入提前训练的model
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN 需要知道backbone输出的通道. mobilenet_v2 的通道是1280
backbone.out_channels = 1280

# 生成一个 5*3的anchors
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=(0.5, 1.0, 2.0))

# 让我们来定义我们将使用哪些特征图来执行感兴趣区域的裁剪以及调整后的大小。

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

# 将各个部分放在一个FasterRCNN模型中

model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

# 这个案例中,我们为了调优pre-trained 模型, 由于数据集很小, 所以采取方案1
# 使用Mask R-CNN 计算实例的边框

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # pretrained_backbone = False
    # backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    # model = MaskRCNN(backbone, num_classes=91)
    # state_dict = torch.load("D:/python/pycharmWorkspace/my_pytorch_tutorials/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth")
    # model.load_state_dict(state_dict)

    # 获取分类的类别
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 使用新的in_features 替换预先训练的模型的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 获取对mask 分类的输入特性的数字
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 替换mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes
                                                       )

    return model


"""
把之前的全都集合起来
"""
import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
#
# import utils
#
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=2, shuffle=True, num_workers=4,
#     collate_fn=utils.collate_fn)
#
# # For Training
# images, targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images, targets)  # Returns losses and detections
#
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)  # Returns predictions

from engine import train_one_epoch, evaluate
import utils


from engine import train_one_epoch, evaluate
import utils


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == '__main__':
    main()