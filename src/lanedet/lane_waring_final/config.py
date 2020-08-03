Dataset_Path = dict(
    CULane = "/media/iairiv/dbc2aff7-a3bd-4fe8-89f6-9b61c52fc822/kongdeqian/CUlane",
    Tusimple = "/media/iairiv/7f072dbd-7037-43d9-9b3f-e8b67a86e0be/home/admin1/DATA/Tusimple/train_set"
)

Img_Size = dict(
    CULane=(590, 1640),
    Tusimple=(720, 1280)
)

IMG_MEAN = dict(
    IMAGENET=(0.485, 0.456, 0.406),
    CULane=(0.36192531, 0.3675953, 0.36898042)
)
IMG_STD = dict(
    IMAGENET=(0.229, 0.224, 0.225),
    CULane=(0.28024873, 0.29112347, 0.3054787)
)

pretrained_weights = dict(
    vgg16_bn=r'/home/iairiv/PycharmProjects/pertrained_weights/vgg16_bn-6c64b313.pth',
    vgg16=r'/home/iairiv/PycharmProjects/pertrained_weights/vgg16-397923af.pth',
    resnet50=r'/home/iairiv/PycharmProjects/pertrained_weights/resnet50-19c8e357.pth'
)

