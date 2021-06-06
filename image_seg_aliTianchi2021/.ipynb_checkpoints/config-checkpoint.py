# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:20   xin      1.0         None
'''


from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()


# Alias for easy usage
cfg = _C


_C.MODEL = CN()
_C.MODEL.NAME = "unet"
_C.MODEL.N_CHANNEL = 3
_C.MODEL.N_CLASS = 16
_C.MODEL.LABEL_SMOOTH = False
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.DEVICE_IDS = "0"

_C.MODEL.DROPOUT = 0.5


# UNet
_C.MODEL.UNET = CN()
_C.MODEL.UNET.ENCODE_DIM = 64
_C.MODEL.UNET.BILINEAR = True

# Res_UNet
_C.MODEL.RES_UNET = CN()
_C.MODEL.RES_UNET.BACKBONE = ''

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 2


_C.DATASETS = CN()
_C.DATASETS.NAMES = ('Cityscapes')
# Root PATH to the dataset
_C.DATASETS.DATA_PATH = r'/data/bitahub/Cityscapes'
_C.DATASETS.IMG_SUFFIX = '.tif'
_C.DATASETS.SEG_MAP_SUFFIX = '.tif'

_C.DATASETS.IMAGE_FOLDER = r'rs_imgs'
_C.DATASETS.GT_FOLDER = r'gt_imgs'


_C.INPUT = CN()
_C.INPUT.PIXEL_MEAN = [0.28689552631651594,0.32513300997108185,0.2838917598507514]
_C.INPUT.PIXEL_STD = [0.17613640748441528,0.18099167120139084,0.1777223070810445]
_C.INPUT.SIZE_TRAIN = [512, 512]
_C.INPUT.SIZE_TEST = [512, 512]
_C.INPUT.NORMALIZATION = True

_C.INPUT.USE_MIX_UP = False
_C.INPUT.USE_AUGMIX = False


_C.SOLVER = CN()

_C.SOLVER.OPTIMIZER_NAME = "Adam"


_C.SOLVER.MAX_EPOCHS = 300

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.MIN_LR = 3e-6

_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.USE_WARMUP = True

_C.SOLVER.MOMENTUM = 0.9


_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [40, 70]

_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_EPOCH = 10
_C.SOLVER.WARMUP_BEGAIN_LR = 3e-6
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.START_SAVE_EPOCH = 250

_C.SOLVER.TENSORBOARD = CN()
_C.SOLVER.TENSORBOARD.USE = True
_C.SOLVER.TENSORBOARD.LOG_PERIOD = 20

_C.SOLVER.PER_BATCH = 4
_C.SOLVER.SYNCBN = False
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_CHECKPOINT = r'/usr/demo/common_data/seg_data/guanggu/output/exp2-unet-nowarmup/unet_epoch_300.pth'



_C.TEST = CN()
_C.TEST.WEIGHT = r'/usr/demo/common_data/seg_data/guanggu/output/unet_epoch118.pth'


_C.OUTPUT_DIR = CN()
_C.OUTPUT_DIR = r'/output/unet_noaug_512*512_warmup10_cosineannealinglr3e-4_epo300'
