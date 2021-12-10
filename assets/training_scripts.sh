# Training LiveTrack dataset
## conf1 (basic)
CUDA_VISIBLE_DEVICES=5 python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_s.pth -d 1 -expn conf1

## conf2 (overfit erly)
CUDA_VISIBLE_DEVICES=6 python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_s.pth -d 1 -expn conf2
# # --------------  training config --------------------- #
#     self.warmup_epochs = 5
#     self.max_epoch = 300
#     self.warmup_lr = 0
#     self.basic_lr_per_img = 0.01 / 64.0
#     self.scheduler = "yoloxwarmcos"
#     self.no_aug_epochs = 15
#     self.min_lr_ratio = 0.05
#     self.ema = True

#     self.weight_decay = 5e-4
#     self.momentum = 0.9

## conf3 (yolo original)
CUDA_VISIBLE_DEVICES=7 python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_s.pth -d 1 -expn conf3
# self.mosaic_prob = 1.0
# self.mixup_prob = 1.0
# self.hsv_prob = 1.0
# self.flip_prob = 0.5
# self.degrees = 10.0
# self.translate = 0.1
# self.mosaic_scale = (0.1, 2)
# self.mixup_scale = (0.5, 1.5)
# self.shear = 2.0
# self.enable_mixup = True

# # --------------  training config --------------------- #
# self.warmup_epochs = 5
# self.warmup_lr = 0
# self.basic_lr_per_img = 0.01 / 64.0
# self.scheduler = "yoloxwarmcos"
# self.no_aug_epochs = 15
# self.min_lr_ratio = 0.05
# self.ema = True

# self.weight_decay = 5e-4
# self.momentum = 0.9
# self.print_interval = 10
# self.eval_interval = 10
# self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

# # -----------------  testing config ------------------ #
# self.test_size = (640, 640)
# self.test_conf = 0.01
# self.nmsthre = 0.65

# Conf3 + yolox-m
CUDA_VISIBLE_DEVICES=2 python tools/train.py -f exps/example/custom/yolox_m_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_m.pth -d 1 -expn yolox_m_conf3

# Conf3 + yolox-x
CUDA_VISIBLE_DEVICES=3 python tools/train.py -f exps/example/custom/yolox_x_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_x.pth -d 1 -expn yolox_x_conf3

# Superpod
# 1gpu
python tools/train.py -f exps/example/custom/yolox_x_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_x.pth -d 1 -expn yolox_x_conf3
python tools/train.py -f exps/example/custom/yolox_m_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_m.pth -d 1 -expn yolox_m_conf3
# 2gpu
python tools/train.py -f exps/example/custom/yolox_x_livetrack.py -b 64 --fp16 -o -c checkpoints/yolox_x.pth -d 2 -expn yolox_x_conf3_2gpu
# No classes head
python tools/train.py -f exps/example/custom/yolox_x_livetrack_no_classes.py -b 32 --fp16 -o -c checkpoints/yolox_x.pth -d 1 -expn yolox_x_conf3_no_classes
