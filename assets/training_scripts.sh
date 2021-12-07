# Training LiveTrack dataset
## conf1 (basic)
python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 16 --fp16 -o -c checkpoints/yolox_s.pth -d 1 -expn conf1

## conf1 (overfit erly)
python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 16 --fp16 -o -c checkpoints/yolox_s.pth -d 2 -expn conf2
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