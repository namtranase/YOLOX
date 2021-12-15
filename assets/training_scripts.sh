# Training LiveTrack dataset
## conf1 (basic)
CUDA_VISIBLE_DEVICES=5 python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_s.pth -d 1 -expn conf1
## conf2 (overfit erly)
CUDA_VISIBLE_DEVICES=6 python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_s.pth -d 1 -expn conf2
# # --------------  training config --------------------- #
## conf3 (yolo original)
CUDA_VISIBLE_DEVICES=7 python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 32 --fp16 -o -c checkpoints/yolox_s.pth -d 1 -expn conf3


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

# Truongxa
# wandb
CUDA_VISIBLE_DEVICES=2 python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 32 --fp16 -o -c weights/yolox_s.pth -d 1 -expn yolox_s_noflip -wb -rn yolox_s_noflip
CUDA_VISIBLE_DEVICES=3 python tools/train.py -f exps/example/custom/yolox_m_livetrack.py -b 32 --fp16 -o -c weights/yolox_m.pth -d 1 -expn yolox_m_noflip -wb -rn yolox_m_noflip
CUDA_VISIBLE_DEVICES=4 python tools/train.py -f exps/example/custom/yolox_x_livetrack.py -b 32 --fp16 -o -c weights/yolox_x.pth -d 1 -expn yolox_x_noflip -wb -rn yolox_x_noflip

# no wandb
CUDA_VISIBLE_DEVICES=2 python tools/train.py -f exps/example/custom/yolox_s_livetrack.py -b 32 --fp16 -o -c weights/yolox_s.pth -d 1 -expn yolox_s_noflip
CUDA_VISIBLE_DEVICES=3 python tools/train.py -f exps/example/custom/yolox_m_livetrack.py -b 32 --fp16 -o -c weights/yolox_m.pth -d 1 -expn yolox_m_noflip
CUDA_VISIBLE_DEVICES=4 python tools/train.py -f exps/example/custom/yolox_x_livetrack.py -b 32 --fp16 -o -c weights/yolox_x.pth -d 1 -expn yolox_x_noflip
