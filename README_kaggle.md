1. Set up your training data.
- I put training data on folder ../data/COTS_YOLOX.
- Config file for trainning is `
2. Train
```bash
python tools/train.py -f exps/example/custom/yolox_x_cots.py -b 32 --fp16 -o -c weights/yolox_x.pth -d 1 -expn lab
```
3. Infer
