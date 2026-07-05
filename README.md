## Installation
```bash
conda create -n octfractal python=3.10 -y

conda activate octfractal

pip install torch torchvision torchaudio

pip install -r requirements.txt
```
## Data Process

原始的shapenet经过octgpt预处理后再运行下面的命令进行预处理
```bash
python tools/preprocess.py \
  --location /data/ShapeNet/datasets_256_test \
  --filelist /data/ShapeNet/filelist/train_airplane.txt \
  --output /data/ShapeNet_process/airplane_train \
  --points_scale 0.5 \
  --full_depth 2 \
  --depth_stop 6 \
  --overwrite
```


## Train
```bash
python main_fractal.py --config configs/shapenet_fractal.yaml
```

## Generate
```bash
python main_fractal.py --config configs/shapenet_fractal.yaml SOLVER.run generate SOLVER.ckpt logs/fractal/airplane/best_model.pth
```
最终生成的体素占据结果保存在logs/fractal/airplane/visual目录下，生成的sdf与mesh则在logs/fractal/airplane/sdf目录下

## Checkpoints

训练好的权重与完整大作业文件已上传北大网盘：[2300094810_王兴辰_几何计算前沿大作业_Octfractal](https://disk.pku.edu.cn/link/AA231A756E92994BD9BBCCD4D104033F6F)（永久有效）。

## P.S
如果想要生成别的数据，请自行修改预处理的数据类别和参数文件进行训练