#  单臂 piper 强化学习

本项目支持单臂（single_piper）强化学习逆运动学

---


## 安装

### 1. 拉取项目代码
```bash
$ git clone https://github.com/mlpchenxl/rl_piper.git
$ cd rl_piper/
```
### 2. 创建虚拟环境并安装依赖
```bash
$ conda create -n rl_piper python=3.10.9
$ conda activate rl_piper
$ pip install -r requirements.txt
```

### 3. train
python rl_policy/rl_piper_ik__train.py

### 4. 验证
python rl_policy/rl_piper_ik__test.py



