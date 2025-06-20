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
> ⚠️ **Note**
> 
> 可视化训练不支持并行 ！
>
> 并行训练必须关闭可视化 ！
>
> --n_env 可根据电脑性能自行选择并行环境的个数
#### 可视化训练过程
```bash
python rl_policy/rl_piper_ik_train.py --render --n_env 1
```
#### 并行训练
```bash
python rl_policy/rl_piper_ik_train.py --n_env 100
```

#### 训练结束后会在 ${项目路径} 下生成一个名为 piper_ik_ppo_model.zip
### 4. 验证
```bash
python rl_policy/rl_piper_ik_test.py
```

## Notion
### 1. action space （机械臂 6 自由度关节）
#### 维度是 6
### 2.observation space 当前末端 pose (xyz + quat) + 目标末端 pose ((xyz + quat))
#### 维度是 7 + 7 = 14

