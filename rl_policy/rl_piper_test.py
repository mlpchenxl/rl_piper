
import numpy as np
import mujoco
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import os
import time

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PiperEnv(gym.Env):
    def __init__(self):
        super(PiperEnv, self).__init__()
        # 获取当前脚本文件所在目录
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # 构造 scene.xml 的完整路径
        xml_path = os.path.join(script_dir, '..', 'mujoco_asserts', 'agilex_piper', 'scene.xml')
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = 3
        self.handle.cam.azimuth = 0
        self.handle.cam.elevation = -30

        self.rl_model = PPO.load("./piper_ppo_model.zip")
        # 动作空间，7个关节
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
        # 观测空间，包含关节位置和目标位置
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 + 3,))
        self.goal = np.array([
            np.random.uniform(0.1, 0.3),    # x
            np.random.uniform(-0.3, 0.3),   # y
            np.random.uniform(0.1, 0.5)     # z
        ])
        print("goal:", self.goal)
        self.np_random = None  

 

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.goal = np.array([
            self.np_random.uniform(0.1, 0.3),
            self.np_random.uniform(-0.3, 0.3),
            self.np_random.uniform(0.1, 0.5)
        ])
        print("goal:", self.goal)
        obs = np.concatenate([self.data.qpos[:6], self.goal])
        return obs

    def step(self, action):
        self.data.qpos[:6] = action
        mujoco.mj_step(self.model, self.data)
        achieved_goal = self.data.body(self.end_effector_id).xpos
        reward = -np.linalg.norm(achieved_goal - self.goal)
        reward -= 0.3*self.data.ncon
        terminated = np.linalg.norm(achieved_goal - self.goal) < 0.01
        truncated = False
        info = {'is_success': terminated}
        obs = np.concatenate([self.data.qpos[:6], achieved_goal])

        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.handle.sync()

        return obs, reward, terminated, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]


if __name__ == "__main__":
    env = PiperEnv()
    observation = env.reset()

    try:
        # Run the simulation loop
        for step in range(20000):
            action, _states = env.rl_model.predict(observation)

            current_joint_positions = observation[:6]
            observation, reward, done, truncated, info = env.step(action)
            if step % 100 == 0:
                print("*****************************")
                print(f"Goal goal: {env.goal}")
                print(f"Current Joint Positions: {observation[:6]}")
                print(reward)

            if done or truncated:
                observation = env.reset()
                break

            time.sleep(0.05)
    finally:
        env.close()