
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

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PiperEnv(gym.Env):
    def __init__(self):
        super(PiperEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path(
            '../mujoco_asserts/agilex_piper/scene.xml')
        self.data = mujoco.MjData(self.model)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = 3
        self.handle.cam.azimuth = 0
        self.handle.cam.elevation = -30

        # 动作空间，7个关节
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,))
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
        return obs, {}

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
    env = make_vec_env(lambda: PiperEnv(), n_envs=1)

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="../tensorboard/"
    )

    model.learn(total_timesteps=2048*100)
    model.save("piper_ppo_model")
    