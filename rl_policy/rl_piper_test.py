
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
from scipy.spatial.transform import Rotation as Rotation

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
        # 可视化相关参数
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = 3
        self.handle.cam.azimuth = 0
        self.handle.cam.elevation = -30
        self.rl_model = PPO.load("./piper_ppo_model.zip")

        # 各关节运动限位
        self.joint_limits = np.array([
            (-2.618, 2.618),
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-1.7452, 1.7452),
        ])

        # 动作空间，6个关节
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
        # 观测空间，包含关节位置和目标位置
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 + 7,))
        self.goal = np.array([
            np.random.uniform(0.1, 0.3),    # x
            np.random.uniform(0.0, 0.3),   # y
            np.random.uniform(0.1, 0.5)     # z
        ])
        print("goal:", self.goal)
        self.np_random = None   

        self.step_number = 0

        #workspace limit of robot
        self.workspace_limits = {
            'x' : (0.2, 0.7),
            'y' : (-0.5, 0.5),
            'z' : (0.2, 0.5)
        }

        self._reset_noise_scale = 1e-2

        # 初始化目标 pose
        self.goal_pos = None
        self.goal_wxyz = None
        self.goal_angle = None

        self._set_goal_pose()
        if self.goal_pos is not None and self.goal_wxyz is not None:
            print(f"self.goal_pos : {self.goal_pos}, self.goal_wxyz : {self.goal_wxyz}")

        self.episode_len = 2000

        self.init_qpos = np.zeros(6)
        self.init_qvel = np.zeros(6)

    def _matrix_to_pose_quat(self, T):
        """
        将4x4齐次变换矩阵转换为位置 + 四元数 (w, x, y, z)
        
        参数：
            T (np.ndarray): 4x4 齐次变换矩阵

        返回：
            position (np.ndarray): 3维位置向量 [x, y, z]
            quaternion (np.ndarray): 四元数 [w, x, y, z]
        """
        assert T.shape == (4, 4), "输入必须是 4x4 的齐次变换矩阵"

        # 提取位置
        position = T[:3, 3]

        # 提取旋转矩阵
        rotation_matrix = T[:3, :3]

        # 转换为四元数 (默认 xyzw)
        quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()

        # 转换为 wxyz 格式
        quat_wxyz = np.roll(quat_xyzw, 1)  # xyzw -> wxyz

        return position, quat_wxyz
    
    #set random goal position for cartesian space
    def _label_goal_pose(self, position, quat_wxyz):
        """
        设置目标位姿（位置 + 姿态）

        Args:
            position: 目标的位置，(x, y, z)，类型为 numpy.ndarray 或 list。
            quat_wxyz: 目标的姿态，四元数 (w, x, y, z)，类型为 numpy.ndarray 或 list。
        """
        ## ====== 设置 target 的位姿 ======
        goal_body_name = "target"
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_body_name)

        if goal_body_id == -1:
            raise ValueError(f"Body named '{goal_body_name}' not found in the model.")

        # 获取 joint ID 和 qpos 起始索引
        goal_joint_id = self.model.body_jntadr[goal_body_id]
        goal_qposadr = self.model.jnt_qposadr[goal_joint_id]

        # 设置位姿
        if goal_qposadr + 7 <= self.model.nq:
            self.data.qpos[goal_qposadr     : goal_qposadr + 3] = position
            self.data.qpos[goal_qposadr + 3 : goal_qposadr + 7] = quat_wxyz
        else:
            print("[警告] target 的 qpos 索引越界或 joint 设置有误")

    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 '{site_name}' 的site")

        # 位置
        position = np.array(self.data.site(site_id).xpos)        # shape (3,)

        # 方向：MuJoCo 已存成9元素向量，无需reshape
        xmat = np.array(self.data.site(site_id).xmat)            # shape (9,)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat)                    # [w, x, y, z]

        return position, quaternion



    def _set_goal_pose(self):
        while True:
            # piper xml 里定义的 6 个关节角名字
            joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            # 通过随机在关节空间内采样得到的目标关节角
            angles = []

            # 随机在关节空间采样
            for joint_name in joints:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id == -1:
                    raise ValueError(f"Joint named '{joint_name}' not found in the model.")
                low_limit = self.model.jnt_range[joint_id, 0] if self.model.jnt_limited[joint_id] else -np.pi
                high_limit = self.model.jnt_range[joint_id, 1] if self.model.jnt_limited[joint_id] else np.pi
                random_angle = np.random.uniform(low_limit, high_limit)
                angles.append(random_angle)

            angles = [0.63853179, 1.30619515, -1.1758934, -0.9242861, -0.56871957, -2.61769393]
            angles = np.array(angles)
            # 
            ori_qpos = self.data.qpos[:6].copy()
            # 模型往前一步
            self.data.qpos[:6] = angles
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            goal_pos, goal_wxyz = self._get_site_pos_ori("end_ee")

            # 恢复
            self.data.qpos[:6] = ori_qpos
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

            x_goal, y_goal, z_goal = goal_pos[0], goal_pos[1], goal_pos[2]

            if (self.workspace_limits['x'][0] <= x_goal <= self.workspace_limits['x'][1] and
                self.workspace_limits['y'][0] <= y_goal <= self.workspace_limits['y'][1] and
                self.workspace_limits['z'][0] <= z_goal <= self.workspace_limits['z'][1]):

                goal_position = np.array([x_goal, y_goal, z_goal])
                self._label_goal_pose(goal_position, goal_wxyz)
                print(f"goal_position : {goal_position}, angles : {angles}")


                self.goal_pos = goal_pos
                self.goal_wxyz = goal_wxyz
                self.goal_angle = angles
                return
    
    def map_action_to_joint_limits(self, action: np.ndarray) -> np.ndarray:
        """
        将 [-1, 1] 范围内的 action 映射到每个关节的具体角度范围。

        Args:
            action (np.ndarray): 形状为 (6,) 的数组，值范围在 [-1, 1]

        Returns:
            np.ndarray: 形状为 (6,) 的数组，映射到实际关节角度范围，类型为 numpy.ndarray
        """

        normalized = (action + 1) / 2
        lower_bounds = self.joint_limits[:, 0]
        upper_bounds = self.joint_limits[:, 1]# 从URDF文件加载机械臂的运动学连接
        # 插值计算
        mapped_action = lower_bounds + normalized * (upper_bounds - lower_bounds)

        return mapped_action
    
    def _set_state(self, qpos, qvel):
        assert qpos.shape == (6,) and qvel.shape == (6,)
        self.data.qpos[:6] = np.copy(qpos)
        self.data.qvel[:6] = np.copy(qvel)
        # mujoco 仿真向前推进一步
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=6
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=6
        )
        
        self._set_state(qpos, qvel)
        self._set_goal_pose()
        obs = self._get_observation()
        self.step_number = 0
        print(f"reset env !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.goal_pos is not None and self.goal_wxyz is not None:
            print(f"self.goal_pos : {self.goal_pos}, self.goal_wxyz : {self.goal_wxyz}")

        return obs, {}
    
    def _get_observation(self):
        return np.concatenate([
            self.data.qpos.flat[:6],
            self.goal_pos,
            self.goal_wxyz
            ])

    def step(self, action):
        # 将 action 映射回真实机械臂关节空间
        mapped_action = self.map_action_to_joint_limits(action)
        self.data.qpos[:6] = mapped_action
        self._label_goal_pose(self.goal_pos, self.goal_wxyz)
        # mujoco 仿真向前推进一步
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

        self.step_number += 1
        observation = self._get_observation()
        # Check if observation contains only finite values
        is_finite = np.isfinite(observation).all()
        current_joint_positions = observation[:6]

        # Check if current joint positions are close to the goal
        goal_reached = np.allclose(current_joint_positions, self.goal_angle, atol=1e-2) 

        


        if goal_reached:
            self.goal_reached_count += 1
            print(f" goal reach !!! ")
            reward = 10
        else:       
            vec_1 = current_joint_positions - self.goal_angle
            reward_dist = -np.linalg.norm(vec_1)
            reward_ctrl = -np.square(action).sum()
            reward = 0.5 * reward_dist + 0.1 * reward_ctrl

        
        done = not is_finite or goal_reached
        info = {'is_success': done}
        truncated = self.step_number > self.episode_len

        self.handle.sync()

        return observation, reward, done, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]


if __name__ == "__main__":
    env = PiperEnv()
    observation, _ = env.reset()

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