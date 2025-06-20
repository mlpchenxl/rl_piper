
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
import argparse

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PiperEnv(gym.Env):
    def __init__(self, render=True):
        super(PiperEnv, self).__init__()
        # 获取当前脚本文件所在目录
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # 构造 scene.xml 的完整路径
        xml_path = os.path.join(script_dir, '..', 'mujoco_asserts', 'agilex_piper', 'scene.xml')
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
        self.render_mode = render
        if self.render_mode:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3
            self.handle.cam.azimuth = 0
            self.handle.cam.elevation = -30
        else:
            self.handle = None

        self.rl_model = PPO.load("./piper_ik_ppo_model.zip")

        # 各关节运动限位
        self.joint_limits = np.array([
            (-2.618, 2.618),
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-3.14, 3.14),
        ])

        # 动作空间，6个关节
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
        # 观测空间，包含末端位姿和目标位姿
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 7,))
        self.np_random = None   
        self.step_number = 0

        #workspace limit of robot
        self.workspace_limits = {
            'x' : (0.1, 0.7),
            'y' : (-0.7, 0.7),
            'z' : (0.1, 0.7)
        }

        self.goal_reached = False
        self._reset_noise_scale = 0.0

        # 初始化目标 pose
        self.goal_pos = None
        self.goal_quat = None
        self.goal_angle = None
        self._set_goal_pose()

        self.episode_len = 50
        self.init_qpos = np.zeros(6)
        self.init_qvel = np.zeros(6)
    
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

            # angles = [0.63853179, 1.30619515, -1.1758934, -0.9242861, -0.56871957, -2.61769393]
            angles = np.array(angles)
            # 
            ori_qpos = self.data.qpos[:6].copy()
            # 模型往前一步
            self.data.qpos[:6] = angles
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            goal_pos, goal_quat = self._get_site_pos_ori("end_ee")

            # 恢复
            self.data.qpos[:6] = ori_qpos
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

            x_goal, y_goal, z_goal = goal_pos[0], goal_pos[1], goal_pos[2]

            if (self.workspace_limits['x'][0] <= x_goal <= self.workspace_limits['x'][1] and
                self.workspace_limits['y'][0] <= y_goal <= self.workspace_limits['y'][1] and
                self.workspace_limits['z'][0] <= z_goal <= self.workspace_limits['z'][1]):

                goal_position = np.array([x_goal, y_goal, z_goal])
                self._label_goal_pose(goal_position, goal_quat)
                print(f"goal_angles : {angles}")


                self.goal_pos = goal_pos
                self.goal_quat = goal_quat
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
        upper_bounds = self.joint_limits[:, 1]
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

        self.goal_reached = False

        return obs, {}
    
    def _get_observation(self):
        # 读取夹爪末端 pose
        cur_gripper_pos, cur_gripper_quat = self._get_site_pos_ori("end_ee")
        return np.concatenate([
            cur_gripper_pos,
            cur_gripper_quat,
            self.goal_pos,
            self.goal_quat
            ])
    
    def _compute_pos_error_and_reward(self, cur_pos, goal_pos):
        # 计算位置误差（欧氏距离）
        pos_error = np.linalg.norm(cur_pos - goal_pos)
        pos_reward = -np.arctan(pos_error)
        return pos_reward, pos_error
    
    def _compute_ori_error_and_reward(self, cur_quat, goal_quat, axis_weight=None, use_arctan=True):
        """
        Compute orientation reward based on quaternion difference.

        Args:
            cur_quat (np.array): current orientation [w, x, y, z]
            goal_quat (np.array): goal orientation [w, x, y, z]
            axis_weight (np.array or None): weights for [rx, ry, rz] axes
            use_arctan (bool): whether to apply arctangent smoothing to reward

        Returns:
            float: orientation reward (the more negative, the worse the orientation mismatch)
        """

        # Convert to scipy Rotation objects
        r_current = Rotation.from_quat([cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]])  # x,y,z,w
        r_goal = Rotation.from_quat([goal_quat[1], goal_quat[2], goal_quat[3], goal_quat[0]])  # x,y,z,w

        # 计算相对旋转（姿态差）∈ [0, π]
        r_diff = r_goal * r_current.inv()

        # Convert to rotation vector ∈ [[-π, π], [-π, π], [-π, π]]
        rotvec = r_diff.as_rotvec()  # shape (3,), angle * axis

        if axis_weight is None:
            # Default: equal weights
            axis_weight = np.array([1.0, 1.0, 1.0])

        # weighted_error ∈ [0, π ⋅ 1.732]
        weighted_error = np.sum(axis_weight * np.abs(rotvec))

        # Optional: apply arctan smoothing
        if use_arctan:
            # orientation_reward ∈ [-1.387, 0]
            orientation_reward = -np.arctan(weighted_error)
        else:
            # orientation_reward ∈ [-π ⋅ 1.732, 0]
            orientation_reward = -weighted_error

        return orientation_reward, weighted_error
    
    def _compute_reward(self, observation):
        # 提取当前末端 pose
        cur_gripper_pos = observation[:3].copy()
        cur_gripper_quat = observation[3:7].copy()

        # 目标 pose
        goal_pos = self.goal_pos.copy()
        goal_quat = self.goal_quat.copy()

        # 计算位置误差与位置reward
        pos_reward, pos_error = self._compute_pos_error_and_reward(cur_gripper_pos, goal_pos)
        # 计算姿态误差 reward
        ori_reward, ori_error = self._compute_ori_error_and_reward(cur_gripper_quat, goal_quat)

        # 综合奖励，给各个reward设置权重
        w_pos =  2.0
        w_ori = 0.2
        reward = w_pos * pos_reward + w_ori * ori_reward

        # 达到目标阈值时，给额外奖励
        pos_thresh = 0.02  # 2cm
        ori_thresh = 0.1   # 6°
        # print(f"pos_reward : {w_pos *pos_reward}, ori_reward : {w_ori *ori_reward}, angle_reward : {w_angle *angle_reward}")

        if pos_error < pos_thresh:
            reward += 10.0  # 达成位置目标大奖励
            if ori_error < ori_thresh:
                self.goal_reached = True
                reward += 10.0  # 在位置目标满足的情况下，姿态误差也小，再增加一部分奖励
        return reward

    def step(self, action):
        # 将 action 映射回真实机械臂关节空间
        mapped_action = self.map_action_to_joint_limits(action)
        self.data.qpos[:6] = mapped_action
        self._label_goal_pose(self.goal_pos, self.goal_quat)
        # mujoco 仿真向前推进一步 (这里只更新 qpos , 并不会做动力学积分)
        mujoco.mj_forward(self.model, self.data)

        self.step_number += 1
        observation = self._get_observation()
        # 检查观测量是否包含无效值
        is_finite = np.isfinite(observation).all()

        # 计算 reward
        reward = self._compute_reward(observation)
        done = not is_finite or self.goal_reached
        info = {'is_success': done}

        # 检查是否提前终止当前环境采样
        truncated = self.step_number > self.episode_len
        if self.handle is not None:
            self.handle.sync()

        return observation, reward, done, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PiperEnv RL simulation.")
    env = PiperEnv()
    observation, _ = env.reset()

    try:
        # Run the simulation loop
        for step in range(20000):
            action, _states = env.rl_model.predict(observation)
            observation, reward, done, truncated, info = env.step(action)
            if step % 100 == 0:
                print("*****************************")
                print(f"goal_pos : {env.goal_pos}, goal_quat : {env.goal_quat}")
                print(f"cur_pos: {observation[:3]}, cur_quat : {observation[3:7]}")
                print(f"reward : {reward}")

            if done or truncated:
                observation, _ = env.reset()

            time.sleep(0.05)
    finally:
        env.close()