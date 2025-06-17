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
from scipy.spatial.transform import Rotation as Rotation
from stable_baselines3.common.evaluation import evaluate_policy
import cv2

import glfw

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PiperEnv(gym.Env):
    def __init__(self, render=True):
        super(PiperEnv, self).__init__()
        # 获取当前脚本文件所在目录
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # 构造 scene.xml 的完整路径
        xml_path = os.path.join(script_dir, '..', 'mujoco_asserts', 'agilex_piper_grasp', 'scene.xml')
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self._glfw_window = glfw.create_window(640, 480, "Hidden", None, None)
        glfw.make_context_current(self._glfw_window)

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

            # 初始化渲染所需结构体（顺序不能错）
            self.camera = mujoco.MjvCamera()
            self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)



        else:
            self.handle = None

        # 各关节运动限位
        self.ctrl_limits = np.array([
            (-2.618, 2.618),
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-3.14, 3.14),
            (0, 0.035),
        ])

        # 动作空间，7 个控制量
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,))
        

        # 所需要用的相机
        self.camera_names = ["wrist"]
        # 环境中 robot 关节 
        self.robot_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        # 环境中需要交互的物体
        self.object_names = ["apple"]
        # 计算关节数量
        num_joints = len(self.robot_joint_names)
        # 构建 observation_space
        obs_dict = {}
        # 为每个摄像头添加一个图像空间，命名为"{camera_name}_image"
        for cam in self.camera_names:
            obs_dict[f"{cam}_image"] = spaces.Box(
                low=0,
                high=255,
                shape=(480, 640, 3),
                dtype=np.uint8
            )

        # 添加关节位置空间
        obs_dict["joint_pos"] = spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(num_joints,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict(obs_dict)

        # 随机采样设置
        self.np_random = None   
        self.step_number = 0
        self._reset_noise_scale = 1e-2
        # 一个环境最大采样次数
        self.episode_len = 2000

    def get_sensor_data(self, sensor_name):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in model!")
        start_idx = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        sensor_values = self.data.sensordata[start_idx : start_idx + dim]  # ← 这里改了
        return sensor_values

    
    def get_image_by_camera_name(self, camera_name: str, w: int, h: int) -> np.ndarray:
        cam_id = -1
        for i in range(self.model.ncam):
            name_str = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name_str == camera_name:
                cam_id = i
                break
        if cam_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in model.")

        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = cam_id

        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        return bgr



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

    
    def map_action_to_joint_limits(self, action: np.ndarray) -> np.ndarray:
        """
        将 [-1, 1] 范围内的 action 映射到每个关节的具体角度范围。

        Args:
            action (np.ndarray): 形状为 (6,) 的数组，值范围在 [-1, 1]

        Returns:
            np.ndarray: 形状为 (6,) 的数组，映射到实际关节角度范围，类型为 numpy.ndarray
        """

        normalized = (action + 1) / 2
        lower_bounds = self.ctrl_limits[:, 0]
        upper_bounds = self.ctrl_limits[:, 1]
        # 插值计算
        mapped_action = lower_bounds + normalized * (upper_bounds - lower_bounds)

        return mapped_action
    
    def _set_state(self, joint_names):
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"未找到名为 '{name}' 的关节")

            qpos_start = self.model.jnt_qposadr[joint_id]
            qvel_start = self.model.jnt_dofadr[joint_id]

            joint_type = self.model.jnt_type[joint_id]
            if joint_type == 0:
                dof = 7
            elif joint_type == 1:
                dof = 4
            else:
                dof = 1

            self.data.qpos[qpos_start : qpos_start + dof] = np.zeros(dof)
            self.data.qvel[qvel_start : qvel_start + dof] = np.zeros(dof)


    def _reset_objects_positions(self, object_names, xy_low=(-0.2, -0.2), xy_high=(0.2, 0.2), fixed_z=0.766):
        if isinstance(object_names, str):
            object_names = [object_names]

        for name in object_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"未找到名为 '{name}' 的关节")

            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]

            joint_type = self.model.jnt_type[joint_id]
            if joint_type == 0:  # 自由关节（free joint）
                dof_qpos = 7  # 位置(3) + 四元数(4)
                dof_qvel = 6  # 线速度(3) + 角速度(3)
            elif joint_type == 1:  # 球关节 (ball joint)
                dof_qpos = 4  # 四元数(4)
                dof_qvel = 3  # 角速度(3)
            else:  # 其他关节（铰链、滑动）
                dof_qpos = 1  # 标量位置
                dof_qvel = 1  # 标量速度

            xy = self.np_random.uniform(low=xy_low, high=xy_high)
            z = fixed_z

            def random_unit_quaternion(rng):
                """生成随机单位四元数, rng为numpy随机生成器"""
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                return q

            if joint_type == 0:  # free joint，位置 + 四元数
                self.data.qpos[qpos_idx : qpos_idx + dof_qpos] = np.concatenate([xy, [z], random_unit_quaternion(self.np_random)])
            elif joint_type == 1:  # 球关节，只设置四元数
                self.data.qpos[qpos_idx : qpos_idx + dof_qpos] = random_unit_quaternion(self.np_random)
            else:  # 其他关节，只设置标量位置（xy和z无意义）
                self.data.qpos[qpos_idx] = 0.0  # 可以改成其他合理的初始值

            # 速度全部归零
            self.data.qvel[qvel_idx : qvel_idx + dof_qvel] = np.zeros(dof_qvel)
        

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        if self.robot_joint_names is not None and self.object_names is not None:
            self._set_state(self.robot_joint_names)
            self._reset_objects_positions(self.object_names)
        else:
            raise ValueError(f" You need provide robot_joint_names and object_names to reset env. ")
        # mujoco 环境往前走一步
        mujoco.mj_step(self.model, self.data)
        obs = self._get_observation()
        self.step_number = 0
        print(f"reset env successed. ")

        return obs, {}
    
    def _get_observation(self):
        obs = {}
        for cam_name in self.camera_names:
            obs[f"{cam_name}_image"] = self.get_image_by_camera_name(cam_name, 640, 480)
        obs["joint_pos"] = np.array(
            [self.get_sensor_data(name + "_pos").flatten() for name in self.robot_joint_names],
            dtype=np.float32
        ).flatten()
        
        return obs


    
    def _get_body_pose(self, body_name: str) -> np.ndarray:
        """
        通过body名称获取其位姿信息, 返回一个7维向量
        :param body_name: body名称字符串
        :return: 7维numpy数组, 格式为 [x, y, z, w, x, y, z]
        :raises ValueError: 如果找不到指定名称的body
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的body")
        
        # 提取位置和四元数并合并为一个7维向量
        position = np.array(self.data.body(body_id).xpos)  # [x, y, z]
        quaternion = np.array(self.data.body(body_id).xquat)  # [w, x, y, z]
        
        return position, quaternion
    
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
    
    def _get_reward(self, gripper_site_name: str, target_body_name: str) -> float:
        """
        根据末端夹爪的 site 和目标物体的 body，计算靠近奖励。
        奖励越高，表示夹爪越接近目标物体。
        
        :param gripper_site_name: 夹爪 site 名称
        :param target_body_name: 目标物体的 body 名称
        :return: 奖励值，float，范围大致在 [0, 1]
        """
        # 获取两者的位置
        gripper_pos, _ = self._get_site_pos_ori(gripper_site_name)
        target_pos, _ = self._get_body_pose(target_body_name)
        
        # 计算欧几里得距离
        distance = np.linalg.norm(gripper_pos - target_pos)
        
        # 奖励函数：距离越小，奖励越接近 1；距离越远，接近 0
        reward = 1 - np.tanh(5 * distance)
        
        return reward

    def step(self, action):
        # 将 action 映射回真实机械臂关节空间
        mapped_action = self.map_action_to_joint_limits(action)
        self.data.ctrl[:7] = mapped_action
        # mujoco 仿真向前推进一步 (这里只更新 qpos , 并不会做动力学积分)
        mujoco.mj_step(self.model, self.data)

        self.step_number += 1
        observation = self._get_observation()
        # Check if observation contains only finite values
        is_finite = False
        reward = self._get_reward("end_ee", "apple")
        
        done = False
        info = {'is_success': done}
        truncated = self.step_number > self.episode_len
        if self.handle is not None:
            self.handle.sync()

        return observation, reward, done, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]


if __name__ == "__main__":
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(1200,900,"mujoco",None,None)
    glfw.make_context_current(window)
    env = make_vec_env(lambda: PiperEnv(), n_envs=1)

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[
            {
                # 分别为共享层（用于所有输入），
                # pi（策略） 和 vf（值函数）分支
                "wrist": [64, 64],  # 可选：图像路径单独预处理的层（optional）
                "joint_pos": [64],  # 可选：向量路径单独预处理的层（optional）
                "pi": [256, 128],
                "vf": [256, 128],
            }
        ]
    )

    model = PPO(
        "MultiInputPolicy",     # ← 必须是 MultiInputPolicy
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./ppo_piper_grasp/"
    )

    model.learn(total_timesteps=2048*100, progress_bar=True)
    model.save("piper_grasp_ppo_model")

    print(" model sava success ! ")
