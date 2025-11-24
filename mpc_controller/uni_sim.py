"""
Lite3四足机器人配置文件
基于URDF自动生成的参数
机器人总质量: 11.938 kg
身体质量: 5.606 kg
腿数: 4
关节数: 12 (每条腿3个关节)
"""
import re
import numpy as np

# ==================== URDF 配置 ====================
URDF_NAME = "C:\\Users\\10947\\Desktop\\URDF_model-main\\Lite3\\Lite3_urdf\\urdf\\Lite3.urdf"
START_POS = [0, 0, 0.25]  # 初始位置 [x, y, z]

# ==================== MPC 参数 ====================
# 从URDF提取的身体参数
MPC_BODY_MASS = 11.938  # 使用实际质量，不除以9.8
MPC_BODY_INERTIA = np.array([
    0.02456, 0, 0,
    0, 0.05518, 0,
    0, 0, 0.07016
])
MPC_BODY_HEIGHT = 0.25  # 身体高度（恢复为先前能站立的默认）
MPC_VELOCITY_MULTIPLIER = 0.5  # 速度倍增系数
MPC_VELOCITY_MULTIPLIER = 0.5  # 速度倍增系数
  
ACTION_REPEAT = 5  # 动作重复次数

# ==================== 关节模式匹配 ====================
_IDENTITY_ORIENTATION = [0, 0, 0, 1]

# Lite3关节命名模式：
# HipX = 外展/内收关节 (abduction)
# HipY = 髋关节 (hip)  
# Knee = 膝关节 (knee)
HIPX_NAME_PATTERN = re.compile(r"\w+_HipX_\w+")
HIPY_NAME_PATTERN = re.compile(r"\w+_HipY_\w+")
KNEE_NAME_PATTERN = re.compile(r"\w+_Knee_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_FOOT")  # 如果有足端
IMU_NAME_PATTERN = re.compile(r"imu\d*")

# ==================== 髋关节位置 ====================
# 从URDF提取的髋关节位置（相对于身体中心）
# 顺序: FR, FL, RR, RL (前右, 前左, 后右, 后左)
_DEFAULT_HIP_POSITIONS = (
    (0.1745, -0.062, 0),   # FR - 前右
    (0.1745, 0.062, 0),    # FL - 前左
    (-0.1745, -0.062, 0),  # RR - 后右  
    (-0.1745, 0.062, 0),   # RL - 后左
)

# ==================== PD 增益 ====================
# 这些值需要根据实际机器人进行调整
# HipX关节（外展/内收）
ABDUCTION_P_GAIN = 300.0    # 从200增加
ABDUCTION_D_GAIN = 3.0      # 从1.0增加

HIP_P_GAIN = 300.0          # 从200增加
HIP_D_GAIN = 5.0            # 从2.0增加

KNEE_P_GAIN = 350.0         # 从220增加
KNEE_D_GAIN = 5.0           # 从2.0增加

# ==================== PyBullet 链接常量 ====================
_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3                                                                                                                                                                                                                                                                                                                                                                                                          

# ==================== 关节偏移 ====================
# 如果需要补偿机械零点偏移，在这里设置
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0

# ==================== 机器人配置 ====================
NUM_LEGS = 4
NUM_MOTORS = 12  # 4条腿 x 3个关节

# 从URDF提取的初始关节角度（中间值）
# 顺序: FL_HipX, FL_HipY, FL_Knee, FR_HipX, FR_HipY, FR_Knee,
#      HL_HipX, HL_HipY, HL_Knee, HR_HipX, HR_HipY, HR_Knee
LITE3_DEFAULT_ABDUCTION_ANGLE = 0.0      # HipX 中间值
LITE3_DEFAULT_HIP_ANGLE = -1.178         # HipY 中间值 (恢复原始)
LITE3_DEFAULT_KNEE_ANGLE = 1.658         # Knee 中间值 (恢复原始)

INIT_MOTOR_ANGLES = np.array([
    LITE3_DEFAULT_ABDUCTION_ANGLE,
    LITE3_DEFAULT_HIP_ANGLE,
    LITE3_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

# ==================== 电机名称 ====================
# 必须与URDF中的关节名称完全匹配
# 注意：电机顺序必须与MPC控制器期望的顺序一致（即A1的顺序）
# MPC期望的顺序: FR, FL, RR(HR), RL(HL)
MOTOR_NAMES = [
    "FR_HipX_joint",      # 0: FR_hip (MPC期望的前右髋)
    "FR_HipY_joint",      # 1: FR_upper
    "FR_Knee_joint",      # 2: FR_lower
    "FL_HipX_joint",      # 3: FL_hip
    "FL_HipY_joint",      # 4: FL_upper
    "FL_Knee_joint",      # 5: FL_lower
    "HR_HipX_joint",      # 6: RR_hip (HR在Lite3中映射到后右)
    "HR_HipY_joint",      # 7: RR_upper
    "HR_Knee_joint",      # 8: RR_lower
    "HL_HipX_joint",      # 9: RL_hip (HL在Lite3中映射到后左)
    "HL_HipY_joint",      # 10: RL_upper
    "HL_Knee_joint",      # 11: RL_lower
]

# ==================== 关节限制 ====================
# 从URDF提取的关节限制 [lower, upper, effort, velocity]
JOINT_LIMITS = {
    "FL_HipX_joint": {"lower": -0.523, "upper": 0.523, "effort": 24.0, "velocity": 26.2},
    "FL_HipY_joint": {"lower": -2.670, "upper": 0.314, "effort": 24.0, "velocity": 26.2},
    "FL_Knee_joint": {"lower": 0.524, "upper": 2.792, "effort": 36.0, "velocity": 17.3},
    "FR_HipX_joint": {"lower": -0.523, "upper": 0.523, "effort": 24.0, "velocity": 26.2},
    "FR_HipY_joint": {"lower": -2.670, "upper": 0.314, "effort": 24.0, "velocity": 26.2},
    "FR_Knee_joint": {"lower": 0.524, "upper": 2.792, "effort": 36.0, "velocity": 17.3},
    "HL_HipX_joint": {"lower": -0.523, "upper": 0.523, "effort": 24.0, "velocity": 26.2},
    "HL_HipY_joint": {"lower": -2.670, "upper": 0.314, "effort": 24.0, "velocity": 26.2},
    "HL_Knee_joint": {"lower": 0.524, "upper": 2.792, "effort": 36.0, "velocity": 17.3},
    "HR_HipX_joint": {"lower": -0.523, "upper": 0.523, "effort": 24.0, "velocity": 26.2},
    "HR_HipY_joint": {"lower": -2.670, "upper": 0.314, "effort": 24.0, "velocity": 26.2},
    "HR_Knee_joint": {"lower": 0.524, "upper": 2.792, "effort": 36.0, "velocity": 17.3},
}

# ==================== 电机控制模式 ====================
# 使用PD控制器
MOTOR_CONTROL_POSITION = 1 
# 直接施加电机扭矩
MOTOR_CONTROL_TORQUE = 2
# 混合模式：对每个电机应用元组 (q, qdot, kp, kd, tau)
# q, qdot是电机位置和速度。kp和kd是PD增益。tau是额外的电机扭矩
# 这是最灵活的控制模式
MOTOR_CONTROL_HYBRID = 3
MOTOR_CONTROL_PWM = 4  # 仅用于Minitaur

# 电机命令维度
MOTOR_COMMAND_DIMENSION = 5

# 电机命令元组中各字段的索引
POSITION_INDEX = 0
POSITION_GAIN_INDEX = 1
VELOCITY_INDEX = 2
VELOCITY_GAIN_INDEX = 3
TORQUE_INDEX = 4


# ==================== Lite3电机模型 ====================
class Lite3MotorModel(object):
    """Lite3机器人的电机模型
    
    支持位置控制、扭矩控制和混合控制模式。
    混合命令格式: [目标位置, kp, 目标速度, kd, 前馈扭矩]
    扭矩 = -kp * (实际位置 - 目标位置) - kd * (实际速度 - 目标速度) + 前馈扭矩
    """

    def __init__(self, kp, kd, torque_limits=None, motor_control_mode=MOTOR_CONTROL_POSITION):
        self._kp = kp
        self._kd = kd
        self._torque_limits = torque_limits
        if torque_limits is not None:
            if isinstance(torque_limits, (list, tuple, np.ndarray)):
                self._torque_limits = np.asarray(torque_limits)
            else:
                self._torque_limits = np.full(NUM_MOTORS, torque_limits)
        self._motor_control_mode = motor_control_mode
        self._strength_ratios = np.full(NUM_MOTORS, 1.0)

    def set_strength_ratios(self, ratios):
        """设置每个电机相对于默认值的强度比例"""
        self._strength_ratios = ratios

    def set_motor_gains(self, kp, kd):
        """设置所有电机的PD增益"""
        self._kp = kp
        self._kd = kd

    def set_voltage(self, voltage):
        pass

    def get_voltage(self):
        return 0.0

    def set_viscous_damping(self, viscous_damping):
        pass

    def get_viscous_damping(self):
        return 0.0

    def convert_to_torque(self, motor_commands, motor_angle, motor_velocity,
                          true_motor_velocity, motor_control_mode=None):
        """将电机命令转换为扭矩
        
        Args:
            motor_commands: 电机命令（位置、扭矩或混合模式）
            motor_angle: 当前电机角度
            motor_velocity: 当前电机速度
            true_motor_velocity: 真实电机速度
            motor_control_mode: 控制模式
            
        Returns:
            actual_torque: 需要施加的扭矩
            observed_torque: 传感器观测的扭矩
        """
        del true_motor_velocity
        
        if not motor_control_mode:
            motor_control_mode = self._motor_control_mode

        # 直接扭矩控制
        if motor_control_mode is MOTOR_CONTROL_TORQUE:
            assert len(motor_commands) == NUM_MOTORS
            motor_torques = self._strength_ratios * motor_commands
            return motor_torques, motor_torques

        # 位置控制模式
        desired_motor_angles = None
        desired_motor_velocities = None
        kp = None
        kd = None
        additional_torques = np.full(NUM_MOTORS, 0)
        
        if motor_control_mode is MOTOR_CONTROL_POSITION:
            assert len(motor_commands) == NUM_MOTORS
            kp = self._kp
            kd = self._kd
            desired_motor_angles = motor_commands
            desired_motor_velocities = np.full(NUM_MOTORS, 0)
            
        elif motor_control_mode is MOTOR_CONTROL_HYBRID:
            # 混合控制：60维向量（每个电机5个值）
            assert len(motor_commands) == MOTOR_COMMAND_DIMENSION * NUM_MOTORS
            kp = motor_commands[POSITION_GAIN_INDEX::MOTOR_COMMAND_DIMENSION]
            kd = motor_commands[VELOCITY_GAIN_INDEX::MOTOR_COMMAND_DIMENSION]
            desired_motor_angles = motor_commands[POSITION_INDEX::MOTOR_COMMAND_DIMENSION]
            desired_motor_velocities = motor_commands[VELOCITY_INDEX::MOTOR_COMMAND_DIMENSION]
            additional_torques = motor_commands[TORQUE_INDEX::MOTOR_COMMAND_DIMENSION]
        
        # PD控制计算
        motor_torques = -1 * (kp * (motor_angle - desired_motor_angles)) - kd * (
            motor_velocity - desired_motor_velocities) + additional_torques
        
        motor_torques = self._strength_ratios * motor_torques
        
        # 应用扭矩限制
        if self._torque_limits is not None:
            if len(self._torque_limits) != len(motor_torques):
                raise ValueError("扭矩限制维度与电机数量不匹配")
            motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits, self._torque_limits)

        return motor_torques, motor_torques


# ==================== 简单机器人类 ====================
class SimpleRobot(object):
    """Lite3机器人的PyBullet仿真接口
    
    提供MPC控制器需要的完整接口，包括：
    - 关节控制
    - 状态读取
    - 运动学计算
    - 力映射
    """
    
    def __init__(self, pybullet_client, robot_uid, simulation_time_step):
        self.pybullet_client = pybullet_client
        self.time_step = simulation_time_step
        self.quadruped = robot_uid
        self.num_legs = NUM_LEGS
        self.num_motors = NUM_MOTORS
        
        self._BuildJointNameToIdDict()
        self._BuildUrdfIds()
        self._BuildMotorIdList()
        self.ResetPose()
        
        self._motor_enabled_list = [True] * self.num_motors
        self._step_counter = 0
        self._state_action_counter = 0
        self._motor_offset = np.array([0] * 12)
        self._motor_direction = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        
        self.ReceiveObservation()
        self._kp = self.GetMotorPositionGains()
        self._kd = self.GetMotorVelocityGains()
        
        # 从关节限制提取扭矩限制
        torque_limits = [JOINT_LIMITS[name]["effort"] for name in MOTOR_NAMES]
        
        self._motor_model = Lite3MotorModel(
            kp=self._kp,
            kd=self._kd,
            torque_limits=np.array(torque_limits),
            motor_control_mode=MOTOR_CONTROL_HYBRID
        )
        
        self._SettleDownForReset(reset_time=1.0)
        self._step_counter = 0

    def _BuildJointNameToIdDict(self):
        """构建关节名称到ID的映射"""
        num_joints = self.pybullet_client.getNumJoints(self.quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildUrdfIds(self):
        """构建URDF中的链接ID"""
        num_joints = self.pybullet_client.getNumJoints(self.quadruped)
        self._foot_link_ids = []
        self._hip_link_ids = []
        self._thigh_link_ids = []
        self._shank_link_ids = []
        
        # Lite3的关节顺序: FL, FR, HL, HR
        # 我们需要的足端顺序: FR, FL, RR(HR), RL(HL)
        foot_links_map = {}
        
        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
            link_name = joint_info[12].decode("UTF-8")
            
            # 识别足端链接（小腿的子链接）
            if "SHANK" in link_name:
                if "FL" in link_name:
                    foot_links_map['FL'] = i
                elif "FR" in link_name:
                    foot_links_map['FR'] = i
                elif "HL" in link_name:
                    foot_links_map['HL'] = i
                elif "HR" in link_name:
                    foot_links_map['HR'] = i
                self._shank_link_ids.append(i)
            
            if "HIP" in link_name and "THIGH" not in link_name:
                self._hip_link_ids.append(i)
            if "THIGH" in link_name:
                self._thigh_link_ids.append(i)
        
        # 按照MPC控制器期望的顺序: FR, FL, RR, RL
        # 注意: HL映射到RL, HR映射到RR
        self._foot_link_ids = [
            foot_links_map.get('FR', -1),  # FR
            foot_links_map.get('FL', -1),  # FL  
            foot_links_map.get('HR', -1),  # RR (URDF中是HR)
            foot_links_map.get('HL', -1),  # RL (URDF中是HL)
        ]
        
        # 过滤掉无效的ID
        self._foot_link_ids = [lid for lid in self._foot_link_ids if lid >= 0]

    def _BuildMotorIdList(self):
        """构建电机ID列表"""
        self._motor_id_list = [
            self._joint_name_to_id[motor_name]
            for motor_name in MOTOR_NAMES
        ]

    def ResetPose(self):
        """重置机器人姿态"""
        # 禁用所有关节的速度控制
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self.pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=joint_id,
                controlMode=self.pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
        
        # 设置初始关节角度
        for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
            angle = INIT_MOTOR_ANGLES[i]
            if "HipX" in name:
                angle += HIP_JOINT_OFFSET
            elif "HipY" in name:
                angle += UPPER_LEG_JOINT_OFFSET
            elif "Knee" in name:
                angle += KNEE_JOINT_OFFSET
            
            self.pybullet_client.resetJointState(
                self.quadruped,
                self._joint_name_to_id[name],
                angle,
                targetVelocity=0)

    def _SettleDownForReset(self, reset_time):
        """稳定到初始姿态"""
        self.ReceiveObservation()
        if reset_time <= 0:
            return
        for _ in range(500):
            self._StepInternal(
                INIT_MOTOR_ANGLES,
                motor_control_mode=MOTOR_CONTROL_POSITION)

    def _StepInternal(self, action, motor_control_mode=MOTOR_CONTROL_HYBRID):
        """执行一步仿真"""
        self.ApplyAction(action, motor_control_mode)
        self.pybullet_client.stepSimulation()
        self.ReceiveObservation()
        self._step_counter += 1

    def Step(self, action):
        """执行仿真步骤（MPC控制器接口）"""
        for i in range(ACTION_REPEAT):
            self._StepInternal(action, motor_control_mode=MOTOR_CONTROL_HYBRID)

    def ApplyAction(self, motor_commands, motor_control_mode=None):
        """应用电机命令"""
        if motor_control_mode is None:
            motor_control_mode = MOTOR_CONTROL_HYBRID
        
        torques, _ = self._motor_model.convert_to_torque(
            motor_commands,
            self.GetMotorAngles(),
            self.GetMotorVelocities(),
            self.GetMotorVelocities(),
            motor_control_mode)
        
        # 先禁用所有电机的内置控制器(否则VELOCITY_CONTROL会干扰TORQUE_CONTROL)
        self.pybullet_client.setJointMotorControlArray(
            bodyIndex=self.quadruped,
            jointIndices=self._motor_id_list,
            controlMode=self.pybullet_client.VELOCITY_CONTROL,
            forces=[0] * self.num_motors)  # 设置force=0禁用速度控制
        
        # 应用扭矩控制
        self.pybullet_client.setJointMotorControlArray(
            bodyIndex=self.quadruped,
            jointIndices=self._motor_id_list,
            controlMode=self.pybullet_client.TORQUE_CONTROL,
            forces=torques)

    def ReceiveObservation(self):
        """接收观测值"""
        self._joint_states = self.pybullet_client.getJointStates(
            self.quadruped, self._motor_id_list)
        self._base_position, self._base_orientation = (
            self.pybullet_client.getBasePositionAndOrientation(self.quadruped))
        self._base_velocity, self._base_angular_velocity = (
            self.pybullet_client.getBaseVelocity(self.quadruped))

    def GetMotorAngles(self):
        """获取电机角度"""
        motor_angles = [state[0] for state in self._joint_states]
        motor_angles = np.multiply(
            np.asarray(motor_angles) - np.asarray(self._motor_offset),
            self._motor_direction)
        return motor_angles

    def GetMotorVelocities(self):
        """获取电机速度"""
        motor_velocities = [state[1] for state in self._joint_states]
        motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities

    def GetMotorPositionGains(self):
        """获取电机位置增益"""
        return np.array([ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] * NUM_LEGS)
    
    def GetMotorVelocityGains(self):
        """获取电机速度增益"""
        return np.array([ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] * NUM_LEGS)
    
    def GetHipPositionsInBaseFrame(self):
        """获取髋关节在身体坐标系中的位置"""
        return _DEFAULT_HIP_POSITIONS
    
    def GetBasePosition(self):
        """获取基座位置"""
        return self._base_position
    
    def GetTrueBaseOrientation(self):
        """获取基座方向（四元数）"""
        return self._base_orientation
    
    def GetBaseRollPitchYaw(self):
        """获取基座姿态（欧拉角）"""
        return self.pybullet_client.getEulerFromQuaternion(self._base_orientation)
    
    def GetBaseVelocity(self):
        """获取基座线速度"""
        return self._base_velocity
    
    def GetBaseRollPitchYawRate(self):
        """获取基座角速度"""
        angular_velocity = self._base_angular_velocity
        orientation = self.GetTrueBaseOrientation()
        return self.TransformAngularVelocityToLocalFrame(angular_velocity, orientation)
    
    def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
        """将角速度从世界坐标系转换到机器人坐标系"""
        _, orientation_inversed = self.pybullet_client.invertTransform([0, 0, 0], orientation)
        relative_velocity, _ = self.pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self.pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)
    
    def GetTimeSinceReset(self):
        """获取重置后的时间"""
        return self._step_counter * self.time_step
    
    def ComputeJacobian(self, leg_id):
        """计算给定腿的雅可比矩阵"""
        assert len(self._foot_link_ids) == self.num_legs
        all_joint_angles = [state[0] for state in self._joint_states]
        zero_vec = [0] * len(all_joint_angles)
        jv, _ = self.pybullet_client.calculateJacobian(
            self.quadruped,
            self._foot_link_ids[leg_id],
            (0, 0, 0),
            all_joint_angles,
            zero_vec,
            zero_vec)
        jacobian = np.array(jv)
        return jacobian
    
    def MapContactForceToJointTorques(self, leg_id, contact_force):
        """将足端接触力映射到关节扭矩"""
        jv = self.ComputeJacobian(leg_id)
        all_motor_torques = np.matmul(contact_force, jv)
        motor_torques = {}
        
        # FIX: The URDF motor ordering does NOT match leg_id * motors_per_leg.
        # The Jacobian has nonzero columns at different indices for each leg:
        # Leg 0 (FR) -> Jac cols [9, 10, 11], Leg 1 (FL) -> [6, 7, 8],
        # Leg 2 (RR) -> [15, 16, 17], Leg 3 (RL) -> [12, 13, 14].
        LEG_TO_JAC_START = {0: 9, 1: 6, 2: 15, 3: 12}
        jac_start = LEG_TO_JAC_START[leg_id]
        motors_per_leg = self.num_motors // self.num_legs
        
        for i, joint_id in enumerate(range(leg_id * motors_per_leg, (leg_id + 1) * motors_per_leg)):
            jac_index = jac_start + i
            if jac_index < len(all_motor_torques):
                motor_torques[joint_id] = all_motor_torques[jac_index] * self._motor_direction[joint_id]
            else:
                # Safety: if index out of bounds, set torque to zero
                motor_torques[joint_id] = 0.0
        
        return motor_torques
    
    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id, foot_local_position):
        """使用IK计算给定足端位置的电机角度"""
        return self._EndEffectorIK(leg_id, foot_local_position, position_in_world_frame=False)
    
    def _EndEffectorIK(self, leg_id, position, position_in_world_frame):
        """根据末端执行器位置计算关节位置"""
        assert len(self._foot_link_ids) == self.num_legs
        toe_id = self._foot_link_ids[leg_id]
        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = [
            i for i in range(leg_id * motors_per_leg, leg_id * motors_per_leg + motors_per_leg)
        ]
        
        if not position_in_world_frame:
            base_position = self.GetBasePosition()
            base_orientation = self.GetTrueBaseOrientation()
            world_link_pos, _ = self.pybullet_client.multiplyTransforms(
                base_position, base_orientation, position, _IDENTITY_ORIENTATION)
        else:
            world_link_pos = position
        
        ik_solver = 0
        all_joint_angles = self.pybullet_client.calculateInverseKinematics(
            self.quadruped, toe_id, world_link_pos, solver=ik_solver)
        
        joint_angles = [all_joint_angles[i] for i in joint_position_idxs]
        joint_angles = np.multiply(
            np.asarray(joint_angles) - np.asarray(self._motor_offset)[joint_position_idxs],
            self._motor_direction[joint_position_idxs])
        
        return joint_position_idxs, joint_angles.tolist()
    
    def GetFootContacts(self):
        """获取足端接触状态"""
        all_contacts = self.pybullet_client.getContactPoints(bodyA=self.quadruped)
        contacts = [False] * self.num_legs
        for contact in all_contacts:
            # 检查接触的链接ID是否是足端
            if contact[3] in self._foot_link_ids:
                leg_id = self._foot_link_ids.index(contact[3])
                contacts[leg_id] = True
        return contacts
    
    def GetFootPositionsInBaseFrame(self):
        """获取足端在基座坐标系中的位置"""
        foot_positions = []
        for foot_link_id in self._foot_link_ids:
            link_state = self.pybullet_client.getLinkState(self.quadruped, foot_link_id)
            world_pos = link_state[0]
            base_pos = self.GetBasePosition()
            base_orn = self.GetTrueBaseOrientation()
            _, base_orn_inv = self.pybullet_client.invertTransform([0, 0, 0], base_orn)
            local_pos, _ = self.pybullet_client.multiplyTransforms(
                [-base_pos[0], -base_pos[1], -base_pos[2]],
                base_orn_inv,
                world_pos,
                [0, 0, 0, 1])
            foot_positions.append(local_pos)
        # Convert to numpy array with shape (4, 3)
        return np.array(foot_positions)


# ==================== 调试信息 ====================
def print_robot_info():
    """打印机器人配置信息"""
    print("\n" + "="*60)
    print("Lite3 Robot Configuration")
    print("="*60)
    print(f"URDF文件: {URDF_NAME}")
    print(f"总质量: {MPC_BODY_MASS * 9.8:.3f} kg")
    print(f"身体惯量: {MPC_BODY_INERTIA[:3]}")
    print(f"腿数: {NUM_LEGS}")
    print(f"电机数: {NUM_MOTORS}")
    print(f"\n髋关节位置:")
    leg_names = ['FR', 'FL', 'RR', 'RL']
    for i, pos in enumerate(_DEFAULT_HIP_POSITIONS):
        print(f"  {leg_names[i]}: {pos}")
    print(f"\n初始关节角度:")
    for i, (name, angle) in enumerate(zip(MOTOR_NAMES, INIT_MOTOR_ANGLES)):
        print(f"  {name}: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
    print(f"\nPD增益:")
    print(f"  外展关节: Kp={ABDUCTION_P_GAIN}, Kd={ABDUCTION_D_GAIN}")
    print(f"  髋关节: Kp={HIP_P_GAIN}, Kd={HIP_D_GAIN}")
    print(f"  膝关节: Kp={KNEE_P_GAIN}, Kd={KNEE_D_GAIN}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # 测试代码 - 不需要PyBullet
    print_robot_info()
    
    print("\n" + "="*60)
    print("配置测试（无PyBullet）")
    print("="*60)
    
    # 测试参数访问
    print(f"\n✓ 电机数量: {NUM_MOTORS}")
    print(f"✓ 腿数量: {NUM_LEGS}")
    print(f"✓ URDF路径: {URDF_NAME}")
    
    # 测试电机模型
    kp = np.array([ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] * NUM_LEGS)
    kd = np.array([ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] * NUM_LEGS)
    torque_limits = [JOINT_LIMITS[name]["effort"] for name in MOTOR_NAMES]
    
    motor_model = Lite3MotorModel(
        kp=kp,
        kd=kd,
        torque_limits=np.array(torque_limits),
        motor_control_mode=MOTOR_CONTROL_HYBRID
    )
    print(f"\n✓ 电机模型创建成功")
    
    # 测试混合模式命令
    test_command = []
    for i in range(NUM_MOTORS):
        test_command.extend([
            INIT_MOTOR_ANGLES[i],  # 目标位置
            kp[i],                 # Kp
            0.0,                   # 目标速度
            kd[i],                 # Kd
            0.0                    # 前馈扭矩
        ])
    
    test_angles = INIT_MOTOR_ANGLES
    test_velocities = np.zeros(NUM_MOTORS)
    
    torques, _ = motor_model.convert_to_torque(
        np.array(test_command),
        test_angles,
        test_velocities,
        test_velocities,
        MOTOR_CONTROL_HYBRID
    )
    
    print(f"✓ 扭矩计算成功")
    print(f"  扭矩范围: [{torques.min():.2f}, {torques.max():.2f}] Nm")
    
    print("\n" + "="*60)
    print("配置验证完成！")
    print("\n下一步:")
    print("1. 使用PyBullet加载URDF")
    print("2. 创建SimpleRobot实例")
    print("3. 集成到MPC控制器")
    print("="*60)