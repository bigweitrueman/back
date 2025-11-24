"""机器人控制相关常量定义"""

# 控制模式
MOTOR_CONTROL_POSITION = 1
MOTOR_CONTROL_TORQUE = 2
MOTOR_CONTROL_HYBRID = 3
MOTOR_CONTROL_PWM = 4

# 电机命令维度
MOTOR_COMMAND_DIMENSION = 5

# 电机命令索引
POSITION_INDEX = 0
POSITION_GAIN_INDEX = 1
VELOCITY_INDEX = 2
VELOCITY_GAIN_INDEX = 3
TORQUE_INDEX = 4

# 默认仿真参数
DEFAULT_ACTION_REPEAT = 5
DEFAULT_SIMULATION_TIME_STEP = 0.001
DEFAULT_CONTROL_TIME_STEP = 0.005

# 默认PD增益（需要根据实际机器人调整）
DEFAULT_KP = 100.0
DEFAULT_KD = 2.0

# 身份四元数
IDENTITY_ORIENTATION = [0, 0, 0, 1]

# This file defines constants used throughout the project.

URDF_PATH = "C:\\Users\\10947\\Desktop\\URDF_model-main\\Lite3\\Lite3_urdf"
DEFAULT_JOINT_LIMITS = {
    "position": (-3.14, 3.14),  # Example limits in radians
    "velocity": (-1.0, 1.0)     # Example limits in radians per second
}
DEFAULT_MASS = 1.0  # Default mass in kg
DEFAULT_GRAVITY = 9.81  # Default gravity in m/s^2
TIME_STEP = 0.01  # Default time step for simulation in seconds