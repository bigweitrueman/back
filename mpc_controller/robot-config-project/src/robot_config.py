import numpy as np
import yaml
import os
try:
    from .urdf_parser import URDFParser
except ImportError:
    from urdf_parser import URDFParser

class RobotConfig:
    """机器人配置管理类"""
    
    def __init__(self, urdf_path, config_path=None):
        self.urdf_path = urdf_path
        self.config_path = config_path
        self.parser = URDFParser(urdf_path)
        self.config = {}
        
    def load_urdf(self):
        """加载并解析URDF文件"""
        try:
            self.parser.parse()
            self._extract_parameters()
            return True
        except Exception as e:
            print(f"Error loading URDF: {e}")
            return False
    
    def _extract_parameters(self):
        """从URDF中提取参数"""
        # 基座参数
        base_name, base_link = self.parser.get_base_link()
        if base_link:
            self.config['body_mass'] = base_link['mass']
            self.config['body_inertia'] = base_link['inertia']
        else:
            # 使用总质量
            self.config['body_mass'] = self.parser.get_total_mass()
            self.config['body_inertia'] = np.array([0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1])
        
        # 电机关节
        motor_joints = self.parser.get_motor_joints()
        self.config['motor_names'] = list(motor_joints.keys())
        self.config['num_motors'] = len(motor_joints)
        
        # 关节限制
        self.config['joint_limits'] = {}
        for name, joint in motor_joints.items():
            self.config['joint_limits'][name] = {
                'lower': joint['limits']['lower'],
                'upper': joint['limits']['upper'],
                'effort': joint['limits']['effort'],
                'velocity': joint['limits']['velocity']
            }
        
        # 腿部配置
        legs = self.parser.get_leg_joints()
        self.config['leg_joints'] = legs
        self.config['num_legs'] = sum(1 for joints in legs.values() if joints)
        
        # 计算髋关节位置（从URDF中提取）
        self.config['hip_positions'] = self._calculate_hip_positions(legs)
        
        # 初始关节角度（从URDF限制中间值）
        self.config['init_motor_angles'] = self._calculate_init_angles(motor_joints)
    
    def _calculate_hip_positions(self, legs):
        """计算髋关节位置"""
        hip_positions = []
        
        for leg_name in ['FR', 'FL', 'RR', 'RL']:
            if leg_name in legs and legs[leg_name]:
                # 获取第一个关节（通常是髋关节）
                first_joint_name, first_joint = legs[leg_name][0]
                xyz = first_joint['origin_xyz']
                hip_positions.append(tuple(xyz))
            else:
                # 默认值
                default_positions = {
                    'FR': (0.2, -0.13, 0),
                    'FL': (0.2, 0.13, 0),
                    'RR': (-0.2, -0.13, 0),
                    'RL': (-0.2, 0.13, 0)
                }
                hip_positions.append(default_positions.get(leg_name, (0, 0, 0)))
        
        return tuple(hip_positions)
    
    def _calculate_init_angles(self, motor_joints):
        """计算初始关节角度（使用限制的中间值）"""
        init_angles = []
        for name, joint in motor_joints.items():
            lower = joint['limits']['lower']
            upper = joint['limits']['upper']
            # 使用中间值，但偏向于0
            mid = (lower + upper) / 2
            if abs(mid) < 0.1:
                mid = 0
            init_angles.append(mid)
        return np.array(init_angles)
    
    def get_joint_limits(self):
        """获取关节限制"""
        return self.config.get('joint_limits', {})
    
    def get_initial_state(self):
        """获取初始状态"""
        return {
            'motor_angles': self.config.get('init_motor_angles'),
            'motor_velocities': np.zeros(self.config.get('num_motors', 12)),
            'base_position': np.array([0, 0, 0.3]),
            'base_orientation': np.array([0, 0, 0, 1])
        }
    
    def save_config(self, output_path):
        """保存配置到YAML文件"""
        # 转换numpy数组为列表
        config_to_save = {}
        for key, value in self.config.items():
            if isinstance(value, np.ndarray):
                config_to_save[key] = value.tolist()
            elif isinstance(value, tuple):
                config_to_save[key] = [list(item) if isinstance(item, tuple) else item 
                                       for item in value]
            else:
                config_to_save[key] = value
        
        with open(output_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False)
        
        print(f"Configuration saved to: {output_path}")
    
    def load_config(self, config_path):
        """从YAML文件加载配置"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 转换列表回numpy数组
        if 'body_inertia' in self.config:
            self.config['body_inertia'] = np.array(self.config['body_inertia'])
        if 'init_motor_angles' in self.config:
            self.config['init_motor_angles'] = np.array(self.config['init_motor_angles'])
    
    def print_config(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("Robot Configuration Summary")
        print("="*60)
        print(f"Body Mass: {self.config.get('body_mass', 0):.3f} kg")
        print(f"Number of Motors: {self.config.get('num_motors', 0)}")
        print(f"Number of Legs: {self.config.get('num_legs', 0)}")
        print(f"\nMotor Names:")
        for name in self.config.get('motor_names', []):
            print(f"  - {name}")
        print(f"\nHip Positions:")
        for i, pos in enumerate(self.config.get('hip_positions', [])):
            leg_names = ['FR', 'FL', 'RR', 'RL']
            print(f"  {leg_names[i]}: {pos}")
        print("="*60 + "\n")