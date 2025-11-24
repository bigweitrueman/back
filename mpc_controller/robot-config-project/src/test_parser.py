import unittest
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.robot_config import RobotConfig
from src.urdf_parser import URDFParser

class TestURDFParser(unittest.TestCase):
    
    def setUp(self):
        self.urdf_path = "C:\\Users\\10947\\Desktop\\URDF_model-main\\Lite3\\Lite3_urdf"
        self.parser = URDFParser(self.urdf_path)
    
    def test_parse_urdf(self):
        """测试URDF解析"""
        result = self.parser.parse()
        self.assertTrue(result)
        self.assertGreater(len(self.parser.joints), 0)
        self.assertGreater(len(self.parser.links), 0)
    
    def test_get_motor_joints(self):
        """测试获取电机关节"""
        self.parser.parse()
        motor_joints = self.parser.get_motor_joints()
        self.assertIsInstance(motor_joints, dict)
        self.assertGreater(len(motor_joints), 0)
    
    def test_get_total_mass(self):
        """测试总质量计算"""
        self.parser.parse()
        total_mass = self.parser.get_total_mass()
        self.assertGreater(total_mass, 0)

class TestRobotConfig(unittest.TestCase):

    def setUp(self):
        self.urdf_path = "C:\\Users\\10947\\Desktop\\URDF_model-main\\Lite3\\Lite3_urdf"
        self.robot_config = RobotConfig(self.urdf_path)

    def test_load_urdf(self):
        """测试URDF加载"""
        self.assertTrue(self.robot_config.load_urdf())

    def test_get_joint_limits(self):
        """测试关节限制获取"""
        self.robot_config.load_urdf()
        joint_limits = self.robot_config.get_joint_limits()
        self.assertIsInstance(joint_limits, dict)
        self.assertGreater(len(joint_limits), 0)

    def test_initial_state(self):
        """测试初始状态"""
        self.robot_config.load_urdf()
        initial_state = self.robot_config.get_initial_state()
        self.assertIsNotNone(initial_state)
        self.assertIn('motor_angles', initial_state)
        self.assertIn('base_position', initial_state)
    
    def test_config_keys(self):
        """测试配置包含必要的键"""
        self.robot_config.load_urdf()
        required_keys = ['body_mass', 'motor_names', 'num_motors', 
                        'hip_positions', 'init_motor_angles']
        for key in required_keys:
            self.assertIn(key, self.robot_config.config)

if __name__ == '__main__':
    unittest.main()