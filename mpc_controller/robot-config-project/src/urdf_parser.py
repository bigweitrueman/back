import xml.etree.ElementTree as ET
import numpy as np
import os

class URDFParser:
    """解析URDF文件提取机器人参数"""
    
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.tree = None
        self.root = None
        self.joints = {}
        self.links = {}
        
    def parse(self):
        """解析URDF文件"""
        # 查找.urdf文件
        urdf_file = None
        if os.path.isdir(self.urdf_path):
            for root, dirs, files in os.walk(self.urdf_path):
                for file in files:
                    if file.endswith('.urdf'):
                        urdf_file = os.path.join(root, file)
                        break
                if urdf_file:
                    break
        else:
            urdf_file = self.urdf_path
            
        if not urdf_file or not os.path.exists(urdf_file):
            raise FileNotFoundError(f"URDF file not found in {self.urdf_path}")
        
        print(f"Parsing URDF file: {urdf_file}")
        self.tree = ET.parse(urdf_file)
        self.root = self.tree.getroot()
        
        self._parse_joints()
        self._parse_links()
        
        return True
    
    def _parse_joints(self):
        """解析所有关节"""
        for joint in self.root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            
            # 获取父链接和子链接
            parent = joint.find('parent')
            child = joint.find('child')
            parent_link = parent.get('link') if parent is not None else None
            child_link = child.get('link') if child is not None else None
            
            # 获取关节原点
            origin = joint.find('origin')
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin is not None:
                if origin.get('xyz'):
                    xyz = [float(x) for x in origin.get('xyz').split()]
                if origin.get('rpy'):
                    rpy = [float(x) for x in origin.get('rpy').split()]
            
            # 获取关节轴
            axis = joint.find('axis')
            axis_xyz = [1, 0, 0]
            if axis is not None and axis.get('xyz'):
                axis_xyz = [float(x) for x in axis.get('xyz').split()]
            
            # 获取关节限制
            limit = joint.find('limit')
            lower = -np.pi
            upper = np.pi
            effort = 100.0
            velocity = 10.0
            if limit is not None:
                lower = float(limit.get('lower', -np.pi))
                upper = float(limit.get('upper', np.pi))
                effort = float(limit.get('effort', 100.0))
                velocity = float(limit.get('velocity', 10.0))
            
            self.joints[joint_name] = {
                'type': joint_type,
                'parent': parent_link,
                'child': child_link,
                'origin_xyz': xyz,
                'origin_rpy': rpy,
                'axis': axis_xyz,
                'limits': {
                    'lower': lower,
                    'upper': upper,
                    'effort': effort,
                    'velocity': velocity
                }
            }
    
    def _parse_links(self):
        """解析所有链接"""
        for link in self.root.findall('link'):
            link_name = link.get('name')
            
            # 获取惯性参数
            inertial = link.find('inertial')
            mass = 0.0
            inertia = np.zeros(9)
            com = [0, 0, 0]
            
            if inertial is not None:
                mass_elem = inertial.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.get('value', 0.0))
                
                origin = inertial.find('origin')
                if origin is not None and origin.get('xyz'):
                    com = [float(x) for x in origin.get('xyz').split()]
                
                inertia_elem = inertial.find('inertia')
                if inertia_elem is not None:
                    ixx = float(inertia_elem.get('ixx', 0))
                    iyy = float(inertia_elem.get('iyy', 0))
                    izz = float(inertia_elem.get('izz', 0))
                    ixy = float(inertia_elem.get('ixy', 0))
                    ixz = float(inertia_elem.get('ixz', 0))
                    iyz = float(inertia_elem.get('iyz', 0))
                    inertia = np.array([ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz])
            
            self.links[link_name] = {
                'mass': mass,
                'inertia': inertia,
                'com': com
            }
    
    def get_motor_joints(self):
        """获取所有电机关节（revolute类型）"""
        motor_joints = {}
        for name, joint in self.joints.items():
            if joint['type'] == 'revolute':
                motor_joints[name] = joint
        return motor_joints
    
    def get_base_link(self):
        """获取基座链接"""
        # 通常是'base_link'或第一个链接
        if 'base_link' in self.links:
            return 'base_link', self.links['base_link']
        elif self.links:
            first_link = list(self.links.keys())[0]
            return first_link, self.links[first_link]
        return None, None
    
    def get_total_mass(self):
        """计算总质量"""
        return sum(link['mass'] for link in self.links.values())
    
    def get_leg_joints(self, leg_pattern=None):
        """根据命名模式获取腿部关节"""
        legs = {'FR': [], 'FL': [], 'RR': [], 'RL': []}
        
        for name, joint in self.joints.items():
            if joint['type'] != 'revolute':
                continue
            
            # 尝试识别腿部位置
            name_upper = name.upper()
            if 'FR' in name_upper or 'RIGHT_FRONT' in name_upper or name_upper.startswith('RF'):
                legs['FR'].append((name, joint))
            elif 'FL' in name_upper or 'LEFT_FRONT' in name_upper or name_upper.startswith('LF'):
                legs['FL'].append((name, joint))
            elif 'RR' in name_upper or 'RIGHT_REAR' in name_upper or name_upper.startswith('RH'):
                legs['RR'].append((name, joint))
            elif 'RL' in name_upper or 'LEFT_REAR' in name_upper or name_upper.startswith('LH'):
                legs['RL'].append((name, joint))
            elif 'HR' in name_upper or name_upper.startswith('HR'):
                legs['RR'].append((name, joint))
            elif 'HL' in name_upper or name_upper.startswith('HL'):
                legs['RL'].append((name, joint))
        
        return legs
    
    def print_summary(self):
        """打印URDF摘要信息"""
        print("\n" + "="*60)
        print("URDF Parser Summary")
        print("="*60)
        
        print(f"\nTotal Links: {len(self.links)}")
        print(f"Total Joints: {len(self.joints)}")
        
        motor_joints = self.get_motor_joints()
        print(f"Motor Joints (revolute): {len(motor_joints)}")
        
        print("\n--- Motor Joints ---")
        for name, joint in motor_joints.items():
            print(f"  {name}:")
            print(f"    Parent: {joint['parent']} -> Child: {joint['child']}")
            print(f"    Limits: [{joint['limits']['lower']:.3f}, {joint['limits']['upper']:.3f}]")
            print(f"    Max Effort: {joint['limits']['effort']:.2f}")
        
        base_name, base_link = self.get_base_link()
        if base_link:
            print(f"\n--- Base Link: {base_name} ---")
            print(f"  Mass: {base_link['mass']:.3f} kg")
            print(f"  Inertia: {base_link['inertia'][:3]}")
        
        print(f"\nTotal Robot Mass: {self.get_total_mass():.3f} kg")
        
        legs = self.get_leg_joints()
        print("\n--- Leg Configuration ---")
        for leg_name, joints in legs.items():
            if joints:
                print(f"  {leg_name}: {len(joints)} joints")
                for joint_name, _ in joints:
                    print(f"    - {joint_name}")
        
        print("\n" + "="*60)