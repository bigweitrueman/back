"""分析URDF文件并生成配置"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from urdf_parser import URDFParser
from robot_config import RobotConfig

def main():
    urdf_path = "C:\\Users\\10947\\Desktop\\URDF_model-main\\Lite3\\Lite3_urdf"
    
    print("="*60)
    print("Starting URDF Analysis for Lite3 Robot")
    print("="*60)
    
    # 1. 解析URDF
    print("\n[1/3] Parsing URDF file...")
    parser = URDFParser(urdf_path)
    parser.parse()
    parser.print_summary()
    
    # 2. 生成配置
    print("\n[2/3] Generating robot configuration...")
    config = RobotConfig(urdf_path)
    config.load_urdf()
    config.print_config()
    
    # 3. 保存配置
    print("\n[3/3] Saving configuration files...")
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    config_file = os.path.join(config_dir, 'lite3_params.yaml')
    config.save_config(config_file)
    
    print(f"\n✓ Configuration saved successfully!")
    print(f"  Location: {config_file}")
    print("\nNext steps:")
    print("  1. Review the generated configuration file")
    print("  2. Adjust PD gains based on your robot's characteristics")
    print("  3. Verify hip positions and initial joint angles")
    print("  4. Use the config to complete uni_sim.py")

if __name__ == "__main__":
    main()
