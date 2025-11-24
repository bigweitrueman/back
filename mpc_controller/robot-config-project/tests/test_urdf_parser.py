"""测试URDF解析器并生成配置"""
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.urdf_parser import URDFParser
from src.robot_config import RobotConfig

def main():
    urdf_path = "C:\\Users\\10947\\Desktop\\URDF_model-main\\Lite3\\Lite3_urdf"
    
    print("="*60)
    print("Starting URDF Analysis for Lite3 Robot")
    print("="*60)
    
    # 1. 解析URDF
    parser = URDFParser(urdf_path)
    parser.parse()
    parser.print_summary()
    
    # 2. 生成配置
    config = RobotConfig(urdf_path)
    config.load_urdf()
    config.print_config()
    
    # 3. 保存配置
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    os.makedirs(output_dir, exist_ok=True)
    
    config_file = os.path.join(output_dir, 'lite3_params.yaml')
    config.save_config(config_file)
    
    print(f"\n✓ Configuration saved successfully!")
    print(f"  Location: {config_file}")
    print("\nNext steps:")
    print("  1. Review the generated configuration file")
    print("  2. Adjust PD gains based on your robot's characteristics")
    print("  3. Verify hip positions and initial joint angles")
    print("  4. Run unit tests: python -m pytest tests/")

if __name__ == "__main__":
    main()