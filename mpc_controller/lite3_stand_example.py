"""
Lite3机器人MPC控制示例
演示如何使用uni_sim配置进行基本的站立控制
"""
import pybullet as p
import pybullet_data
import numpy as np
import time

import uni_sim

def main():
    """主函数：演示基本的站立控制"""
    
    # 1. 初始化PyBullet
    print("初始化PyBullet...")
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(0.001)
    
    # 加载地面
    plane_id = p.loadURDF("plane.urdf")
    
    # 2. 加载Lite3机器人
    print("加载Lite3机器人...")
    start_pos = uni_sim.START_POS
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
    robot_id = p.loadURDF(
        uni_sim.URDF_NAME,
        start_pos,
        start_orientation,
        useFixedBase=False
    )
    
    # 3. 创建机器人实例
    print("创建SimpleRobot实例...")
    robot = uni_sim.SimpleRobot(p, robot_id, simulation_time_step=0.001)
    
    print(f"\n机器人信息:")
    print(f"  电机数量: {robot.num_motors}")
    print(f"  腿数量: {robot.num_legs}")
    print(f"  初始高度: {robot.GetBasePosition()[2]:.3f} m")
    
    # 4. 准备控制命令（站立姿态）
    kp = robot.GetMotorPositionGains()
    kd = robot.GetMotorVelocityGains()
    
    # 站立姿态的目标角度（可以调整这些值来改变站立高度）
    stand_angles = uni_sim.INIT_MOTOR_ANGLES.copy()
    
    # 创建混合模式命令
    def create_motor_commands(target_angles, kp_gains, kd_gains):
        """创建混合模式电机命令"""
        commands = []
        for i in range(uni_sim.NUM_MOTORS):
            commands.extend([
                target_angles[i],  # 目标位置
                kp_gains[i],       # Kp增益
                0.0,               # 目标速度
                kd_gains[i],       # Kd增益
                0.0                # 前馈扭矩
            ])
        return np.array(commands)
    
    motor_commands = create_motor_commands(stand_angles, kp, kd)
    
    # 5. 运行控制循环
    print("\n开始控制循环...")
    print("按Ctrl+C停止\n")
    
    try:
        step_count = 0
        while True:
            # 应用控制命令
            robot.ApplyAction(motor_commands, uni_sim.MOTOR_CONTROL_HYBRID)
            
            # 步进仿真
            p.stepSimulation()
            
            # 更新观测
            robot.ReceiveObservation()
            
            # 每100步打印一次状态
            if step_count % 100 == 0:
                base_pos = robot.GetBasePosition()
                base_rpy = robot.GetBaseRollPitchYaw()
                motor_angles = robot.GetMotorAngles()
                
                print(f"步 {step_count}:")
                print(f"  基座位置: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
                print(f"  基座姿态: [{base_rpy[0]:.3f}, {base_rpy[1]:.3f}, {base_rpy[2]:.3f}] rad")
                
                # 计算角度误差
                angle_errors = np.abs(motor_angles - stand_angles)
                print(f"  最大角度误差: {angle_errors.max():.4f} rad")
                print(f"  平均角度误差: {angle_errors.mean():.4f} rad")
                print()
            
            step_count += 1
            time.sleep(0.001)  # 实时同步
            
    except KeyboardInterrupt:
        print("\n控制循环已停止")
    
    # 6. 清理
    p.disconnect()
    print("仿真已关闭")

if __name__ == "__main__":
    main()
