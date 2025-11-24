"""简化的Lite3行走测试 - 不使用MPC,直接用PD控制"""
import numpy as np
import pybullet as p
import pybullet_data
import time
import uni_sim

def main():
    print("="*60)
    print("Lite3 Simple Walk Test - Pure PD Control")
    print("="*60)
    
    # 初始化PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(0.001)
    
    # 加载地面和机器人
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(uni_sim.URDF_NAME, uni_sim.START_POS)
    robot = uni_sim.SimpleRobot(p, robot_id, simulation_time_step=0.001)
    
    print(f"Robot Kp: {robot.GetMotorPositionGains()}")
    print(f"Robot Kd: {robot.GetMotorVelocityGains()}")
    
    # 1. 站立阶段
    print("\n阶段1: 站立 (2000步)")
    kp = robot.GetMotorPositionGains() * 2.0
    kd = robot.GetMotorVelocityGains() * 2.0
    stand_angles = uni_sim.INIT_MOTOR_ANGLES
    
    for step in range(2000):
        motor_commands = []
        for i in range(12):
            motor_commands.extend([stand_angles[i], kp[i], 0.0, kd[i], 0.0])
        robot.ApplyAction(np.array(motor_commands), uni_sim.MOTOR_CONTROL_HYBRID)
        p.stepSimulation()
        robot.ReceiveObservation()
        
        if step % 500 == 0:
            height = robot.GetBasePosition()[2]
            print(f"  步 {step}: 高度={height:.3f}m")
        time.sleep(0.001)
    
    final_height = robot.GetBasePosition()[2]
    print(f"站立完成, 高度={final_height:.3f}m")
    
    if final_height < 0.20:
        print("❌ 站立失败,退出")
        return
    
    # 2. 优化的Trot步态 - 平滑过渡 + 前进动作
    print("\n阶段2: 优化Trot步态 (稳定+前进)")
    print("策略: 平滑抬腿+前向迈步+后向蹬地")
    
    swing_duration = 150      # 摆动腿运动时长
    stance_duration = 50      # 过渡阶段（四足支撑）
    lift_height = 0.15        # 抬腿高度
    knee_bend = 0.1           # 膝关节弯曲
    stance_abduction = 0.05   # 支撑腿外展角度
    forward_reach = 0.15      # 摆动腿前伸幅度（增加前进距离）
    backward_push = 0.1       # 支撑腿后蹬幅度（产生推进力）
    
    # 平滑插值函数
    def smooth_transition(start, end, progress):
        """使用余弦插值实现平滑过渡"""
        t = (1 - np.cos(progress * np.pi)) / 2  # 0到1的平滑曲线
        return start + (end - start) * t
    
    for cycle in range(30):  # 增加到30个周期以前进更远
        print(f"\n步态周期 {cycle+1}/30:")
        
        # ========== 相位1: FR/RL摆动, FL/RR支撑 ==========
        print(f"  相位1: FR/RL摆动")
        for step in range(swing_duration):
            progress = step / swing_duration  # 0到1
            motor_commands = []
            
            # FR腿(0-2): 平滑抬起-前伸-落下
            hip_y = smooth_transition(-1.178, -1.178 + lift_height, progress if progress < 0.5 else 1 - progress)
            knee = smooth_transition(1.658, 1.658 - knee_bend, progress if progress < 0.5 else 1 - progress)
            # 摆动腿：完全反转
            hip_y_forward = smooth_transition(-1.178 + forward_reach, -1.178 - forward_reach, progress)
            motor_commands.extend([0.0, kp[0], 0.0, kd[0], 0.0])
            motor_commands.extend([hip_y_forward, kp[1], 0.0, kd[1], 0.0])
            motor_commands.extend([knee, kp[2], 0.0, kd[2], 0.0])
            
            # FL腿(3-5): 支撑+后蹬（反转）
            support_hip_y = smooth_transition(-1.178 - forward_reach, -1.178 + forward_reach, progress)
            motor_commands.extend([stance_abduction, kp[3], 0.0, kd[3], 0.0])
            motor_commands.extend([support_hip_y, kp[4], 0.0, kd[4], 0.0])
            motor_commands.extend([1.658, kp[5], 0.0, kd[5], 0.0])
            
            # RR腿(6-8): 支撑+后蹬
            motor_commands.extend([stance_abduction, kp[6], 0.0, kd[6], 0.0])
            motor_commands.extend([support_hip_y, kp[7], 0.0, kd[7], 0.0])
            motor_commands.extend([1.658, kp[8], 0.0, kd[8], 0.0])
            
            # RL腿(9-11): 平滑抬起-前伸-落下
            motor_commands.extend([0.0, kp[9], 0.0, kd[9], 0.0])
            motor_commands.extend([hip_y_forward, kp[10], 0.0, kd[10], 0.0])
            motor_commands.extend([knee, kp[11], 0.0, kd[11], 0.0])
            
            robot.ApplyAction(np.array(motor_commands), uni_sim.MOTOR_CONTROL_HYBRID)
            p.stepSimulation()
            robot.ReceiveObservation()
            time.sleep(0.001)
        
        # 过渡阶段：四足着地稳定
        for step in range(stance_duration):
            motor_commands = []
            for i in range(4):  # 四条腿相同姿态
                motor_commands.extend([0.0, kp[i*3], 0.0, kd[i*3], 0.0])
                motor_commands.extend([-1.178, kp[i*3+1], 0.0, kd[i*3+1], 0.0])
                motor_commands.extend([1.658, kp[i*3+2], 0.0, kd[i*3+2], 0.0])
            robot.ApplyAction(np.array(motor_commands), uni_sim.MOTOR_CONTROL_HYBRID)
            p.stepSimulation()
            robot.ReceiveObservation()
            time.sleep(0.001)
        
        pos = robot.GetBasePosition()
        vel = robot.GetBaseVelocity()
        rpy = robot.GetBaseRollPitchYaw()
        print(f"    位置=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), 姿态=({np.rad2deg(rpy[0]):.1f}°, {np.rad2deg(rpy[1]):.1f}°)")
        
        # ========== 相位2: FL/RR摆动, FR/RL支撑 ==========
        print(f"  相位2: FL/RR摆动")
        for step in range(swing_duration):
            progress = step / swing_duration
            motor_commands = []
            
            # FR腿(0-2): 支撑+后蹬（反转）
            support_hip_y = smooth_transition(-1.178 - forward_reach, -1.178 + forward_reach, progress)
            motor_commands.extend([stance_abduction, kp[0], 0.0, kd[0], 0.0])
            motor_commands.extend([support_hip_y, kp[1], 0.0, kd[1], 0.0])
            motor_commands.extend([1.658, kp[2], 0.0, kd[2], 0.0])
            
            # FL腿(3-5): 平滑抬起-前伸-落下（反转）
            hip_y = smooth_transition(-1.178, -1.178 + lift_height, progress if progress < 0.5 else 1 - progress)
            knee = smooth_transition(1.658, 1.658 - knee_bend, progress if progress < 0.5 else 1 - progress)
            hip_y_forward = smooth_transition(-1.178 + forward_reach, -1.178 - forward_reach, progress)
            motor_commands.extend([0.0, kp[3], 0.0, kd[3], 0.0])
            motor_commands.extend([hip_y_forward, kp[4], 0.0, kd[4], 0.0])
            motor_commands.extend([knee, kp[5], 0.0, kd[5], 0.0])
            
            # RR腿(6-8): 平滑抬起-前伸-落下
            motor_commands.extend([0.0, kp[6], 0.0, kd[6], 0.0])
            motor_commands.extend([hip_y_forward, kp[7], 0.0, kd[7], 0.0])
            motor_commands.extend([knee, kp[8], 0.0, kd[8], 0.0])
            
            # RL腿(9-11): 支撑+后蹬
            motor_commands.extend([stance_abduction, kp[9], 0.0, kd[9], 0.0])
            motor_commands.extend([support_hip_y, kp[10], 0.0, kd[10], 0.0])
            motor_commands.extend([1.658, kp[11], 0.0, kd[11], 0.0])
            
            robot.ApplyAction(np.array(motor_commands), uni_sim.MOTOR_CONTROL_HYBRID)
            p.stepSimulation()
            robot.ReceiveObservation()
            time.sleep(0.001)
        
        # 过渡阶段
        for step in range(stance_duration):
            motor_commands = []
            for i in range(4):
                motor_commands.extend([0.0, kp[i*3], 0.0, kd[i*3], 0.0])
                motor_commands.extend([-1.178, kp[i*3+1], 0.0, kd[i*3+1], 0.0])
                motor_commands.extend([1.658, kp[i*3+2], 0.0, kd[i*3+2], 0.0])
            robot.ApplyAction(np.array(motor_commands), uni_sim.MOTOR_CONTROL_HYBRID)
            p.stepSimulation()
            robot.ReceiveObservation()
            time.sleep(0.001)
        
        pos = robot.GetBasePosition()
        vel = robot.GetBaseVelocity()
        rpy = robot.GetBaseRollPitchYaw()
        print(f"    位置=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), 姿态=({np.rad2deg(rpy[0]):.1f}°, {np.rad2deg(rpy[1]):.1f}°)")
    
    print("\n" + "="*60)
    final_pos = robot.GetBasePosition()
    print(f"测试完成!")
    print(f"最终位置: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
    print(f"前进距离: {final_pos[0]:.3f}m")
    print("="*60)
    
    input("\n按回车键关闭...")
    p.disconnect()

if __name__ == "__main__":
    main()
