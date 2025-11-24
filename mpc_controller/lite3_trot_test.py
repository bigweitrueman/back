"""
Lite3机器人Trot步态测试
使用简化的步态生成器进行trot步态测试
"""
import pybullet as p
import pybullet_data
import numpy as np
import time
import math

import uni_sim

class SimpleTrotGait:
    """简化的Trot步态生成器"""
    
    def __init__(self, stance_duration=0.3, duty_factor=0.5):
        """
        Args:
            stance_duration: 支撑相时长 (秒)
            duty_factor: 占空比 (支撑时间/总周期)
        """
        self.stance_duration = stance_duration
        self.duty_factor = duty_factor
        self.swing_duration = stance_duration / duty_factor - stance_duration
        self.period = stance_duration / duty_factor
        
        # Trot步态：对角腿同时摆动
        # FR和RL一组，FL和RR一组，相位差0.5
        self.phase_offset = [0.0, 0.5, 0.5, 0.0]  # FR, FL, RR, RL
        
        self.start_time = 0
        
    def reset(self, current_time):
        """重置步态"""
        self.start_time = current_time
        
    def get_leg_state(self, current_time):
        """获取各腿的状态
        
        Returns:
            list of bool: True=STANCE, False=SWING
        """
        elapsed = current_time - self.start_time
        leg_states = []
        
        for phase_offset in self.phase_offset:
            # 计算当前相位
            phase = math.fmod(elapsed + phase_offset * self.period, self.period) / self.period
            
            # duty_factor内为支撑相，否则为摆动相
            is_stance = phase < self.duty_factor
            leg_states.append(is_stance)
            
        return leg_states
    
    def get_normalized_phase(self, current_time):
        """获取各腿在当前状态内的归一化相位 [0, 1]
        
        Returns:
            list of float: 归一化相位
        """
        elapsed = current_time - self.start_time
        normalized_phases = []
        
        for phase_offset in self.phase_offset:
            # 计算当前相位
            phase = math.fmod(elapsed + phase_offset * self.period, self.period) / self.period
            
            if phase < self.duty_factor:
                # 支撑相
                normalized = phase / self.duty_factor
            else:
                # 摆动相
                normalized = (phase - self.duty_factor) / (1 - self.duty_factor)
            
            normalized_phases.append(normalized)
            
        return normalized_phases

def compute_swing_foot_position(hip_position, phase, max_forward_reach=0.15, step_height=0.08):
    """计算摆动腿足端位置（简化版）
    
    Args:
        hip_position: 髋关节位置 (x, y, z)
        phase: 摆动相位 [0, 1]
        max_forward_reach: 最大前伸距离
        step_height: 抬腿高度
    
    Returns:
        足端位置 (x, y, z)
    """
    # 水平方向：从后向前摆动
    x_offset = max_forward_reach * (2 * phase - 1)
    
    # 垂直方向：抛物线轨迹
    z_offset = -step_height * 4 * phase * (1 - phase)
    
    foot_pos = [
        hip_position[0] + x_offset,
        hip_position[1],
        hip_position[2] - 0.3 + z_offset  # 基准腿长0.3m
    ]
    
    return foot_pos

def compute_stance_foot_position(hip_position, phase, max_backward_reach=0.15):
    """计算支撑腿足端位置（简化版）
    
    Args:
        hip_position: 髋关节位置
        phase: 支撑相位 [0, 1]
        max_backward_reach: 最大后伸距离
    
    Returns:
        足端位置
    """
    # 水平方向：从前向后滑动
    x_offset = max_backward_reach * (1 - 2 * phase)
    
    foot_pos = [
        hip_position[0] + x_offset,
        hip_position[1],
        hip_position[2] - 0.3  # 固定腿长
    ]
    
    return foot_pos

def main():
    """主函数：Trot步态测试"""
    
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
    
    # 4. 创建Trot步态生成器
    print("创建Trot步态生成器...")
    gait = SimpleTrotGait(
        stance_duration=0.25,  # 支撑时长0.25秒
        duty_factor=0.6        # 占空比60%
    )
    
    # 获取机器人参数
    kp = robot.GetMotorPositionGains()
    kd = robot.GetMotorVelocityGains()
    hip_positions = robot.GetHipPositionsInBaseFrame()
    
    print(f"\n机器人信息:")
    print(f"  电机数量: {robot.num_motors}")
    print(f"  步态周期: {gait.period:.3f} 秒")
    print(f"  支撑时长: {gait.stance_duration:.3f} 秒")
    print(f"  摆动时长: {gait.swing_duration:.3f} 秒")
    print(f"\n电机控制参数:")
    print(f"  Kp值: {kp}")
    print(f"  Kd值: {kd}")
    print(f"  髋关节位置: {hip_positions}")
    print(f"\nLite3机器人配置:")
    print(f"  MPC_BODY_MASS: {uni_sim.MPC_BODY_MASS} kg")
    print(f"  MPC_BODY_HEIGHT: {uni_sim.MPC_BODY_HEIGHT} m")
    print(f"  INIT_MOTOR_ANGLES: {uni_sim.INIT_MOTOR_ANGLES}")
    print(f"  MOTOR_NAMES: {uni_sim.MOTOR_NAMES}")
    
    # 5. 先让机器人稳定站立
    print("\n稳定站立...")
    stand_commands = []
    for i in range(uni_sim.NUM_MOTORS):
        stand_commands.extend([
            uni_sim.INIT_MOTOR_ANGLES[i],
            kp[i],
            0.0,
            kd[i],
            0.0
        ])
    stand_commands = np.array(stand_commands)
    
    print(f"站立命令（前5维）: {stand_commands[:5]}")
    
    for iter_idx in range(1000):
        robot.ApplyAction(stand_commands, uni_sim.MOTOR_CONTROL_HYBRID)
        p.stepSimulation()
        robot.ReceiveObservation()
        
        # 每500步打印一次诊断信息
        if iter_idx % 500 == 0:
            base_pos = robot.GetBasePosition()
            motor_angles = robot.GetMotorAngles()
            motor_vels = robot.GetMotorVelocities()
            print(f"\n  稳定步 {iter_idx}:")
            print(f"    身体高度: {base_pos[2]:.4f} m")
            print(f"    电机角度范围: [{np.min(motor_angles):.4f}, {np.max(motor_angles):.4f}]")
            print(f"    电机速度范围: [{np.min(motor_vels):.4f}, {np.max(motor_vels):.4f}]")
            
            # 计算并打印PD力矩
            motor_torques_list = []
            for i in range(uni_sim.NUM_MOTORS):
                target_angle = uni_sim.INIT_MOTOR_ANGLES[i]
                actual_angle = motor_angles[i]
                actual_vel = motor_vels[i]
                
                torque = -kp[i] * (actual_angle - target_angle) - kd[i] * actual_vel
                motor_torques_list.append(torque)
            
            print(f"    PD力矩范围: [{np.min(motor_torques_list):.2f}, {np.max(motor_torques_list):.2f}] N·m")
        
        time.sleep(0.001)
    
    print("开始Trot步态...")
    start_time = time.time()
    gait.reset(0)  # 重置为0
    
    # 6. Trot步态控制循环
    step_count = 0
    max_steps = 5000  # 5秒
    
    try:
        while step_count < max_steps:
            # 使用实际经过的时间
            current_time = time.time() - start_time
            
            # 获取步态状态
            leg_states = gait.get_leg_state(current_time)
            normalized_phases = gait.get_normalized_phase(current_time)
            
            # 计算每条腿的目标关节角度
            target_angles = []
            
            for leg_id in range(4):
                hip_pos = hip_positions[leg_id]
                phase = normalized_phases[leg_id]
                is_stance = leg_states[leg_id]
                
                if is_stance:
                    # 支撑相：足端从前向后移动
                    foot_pos = compute_stance_foot_position(hip_pos, phase, max_backward_reach=0.08)
                else:
                    # 摆动相：足端抬起并向前摆动
                    foot_pos = compute_swing_foot_position(hip_pos, phase, max_forward_reach=0.08, step_height=0.05)
                
                # 使用IK计算关节角度
                try:
                    joint_ids, joint_angles = robot.ComputeMotorAnglesFromFootLocalPosition(leg_id, foot_pos)
                    target_angles.extend(joint_angles)
                except Exception as e:
                    # IK失败，使用默认角度
                    default_angles = uni_sim.INIT_MOTOR_ANGLES[leg_id*3:(leg_id+1)*3]
                    target_angles.extend(default_angles)
            
            # 创建混合模式命令
            motor_commands = []
            for i in range(uni_sim.NUM_MOTORS):
                motor_commands.extend([
                    target_angles[i],
                    kp[i],  # 使用完整的Kp增益(300/300/350)
                    0.0,
                    kd[i],  # 使用完整的Kd阻尼(3/5/5)
                    0.0
                ])
            
            # 应用控制
            robot.ApplyAction(np.array(motor_commands), uni_sim.MOTOR_CONTROL_HYBRID)
            
            # 获取实际施加的力矩(直接从motor_model计算)
            applied_torques, _ = robot._motor_model.convert_to_torque(
                np.array(motor_commands),
                robot.GetMotorAngles(),
                robot.GetMotorVelocities(),
                robot.GetMotorVelocities(),
                uni_sim.MOTOR_CONTROL_HYBRID)
            
            p.stepSimulation()
            robot.ReceiveObservation()
            
            # 打印状态
            if step_count % 200 == 0:
                base_pos = robot.GetBasePosition()
                base_vel = robot.GetBaseVelocity()
                motor_angles = robot.GetMotorAngles()
                motor_vels = robot.GetMotorVelocities()
                
                print(f"\n步 {step_count} (t={current_time:.2f}s):")
                print(f"  基座位置: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
                print(f"  基座速度: [{base_vel[0]:.3f}, {base_vel[1]:.3f}, {base_vel[2]:.3f}]")
                print(f"  腿状态: FR={['SWING','STANCE'][leg_states[0]]} FL={['SWING','STANCE'][leg_states[1]]} RR={['SWING','STANCE'][leg_states[2]]} RL={['SWING','STANCE'][leg_states[3]]}")
                
                # 打印前两条腿的详细命令(FR和FL)
                print(f"\n  FR腿(0-2)混合命令:")
                print(f"    目标角度: [{motor_commands[0]:.4f}, {motor_commands[5]:.4f}, {motor_commands[10]:.4f}]")
                print(f"    Kp值: [{motor_commands[1]:.1f}, {motor_commands[6]:.1f}, {motor_commands[11]:.1f}]")
                print(f"    目标速度: [{motor_commands[2]:.4f}, {motor_commands[7]:.4f}, {motor_commands[12]:.4f}]")
                print(f"    Kd值: [{motor_commands[3]:.1f}, {motor_commands[8]:.1f}, {motor_commands[13]:.1f}]")
                print(f"    前馈力矩: [{motor_commands[4]:.2f}, {motor_commands[9]:.2f}, {motor_commands[14]:.2f}] N·m")
                
                print(f"  FL腿(3-5)混合命令:")
                print(f"    目标角度: [{motor_commands[15]:.4f}, {motor_commands[20]:.4f}, {motor_commands[25]:.4f}]")
                print(f"    Kp值: [{motor_commands[16]:.1f}, {motor_commands[21]:.1f}, {motor_commands[26]:.1f}]")
                print(f"    前馈力矩: [{motor_commands[19]:.2f}, {motor_commands[24]:.2f}, {motor_commands[29]:.2f}] N·m")
                
                # 计算力矩统计
                motor_torques_list = []
                for i in range(uni_sim.NUM_MOTORS):
                    target_angle = target_angles[i]
                    actual_angle = motor_angles[i]
                    actual_vel = motor_vels[i]
                    kp_val = motor_commands[i*5 + 1]  # 从命令中提取Kp
                    kd_val = motor_commands[i*5 + 3]  # 从命令中提取Kd
                    ff_torque = motor_commands[i*5 + 4]  # 前馈力矩
                    
                    # 计算PD力矩
                    pd_torque = -kp_val * (actual_angle - target_angle) - kd_val * actual_vel
                    total_torque = pd_torque + ff_torque
                    motor_torques_list.append(total_torque)
                
                # 分析腿部姿态(以FR和FL为例)
                print(f"\n  腿部姿态分析:")
                print(f"    FR腿: HipX={motor_angles[0]:.3f}, HipY={motor_angles[1]:.3f}, Knee={motor_angles[2]:.3f}")
                print(f"    FL腿: HipX={motor_angles[3]:.3f}, HipY={motor_angles[4]:.3f}, Knee={motor_angles[5]:.3f}")
                
                print(f"\n  力矩统计:")
                print(f"    目标角度范围: [{np.min(target_angles):.4f}, {np.max(target_angles):.4f}]")
                print(f"    实际角度范围: [{np.min(motor_angles):.4f}, {np.max(motor_angles):.4f}]")
                print(f"    角度误差范围: [{np.min(np.array(target_angles) - motor_angles):.4f}, {np.max(np.array(target_angles) - motor_angles):.4f}]")
                print(f"    计算总力矩(PD+FF)范围: [{np.min(motor_torques_list):.2f}, {np.max(motor_torques_list):.2f}] N·m")
                print(f"    实际施加力矩范围: [{np.min(applied_torques):.2f}, {np.max(applied_torques):.2f}] N·m")
                print(f"    FR腿施加力矩: HipX={applied_torques[0]:.2f}, HipY={applied_torques[1]:.2f}, Knee={applied_torques[2]:.2f} N·m")
                print(f"    平均电机角度误差: {np.mean(np.abs(np.array(target_angles) - motor_angles)):.4f} rad")
            
            step_count += 1
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\n步态测试已停止")
    
    # 7. 清理
    print(f"\n总步数: {step_count}")
    final_pos = robot.GetBasePosition()
    print(f"最终位置: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"前进距离: {final_pos[0]:.3f} m")
    
    input("\n按回车键关闭...")
    p.disconnect()

if __name__ == "__main__":
    main()
