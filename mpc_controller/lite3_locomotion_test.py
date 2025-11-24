from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import inspect

# Add parent directory to path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import scipy.interpolate
import numpy as np
import pybullet
import pybullet_data as pd
from pybullet_utils import bullet_client
import time

# Import required modules
import uni_sim as robot_sim
from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

# Trotting gait parameters (same as locomotion_controller_example.py)
_STANCE_DURATION_SECONDS = [0.25] * 4  # 减少到0.25s,加快步态循环
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_MAX_TIME_SECONDS = 10

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


def _generate_example_linear_angular_speed(t):
    """Creates an example speed profile based on time"""
    vx = 0.4 * robot_sim.MPC_VELOCITY_MULTIPLIER
    vy = 0.2 * robot_sim.MPC_VELOCITY_MULTIPLIER
    wz = 0.5 * robot_sim.MPC_VELOCITY_MULTIPLIER
    
    time_points = (0, 3, 6, 9, 12)
    speed_points = (
        (0, 0, 0, 0),      # Stand still
        (vx, 0, 0, 0),     # Move forward (修复了Raibert符号后恢复正值)
        (0, 0, 0, wz),     # Turn in place
        (0, -vy, 0, 0),    # Move sideways
        (0, 0, 0, 0)       # Stop
    )
    
    speed = scipy.interpolate.interp1d(
        time_points,
        speed_points,
        kind="previous",
        fill_value="extrapolate",
        axis=0)(t)
    
    return speed[0:3], speed[3]


def _setup_controller(robot):
    """Setup locomotion controller with all components"""
    desired_speed = (0, 0)
    desired_twisting_speed = 0
    
    # Create gait generator
    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
        robot,
        stance_duration=_STANCE_DURATION_SECONDS,
        duty_factor=_DUTY_FACTOR,
        initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
        initial_leg_state=_INIT_LEG_STATE)
    
    # Create state estimator
    state_estimator = com_velocity_estimator.COMVelocityEstimator(
        robot, window_size=20)
    
    # Create swing leg controller
    sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_height=robot_sim.MPC_BODY_HEIGHT,
        foot_clearance=0.01)
    
    # Create stance leg controller
    st_controller = torque_stance_leg_controller.TorqueStanceLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=robot_sim.MPC_BODY_HEIGHT,
        body_mass=robot_sim.MPC_BODY_MASS,
        body_inertia=robot_sim.MPC_BODY_INERTIA)
    
    # Create locomotion controller
    controller = locomotion_controller.LocomotionController(
        robot=robot,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        swing_leg_controller=sw_controller,
        stance_leg_controller=st_controller,
        clock=robot.GetTimeSinceReset)
    
    return controller


def _update_controller_params(controller, lin_speed, ang_speed):
    """Update controller parameters"""
    # 只在速度命令变化时打印
    if not hasattr(_update_controller_params, 'last_speed'):
        _update_controller_params.last_speed = (None, None)
    
    speed_changed = False
    if _update_controller_params.last_speed[0] is None or \
       not np.array_equal(lin_speed, _update_controller_params.last_speed[0]) or \
       ang_speed != _update_controller_params.last_speed[1]:
        print(f"[速度命令变化] lin_speed={lin_speed}, ang_speed={ang_speed}")
        _update_controller_params.last_speed = (lin_speed.copy(), ang_speed)
        speed_changed = True
    
    controller.swing_leg_controller.desired_speed = lin_speed
    controller.swing_leg_controller.desired_twisting_speed = ang_speed
    controller.stance_leg_controller.desired_speed = lin_speed
    controller.stance_leg_controller.desired_twisting_speed = ang_speed


def main():
    """Run Lite3 trot gait test using full MPC controller"""
    print("="*60)
    print("Lite3 Trot Gait Test - Using Full MPC Controller")
    print("="*60)
    
    # Setup PyBullet
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setAdditionalSearchPath(pd.getDataPath())
    
    # Physics parameters
    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.8)
    
    # Load ground plane
    p.loadURDF("plane.urdf")
    
    # Load robot
    print(f"Loading robot from: {robot_sim.URDF_NAME}")
    robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)
    robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=0.001)
    
    print(f"Robot parameters:")
    print(f"  Body mass: {robot_sim.MPC_BODY_MASS} kg")
    print(f"  Body height: {robot_sim.MPC_BODY_HEIGHT} m")
    print(f"  Body inertia: {robot_sim.MPC_BODY_INERTIA}")
    print(f"  Velocity multiplier: {robot_sim.MPC_VELOCITY_MULTIPLIER}")
    print(f"  Motor Kp: {robot.GetMotorPositionGains()}")
    print(f"  Motor Kd: {robot.GetMotorVelocityGains()}")
    print(f"  Motor names: {robot_sim.MOTOR_NAMES}")
    
    # 设置站立姿态(而不是趴下)
    print("\n设置站立姿态...")
    robot.ResetPose()  # 设置为初始站立角度
    
    # Let robot settle to standing position with active PD control
    print("稳定站立姿态 (使用强力PD控制维持站立)...")
    stand_kp = robot.GetMotorPositionGains() * 2.0  # 加倍增益,快速站起
    stand_kd = robot.GetMotorVelocityGains() * 2.0
    stand_angles = robot_sim.INIT_MOTOR_ANGLES
    
    for step in range(2000):  # 增加到2000步,确保完全站起
        # 创建站立命令(混合控制模式)
        motor_commands = []
        for i in range(robot_sim.NUM_MOTORS):
            motor_commands.extend([
                stand_angles[i],  # 目标角度
                stand_kp[i],      # Kp (加倍)
                0.0,              # 目标速度
                stand_kd[i],      # Kd (加倍)
                0.0               # 前馈力矩
            ])
        
        robot.ApplyAction(np.array(motor_commands), robot_sim.MOTOR_CONTROL_HYBRID)
        p.stepSimulation()
        robot.ReceiveObservation()
        
        # 每400步打印一次状态
        if step % 400 == 0:
            base_pos = robot.GetBasePosition()
            motor_angles = robot.GetMotorAngles()
            print(f"  稳定步 {step}: 高度={base_pos[2]:.4f}m, "
                  f"膝关节角度=[{motor_angles[2]:.3f}, {motor_angles[5]:.3f}, {motor_angles[8]:.3f}, {motor_angles[11]:.3f}]")
        
        time.sleep(0.001)
    
    # 检查是否真的站起来了
    final_height = robot.GetBasePosition()[2]
    if final_height < 0.20:
        print(f"⚠️ 警告: 机器人未能站起来,当前高度={final_height:.3f}m < 0.20m")
        print("   请检查PD增益或URDF配置!")
    else:
        print(f"✓ 机器人已站立,高度={final_height:.3f}m")
    
    # Setup controller
    print("\nSetting up MPC controller...")
    controller = _setup_controller(robot)
    
    # 检查reset前的高度
    height_before = robot.GetBasePosition()[2]
    print(f"Reset前高度: {height_before:.3f}m")
    
    controller.reset()
    
    # 检查reset后是否还站立
    for _ in range(100):
        p.stepSimulation()
        robot.ReceiveObservation()
    height_after = robot.GetBasePosition()[2]
    print(f"Reset后高度: {height_after:.3f}m")
    
    if height_after < 0.15:
        print(f"❌ 错误: Reset后机器人倒下了! 从{height_before:.3f}m降到{height_after:.3f}m")
        print("   这说明在reset()和第一个控制周期之间,没有力矩维持姿态")
    
    print("Gait parameters:")
    print(f"  Stance duration: {_STANCE_DURATION_SECONDS[0]}s")
    print(f"  Duty factor: {_DUTY_FACTOR[0]}")
    print(f"  Initial leg states: {_INIT_LEG_STATE}")
    
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    print("\nStarting simulation...")
    current_time = robot.GetTimeSinceReset()
    step_count = 0
    
    # 用于存储每一步的实际施加力矩
    last_hybrid_action = None
    
    # 在开始MPC控制前,先用强力PD维持姿态200步
    print("用PD控制维持站立姿态,等待MPC启动...")
    for _ in range(100):  # 减少到100步,加快启动
        motor_commands = []
        current_angles = robot.GetMotorAngles()
        for i in range(robot_sim.NUM_MOTORS):
            motor_commands.extend([
                current_angles[i],  # 维持当前角度
                stand_kp[i],        # 使用强力Kp
                0.0,
                stand_kd[i],
                0.0
            ])
        robot.ApplyAction(np.array(motor_commands), robot_sim.MOTOR_CONTROL_HYBRID)
        p.stepSimulation()
        robot.ReceiveObservation()
    
    height_before_mpc = robot.GetBasePosition()[2]
    print(f"MPC启动前高度: {height_before_mpc:.3f}m")
    
    while current_time < _MAX_TIME_SECONDS:
        # Get desired speed
        lin_speed, ang_speed = _generate_example_linear_angular_speed(current_time)
        
        # Update controller parameters
        _update_controller_params(controller, lin_speed, ang_speed)
        
        # Update controller (state estimation, gait generation)
        controller.update()
        
        # Get action (hybrid position + torque control)
        hybrid_action, info = controller.get_action()
        last_hybrid_action = hybrid_action
        
        # Apply action to robot
        robot.Step(hybrid_action)
        
        # Print detailed status
        step_count += 1
        if step_count % 200 == 0:  # 每200步=1秒输出一次,更容易看到速度变化趋势
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_uid)
            base_vel, base_ang_vel = p.getBaseVelocity(robot_uid)
            motor_angles = robot.GetMotorAngles()
            motor_vels = robot.GetMotorVelocities()
            
            # 获取腿状态
            gait_gen = controller.gait_generator
            leg_states = gait_gen.desired_leg_state
            leg_phases = gait_gen.normalized_phase
            leg_state_names = ['SWING' if s == gait_generator_lib.LegState.SWING else 'STANCE' 
                              for s in leg_states]
            
            # 计算实际施加的力矩
            applied_torques, _ = robot._motor_model.convert_to_torque(
                hybrid_action,
                motor_angles,
                motor_vels,
                motor_vels,
                robot_sim.MOTOR_CONTROL_HYBRID)
            
            # 提取混合命令中的Kp和目标角度(以FR腿为例)
            fr_target_angles = [hybrid_action[i*5] for i in range(3)]
            fr_kp = [hybrid_action[i*5+1] for i in range(3)]
            fr_torques = applied_torques[0:3]
            
            print(f"\n{'='*70}")
            print(f"步 {step_count} (t={current_time:.2f}s):")
            print(f"  基座: 位置=({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}), "
                  f"速度=({base_vel[0]:.2f}, {base_vel[1]:.2f}, {base_vel[2]:.2f})")
            print(f"  命令: 线速度=({lin_speed[0]:.2f}, {lin_speed[1]:.2f}, {lin_speed[2]:.2f}), 角速度={ang_speed:.2f}")
            print(f"  腿状态: FR={leg_state_names[0]}({leg_phases[0]:.2f}), FL={leg_state_names[1]}({leg_phases[1]:.2f}), "
                  f"RR={leg_state_names[2]}({leg_phases[2]:.2f}), RL={leg_state_names[3]}({leg_phases[3]:.2f})")
            
            # 打印所有4条腿的详细信息
            leg_names = ['FR', 'FL', 'RR', 'RL']
            for leg_idx in range(4):
                motor_start = leg_idx * 3
                leg_target = [hybrid_action[i*5] for i in range(motor_start, motor_start+3)]
                leg_actual = motor_angles[motor_start:motor_start+3]
                leg_kp = [hybrid_action[i*5+1] for i in range(motor_start, motor_start+3)]
                leg_torques = applied_torques[motor_start:motor_start+3]
                
                print(f"  {leg_names[leg_idx]}腿({leg_state_names[leg_idx]}): "
                      f"目标=[{leg_target[0]:6.3f},{leg_target[1]:7.3f},{leg_target[2]:7.3f}], "
                      f"实际=[{leg_actual[0]:6.3f},{leg_actual[1]:7.3f},{leg_actual[2]:7.3f}], "
                      f"Kp=[{leg_kp[0]:5.1f},{leg_kp[1]:5.1f},{leg_kp[2]:5.1f}], "
                      f"力矩=[{leg_torques[0]:6.2f},{leg_torques[1]:6.2f},{leg_torques[2]:6.2f}]N·m")
            
            print(f"  全部力矩范围: [{np.min(applied_torques):.2f}, {np.max(applied_torques):.2f}] N·m")
            print(f"  平均角度误差: {np.mean(np.abs(motor_angles - np.array([hybrid_action[i*5] for i in range(12)]))):.4f} rad")
            
            # 调试: 打印足端位置和速度
            foot_positions = robot.GetFootPositionsInBaseFrame()
            print(f"  足端位置(body frame): FR={foot_positions[0]}, FL={foot_positions[1]}")
            print(f"  实际移动方向: {'后退' if base_vel[0] < -0.01 else '前进' if base_vel[0] > 0.01 else '静止'} (vx={base_vel[0]:.3f})")
        
        current_time = robot.GetTimeSinceReset()
    
    print("\n" + "="*60)
    print("Simulation completed!")
    print("="*60)


if __name__ == "__main__":
    main()
