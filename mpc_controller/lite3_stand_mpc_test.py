"""测试MPC控制下的站立(不行走,纯站立平衡)"""
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

# 站立模式: 所有腿都是STANCE
_STANCE_DURATION_SECONDS = [1.0] * 4  # 长stance时间
_DUTY_FACTOR = [1.0] * 4  # 100% duty = 永远站立,不摆动
_INIT_PHASE_FULL_CYCLE = [0.0, 0.0, 0.0, 0.0]
_MAX_TIME_SECONDS = 5

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
)


def _setup_controller(robot):
    """Setup locomotion controller for standing (all legs in stance)"""
    desired_speed = (0, 0)
    desired_twisting_speed = 0
    
    # Create gait generator (all stance)
    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
        robot,
        stance_duration=_STANCE_DURATION_SECONDS,
        duty_factor=_DUTY_FACTOR,
        initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
        initial_leg_state=_INIT_LEG_STATE)
    
    # Create state estimator
    state_estimator = com_velocity_estimator.COMVelocityEstimator(
        robot, window_size=20)
    
    # Create swing leg controller (won't be used since all legs are stance)
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


def main():
    """Test standing with MPC - all legs in stance mode"""
    print("="*60)
    print("Lite3 MPC Standing Test (No Walking)")
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
    
    print(f"Robot body height target: {robot_sim.MPC_BODY_HEIGHT} m")
    
    # 设置站立姿态
    print("\n初始化站立姿态...")
    robot.ResetPose()
    
    # 用强PD控制站起来
    print("使用PD控制站立...")
    stand_kp = robot.GetMotorPositionGains() * 2.0
    stand_kd = robot.GetMotorVelocityGains() * 2.0
    stand_angles = robot_sim.INIT_MOTOR_ANGLES
    
    for step in range(2000):
        motor_commands = []
        for i in range(robot_sim.NUM_MOTORS):
            motor_commands.extend([
                stand_angles[i],
                stand_kp[i],
                0.0,
                stand_kd[i],
                0.0
            ])
        robot.ApplyAction(np.array(motor_commands), robot_sim.MOTOR_CONTROL_HYBRID)
        p.stepSimulation()
        robot.ReceiveObservation()
        
        if step % 500 == 0:
            height = robot.GetBasePosition()[2]
            print(f"  步 {step}: 高度={height:.3f}m")
    
    final_height = robot.GetBasePosition()[2]
    print(f"✓ 站立完成,高度={final_height:.3f}m\n")
    
    # Setup MPC controller
    print("设置MPC控制器(纯站立模式)...")
    controller = _setup_controller(robot)
    controller.reset()
    
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    print("\n开始MPC站立测试...")
    current_time = robot.GetTimeSinceReset()
    step_count = 0
    
    while current_time < _MAX_TIME_SECONDS:
        # Update controller
        controller.update()
        
        # Get action
        hybrid_action, info = controller.get_action()
        
        # Apply action to robot
        robot.Step(hybrid_action)
        
        # Print status
        step_count += 1
        if step_count % 500 == 0:
            base_pos = robot.GetBasePosition()
            motor_angles = robot.GetMotorAngles()
            
            # 计算实际施加的力矩
            motor_vels = robot.GetMotorVelocities()
            applied_torques, _ = robot._motor_model.convert_to_torque(
                hybrid_action,
                motor_angles,
                motor_vels,
                motor_vels,
                robot_sim.MOTOR_CONTROL_HYBRID)
            
            print(f"\n步 {step_count} (t={current_time:.2f}s):")
            print(f"  高度: {base_pos[2]:.3f}m")
            print(f"  位置: ({base_pos[0]:.3f}, {base_pos[1]:.3f})")
            print(f"  力矩范围: [{np.min(applied_torques):.2f}, {np.max(applied_torques):.2f}] N·m")
            
            # 打印每条腿的信息
            leg_names = ['FR', 'FL', 'RR', 'RL']
            for leg_idx in range(4):
                motor_start = leg_idx * 3
                leg_kp = [hybrid_action[i*5+1] for i in range(motor_start, motor_start+3)]
                leg_torques = applied_torques[motor_start:motor_start+3]
                print(f"  {leg_names[leg_idx]}: Kp=[{leg_kp[0]:5.1f},{leg_kp[1]:5.1f},{leg_kp[2]:5.1f}], "
                      f"力矩=[{leg_torques[0]:6.2f},{leg_torques[1]:6.2f},{leg_torques[2]:6.2f}]N·m")
        
        current_time = robot.GetTimeSinceReset()
    
    final_pos = robot.GetBasePosition()
    print("\n" + "="*60)
    print(f"测试完成! 最终高度: {final_pos[2]:.3f}m")
    print("="*60)


if __name__ == "__main__":
    main()
