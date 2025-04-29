import numpy as np
from pybullet_controller import RobotController

robot = RobotController(robot_type='7_dof', time_step=1/240.)
robot.createWorld(GUI=True)

    # 设置初始关节角度为 0
init_joints = np.array([0, -1.0, 1.0, -1.57, -1.57, -1.57, 0])
robot.setJointPosition(init_joints)

    # 任务空间扭矩控制：在 X 方向做 0.1 m 幅值、1Hz 的正弦振荡，持续 5 秒
    # 其余 Y, Z 保持不变
robot.task_space_torque_control(final_time=1.2, Kp=300, amplitude=0.1, freq=1)