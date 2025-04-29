import numpy as np
from pybullet_controller import RobotController

robot = RobotController(robot_type='7_dof', time_step=1/10000.)
robot.createWorld(GUI=True)

# 设置初始关节角度（7个关节）
th_initial = np.zeros(7)
robot.setJointPosition(th_initial)

    # 任务空间扭矩控制：在 X 方向做 0.1 m 幅值、1Hz 的正弦振荡，持续 5 秒
    # 其余 Y, Z 保持不变
robot.joint_space_impedance_control(
    th_initial=th_initial,
    final_time=1,
    K_val=10,
    amplitude=0.2,
    freq=0.5
)