import numpy as np
from pybullet_controller import RobotController

robot = RobotController(robot_type='7_dof', time_step=1 / 240.)
robot.createWorld(GUI=True)

# 设置初始关节角度
th_initial = np.zeros(7)  # 例如全0
robot.setJointPosition(th_initial)

# 以正弦轨迹进行关节空间扭矩控制
# 参数：初始角度，控制时长（秒），控制器增益，正弦波幅值，频率(Hz)
robot.joint_space_torque_control(th_initial, final_time=1.0, controller_gain=4000, amplitude=0.5, freq=0.5)