import numpy as np
from pybullet_controller import RobotController

robot = RobotController(robot_type='7_dof', time_step=1/1000.)
robot.createWorld(GUI=True)

    # 设置初始关节角度为 0
th_initial= np.array([0, -1.0, 1.0, -1.57, -1.57, -1.57, 0])
robot.setJointPosition(th_initial)
desired_pose = np.array([-0.10857937593446423, 0.4166151451748437, 0.467087828094798, -1.5700006464761673, 0.0007970376813502642, 1.5692036732274044])

robot.task_space_impedance_control(th_initial, desired_pose, controller_gain=100)