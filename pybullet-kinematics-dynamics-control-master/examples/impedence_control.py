
import numpy as np
from pybullet_controller import RobotController

robot = RobotController(robot_type='7_dof')
robot.createWorld(GUI=True)

# Impedence controller
# Input: numpy array of joint angles
thi = np.array([0, -1.0, 1.0, -1.57, -1.57, -1.57, 0]) # initial joint angles
thf = np.array([-1.5, -1.0, 1.0, -1.57, -1.57, -1.57, 0]) # final joint nagles
robot.setJointPosition(thi)
desired_pose = np.array([-0.10857937593446423, 0.6166151451748437, 0.667087828094798, -1.5700006464761673, 0.0007970376813502642, 1.5692036732274044]) #ur5-link
robot.impedenceController_null(thi, desired_pose, controller_gain=30)


