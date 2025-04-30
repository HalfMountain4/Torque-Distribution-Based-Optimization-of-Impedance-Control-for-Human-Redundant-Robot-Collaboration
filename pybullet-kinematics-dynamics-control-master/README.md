# Kinematics, dynamics and control of robotic systems in Pybullet

## How to run the code ?
To make the program easy to use, ```RobotController``` class has been written to perform all kind of simulations. The class has folllowing inputs:

**1.** _robot_type_: specify urdf file initials eg. if urdf file name is 'ur5.urdf', specify 'ur5'

**2.** _controllable_joints_: joint indices of controllable joints. If not specified, by default all joints indices except first joint (first joint is fixed joint between robot stand and base) 

**3.** _end-eff_index_: specify the joint indices for end-effector link. If not specified, by default the last controllable_joints is considered as end-effector joint

**4.** _time_Step_: time step for simulation

The example for doing inverse dynamic simulation using this class is shown below:
```
robot = RobotController(robot_type='ur5')
robot.createWorld(GUI=True)

# Inverse dynamic simulation
# Input: numpy array of joint angles
thi = np.array([0, 0, 0, 0, 0, 0]) # initial joint angles
thf = np.array([-1.5, -1.0, 1.0, -1.57, -1.57, -1.57]) # final joint nagles
robot.setJointPosition(thi)
robot.doInverseDynamics(thi, thf)
