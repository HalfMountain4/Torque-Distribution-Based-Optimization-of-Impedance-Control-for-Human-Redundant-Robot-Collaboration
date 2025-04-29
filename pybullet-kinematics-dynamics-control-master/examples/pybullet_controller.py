
# Input:
# 1. robot_type: specify urdf file initials eg. if urdf file name is 'ur5.urdf', specify 'ur5'
# 2. controllable_joints: joint indices of controllable joints. If not specified, by default all joints indices except first joint (first joint is fixed joint between robot stand and base) 
# 3. end-eff_index: specify the joint indices for end-effector link. If not specified, by default the last controllable_joints is considered as end-effector joint
# 4. time_Step: time step for simulation

import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import cvxpy as cp
import itertools

def compute_jacobian_dot_analytic(robot_id, link_index, q_list, dq_list, delta=1e-6):
    """
    通过数值微分来计算雅可比矩阵的时间导数 (J_dot)。
    J_dot 大小为 (6×n)，但若只关心位置部分，可取前3行。
    """
    # 计算原始的雅可比
    J_lin, J_ang = p.calculateJacobian(robot_id, link_index, [0, 0, 0],
                                       q_list, [0] * len(q_list), [0] * len(q_list))
    J_lin = np.array(J_lin)  # (3×n)
    J_ang = np.array(J_ang)  # (3×n)
    J = np.vstack((J_lin, J_ang))  # (6×n)

    # 初始化 J_dot
    J_dot = np.zeros_like(J)

    # 对每个关节做微小扰动，计算数值差分
    for i in range(len(q_list)):
        q_perturbed = q_list.copy()
        q_perturbed[i] += delta
        J_lin_p, J_ang_p = p.calculateJacobian(robot_id, link_index, [0, 0, 0],
                                               q_perturbed, [0] * len(q_list), [0] * len(q_list))
        J_lin_p = np.array(J_lin_p)
        J_ang_p = np.array(J_ang_p)
        J_p = np.vstack((J_lin_p, J_ang_p))
        dJ_dqi = (J_p - J) / delta
        # 累加 dq_i * dJ/dq_i
        J_dot += dJ_dqi * dq_list[i]

    return J_dot

class RobotController:
    def __init__(self, robot_type = 'ur5', controllable_joints = None, end_eff_index = None, time_step = 1e-3):
        self.robot_type = robot_type
        self.robot_id = None
        self.num_joints = None
        self.controllable_joints = controllable_joints
        self.end_eff_index = end_eff_index
        self.time_step = time_step
    # function to initiate pybullet and engine and create world
    def createWorld(self, GUI=True, view_world=False):
        # load pybullet physics engine
        if GUI:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        GRAVITY = -9.8
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step, numSolverIterations=100, numSubSteps=10)
        p.setRealTimeSimulation(True)
        p.loadURDF("plane.urdf")

        #loading robot into the environment
        urdf_file = 'urdf/' + self.robot_type + '.urdf'
        self.robot_id = p.loadURDF(urdf_file, useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot_id) # Joints
        print('#Joints:',self.num_joints)
        if self.controllable_joints is None:
            self.controllable_joints = list(range(0, self.num_joints-1))
        print('#Controllable Joints:', self.controllable_joints)
        if self.end_eff_index is None:
            self.end_eff_index = self.controllable_joints[-1]+1
        print('#End-effector:', self.end_eff_index)

        if (view_world):
            while True:
                p.stepSimulation()
                time.sleep(self.time_step)

    # function to joint position, velocity and torque feedback
    def getJointStates(self):
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    # function for setting joint positions of robot
    def setJointPosition(self, position, kp=1.0, kv=1.0):
        print('Joint position controller')
        zero_vec = [0.0] * len(self.controllable_joints)
        p.setJointMotorControlArray(self.robot_id,
                                    self.controllable_joints,
                                    p.POSITION_CONTROL,
                                    targetPositions=position,
                                    targetVelocities=zero_vec,
                                    positionGains=[kp] * len(self.controllable_joints),
                                    velocityGains=[kv] * len(self.controllable_joints))
        for _ in range(1000): # to settle the robot to its position
            p.stepSimulation()        

    # function to solve forward kinematics
    def solveForwardPositonKinematics(self, joint_pos):
        print('Forward position kinematics')

        # get end-effector link state
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = eeState
        eePose = list(link_trn) + list(p.getEulerFromQuaternion(link_rot))
        print('End-effector pose:', eePose)
        return eePose

    def solve_ForwardPositionKinematics(self, joint_pos):
        print('Forward position kinematics')

        # Get end-effector link state
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = eeState

        # Return position & orientation (quaternion)
        return np.array(link_trn), np.array(link_rot)

    def computePoseError(self, q, xd):
        """ Compute position and orientation error using rotation matrices """

        # Get current end-effector position and orientation (quaternion)
        x, q_curr = self.solve_ForwardPositionKinematics(q)

        # Desired position and orientation (convert Euler angles to quaternion)
        q_des = p.getQuaternionFromEuler(xd[3:])

        # Compute position error
        pos_error = np.array(xd[:3]) - x

        # Convert quaternions to rotation matrices
        def quaternion_to_rotation_matrix(q):
            """ Converts a quaternion (x, y, z, w) to a 3x3 rotation matrix """
            return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)

        R_des = quaternion_to_rotation_matrix(q_des)  # Desired orientation matrix
        R_curr = quaternion_to_rotation_matrix(q_curr)  # Current orientation matrix

        # Compute relative rotation error: R_error = R_des * R_curr.T
        R_error = np.dot(R_des, R_curr.T)

        # Extract rotation error in axis-angle representation
        trace = np.trace(R_error)
        theta = np.arccos((trace - 1) / 2)  # Rotation error magnitude

        # Ensure numerical stability (avoid NaN due to floating-point precision)
        theta = np.clip(theta, -np.pi, np.pi)

        # Extract rotation axis from R_error
        if np.sin(theta) > 1e-6:  # To avoid division by zero
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(theta))
        else:
            axis = np.array([0, 0, 0])  # If no rotation, set axis to zero

        ori_error = theta * axis  # Convert to a 3D angular error vector

        print("Position Error:", pos_error)
        print("Orientation Error (Axis-Angle):", ori_error)

        return pos_error, ori_error
    # function to solve inverse kinematics
    def solveInversePositionKinematics(self, end_eff_pose):
        print('Inverse position kinematics')
        joint_angles =  p.calculateInverseKinematics(self.robot_id,
                                                    self.end_eff_index,
                                                    targetPosition=end_eff_pose[0:3],
                                                    targetOrientation=p.getQuaternionFromEuler(end_eff_pose[3:6]))
        print('Joint angles:', joint_angles)
        return joint_angles

    # function to get jacobian 
    def getJacobian(self, joint_pos):
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = eeState
        zero_vec = [0.0] * len(joint_pos)
        jac_t, jac_r = p.calculateJacobian(self.robot_id, self.end_eff_index, com_trn, list(joint_pos), zero_vec, zero_vec)
        J_t = np.asarray(jac_t)
        J_r = np.asarray(jac_r)
        J = np.concatenate((J_t, J_r), axis=0)
        #print('Jacobian:', J)
        return J

    # function to solve forward velocity kinematics
    def solveForwardVelocityKinematics(self, joint_pos, joint_vel):
        print('Forward velocity kinematics')
        J  = self.getJacobian(joint_pos)
        eeVelocity = J @ joint_vel
        print('End-effector velocity:', eeVelocity)
        return eeVelocity

    #function to solve inverse velocity kinematics
    def solveInverseVelocityKinematics(self, end_eff_velocity):
        print('Inverse velocity kinematics')
        joint_pos, _ , _ = self.getJointStates()
        J  = self.getJacobian(joint_pos)
        if len(self.controllable_joints) > 1:
            joint_vel = np.linalg.pinv(J) @ end_eff_velocity
        else:
            joint_vel = J.T @ end_eff_velocity
        print('Joint velcoity:', joint_vel)
        return joint_vel

    #function to do joint velcoity control
    def JointVelocityControl(self, joint_velocities, sim_time=2, max_force=200):
        print('Joint velocity controller')
        t=0
        while t<sim_time:
            p.setJointMotorControlArray(self.robot_id,
                                        self.controllable_joints,
                                        p.VELOCITY_CONTROL,
                                        targetVelocities=joint_velocities,
                                        forces = [max_force] * (len(self.controllable_joints)))
            p.stepSimulation()
            time.sleep(self.time_step)
            t += self.time_step

    #function to do joint velcoity control
    def endEffectorVelocityControl(self, end_eff_vel, sim_time=2, max_forc=200):
        print('End-effector velocity controller')
        t=0
        while t<sim_time:
            joint_velocities = self.solveInverseVelocityKinematics(end_eff_vel)
            self.JointVelocityControl(joint_velocities)
            p.stepSimulation()
            time.sleep(self.time_step)
            t += self.time_step

    # Function to define GUI sliders (name of the parameter,range,initial value)
    def TaskSpaceGUIcontrol(self, goal, max_limit = 3.14, min_limit = -3.14):
        xId = p.addUserDebugParameter("x", min_limit, max_limit, goal[0]) #x
        yId = p.addUserDebugParameter("y", min_limit, max_limit, goal[1]) #y
        zId = p.addUserDebugParameter("z", min_limit, max_limit, goal[2]) #z
        rollId = p.addUserDebugParameter("roll", min_limit, max_limit, goal[3]) #roll
        pitchId = p.addUserDebugParameter("pitch", min_limit, max_limit, goal[4]) #pitch
        yawId = p.addUserDebugParameter("yaw", min_limit, max_limit, goal[5]) # yaw
        return [xId, yId, zId, rollId, pitchId, yawId]

    def ForceGUIcontrol(self, forces, max_limit = 1.0, min_limit = -1.0):
        fxId = p.addUserDebugParameter("fx", min_limit, max_limit, forces[0]) #force along x
        fyId = p.addUserDebugParameter("fy", min_limit, max_limit, forces[1]) #force along y
        fzId = p.addUserDebugParameter("fz", min_limit, max_limit, forces[2]) #force along z
        mxId = p.addUserDebugParameter("mx", min_limit, max_limit, forces[3]) #moment along x
        myId = p.addUserDebugParameter("my", min_limit, max_limit, forces[4]) #moment along y
        mzId = p.addUserDebugParameter("mz", min_limit, max_limit, forces[5]) #moment along z
        return [fxId, fyId, fzId, mxId, myId, mzId]

    # function to read the value of task parameter
    def readGUIparams(self, ids):
        val1 = p.readUserDebugParameter(ids[0])
        val2 = p.readUserDebugParameter(ids[1])
        val3 = p.readUserDebugParameter(ids[2])
        val4 = p.readUserDebugParameter(ids[3])
        val5 = p.readUserDebugParameter(ids[4])
        val6 = p.readUserDebugParameter(ids[5])
        return np.array([val1, val2, val3, val4, val5, val6])

    # function to get desired joint trajectory
    def getTrajectory(self, thi, thf, tf, dt):
        desired_position, desired_velocity, desired_acceleration = [], [], []
        t = 0
        while t <= tf:
            th=thi+((thf-thi)/tf)*(t-(tf/(2*np.pi))*np.sin((2*np.pi/tf)*t))
            dth=((thf-thi)/tf)*(1-np.cos((2*np.pi/tf)*t))
            ddth=(2*np.pi*(thf-thi)/(tf*tf))*np.sin((2*np.pi/tf)*t)
            desired_position.append(th)
            desired_velocity.append(dth)
            desired_acceleration.append(ddth)
            t += dt
        desired_position = np.array(desired_position)
        desired_velocity = np.array(desired_velocity)
        desired_acceleration = np.array(desired_acceleration)
        return desired_position, desired_velocity, desired_acceleration 
    
    #function to calculate dynamic matrics: inertia, coriolis, gravity
    def calculateDynamicMatrices(self):
        joint_pos, joint_vel, _ = self.getJointStates()
        n_dof = len(self.controllable_joints)
        InertiaMatrix= np.asarray(p.calculateMassMatrix(self.robot_id, joint_pos))
        GravityMatrix = np.asarray(p.calculateInverseDynamics(self.robot_id, joint_pos, [0.0] * n_dof, [0.0] * n_dof))
        CoriolisMatrix = np.asarray(p.calculateInverseDynamics(self.robot_id, joint_pos, joint_vel, [0.0] * n_dof)) - GravityMatrix
        return InertiaMatrix, GravityMatrix, CoriolisMatrix

    # Function to simulate free fall under gravity
    def doFreeFall(self):
        p.setRealTimeSimulation(False)
        # Enable torque control
        p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                    p.VELOCITY_CONTROL, 
                                    forces=np.zeros(len(self.controllable_joints)))


        tau = [0.0] * len(self.controllable_joints) # for free fall under gravity
        while True:
            p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                        controlMode = p.TORQUE_CONTROL, 
                                        forces = tau)
            p.stepSimulation()
            time.sleep(self.time_step)
        p.disconnect()

    # Function to do inverse dynamics simulation
    def doInverseDynamics(self, th_initial, th_final, final_time=2):
        p.setRealTimeSimulation(False)
        # get the desired trajectory
        q_d, dq_d, ddq_d = self.getTrajectory(th_initial, th_final, tf=final_time, dt=self.time_step)
        traj_points = q_d.shape[0]
        print('#Trajectory points:', traj_points)

        # forward dynamics simulation loop
        # for turning off link and joint damping
        for link_idx in range(self.num_joints+1):
            p.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)

        # Enable torque control
        p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                    p.VELOCITY_CONTROL,
                                    forces=np.zeros(len(self.controllable_joints)))

        kd = 0.7 # from URDF file
        n = 0
        while n < traj_points:
            tau = p.calculateInverseDynamics(self.robot_id, list(q_d[n]), list(dq_d[n]), list(ddq_d[n]))
            # tau += kd * dq_d[n] #if joint damping is turned off, this torque will not be required
            # print(tau)
            
            # torque control  
            p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                        controlMode = p.TORQUE_CONTROL, 
                                        forces = tau)
            theta, _, _ = self.getJointStates()
            print('n:{}::th:{}'.format(n,theta))
            
            p.stepSimulation()
            time.sleep(self.time_step)
            n += 1
        print('Desired joint angles:', th_final)
        p.disconnect()

    # Function to do computed torque control
    def computedTorqueControl(self, th_initial, th_final, final_time=2, controller_gain=400):
        p.setRealTimeSimulation(False)
        # 获取期望轨迹
        q_d, dq_d, ddq_d = self.getTrajectory(th_initial, th_final, tf=final_time, dt=self.time_step)
        traj_points = q_d.shape[0]
        print('#Trajectory points:', traj_points)

        # 关闭链节和关节的阻尼
        for link_idx in range(self.num_joints + 1):
            p.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)

        # 启用扭矩控制（关闭自带的速度控制）
        p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                    p.VELOCITY_CONTROL,
                                    forces=np.zeros(len(self.controllable_joints)))

        Kp = controller_gain
        Kd = 2 * np.sqrt(Kp)
        n = 0

        # 初始化记录变量，用于存储每个采样时刻的关节角度和时间
        joint_history = []  # 存储每个时刻的关节角度（列表形式，每个元素为一个关节角度数组）
        time_history = []  # 存储对应的时间戳
        t = 0

        while n < traj_points:
            # 获取当前关节状态（返回值 q 为列表或数组）
            q, dq, _ = self.getJointStates()
            # 将当前关节角度记录下来
            joint_history.append(np.array(q))
            time_history.append(t)

            # PD 控制：计算位置和速度误差
            q_e = q_d[n] - np.asarray(q)
            dq_e = dq_d[n] - np.asarray(dq)
            # 期望加速度加上 PD 补偿
            aq = ddq_d[n] + Kp * q_e + Kd * dq_e

            # 计算逆动力学得到控制扭矩
            tau = p.calculateInverseDynamics(self.robot_id, list(q), list(dq), list(aq))

            # 扭矩控制
            p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=tau)

            print('n:{}::th:{}'.format(n, q))
            p.stepSimulation()
            time.sleep(self.time_step)
            n += 1
            t += self.time_step

        print('Desired joint angles:', th_final)

        # 仿真结束后，转换记录数据为 numpy 数组便于绘图
        joint_history = np.array(joint_history)  # shape: (num_steps, num_joints)

        # 绘制每个关节的轨迹

        num_joints = joint_history.shape[1]
        for j in range(num_joints):
            plt.figure()
            plt.plot(time_history, joint_history[:, j], label=f'Joint {j}')
            plt.xlabel('Time [s]')
            plt.ylabel('Joint angle [rad]')
            plt.title(f'Trajectory for Joint {j}')
            plt.legend()
            plt.grid(True)
            plt.show()

        p.disconnect()

    def joint_space_torque_control(self, th_initial, final_time=5, controller_gain=400, amplitude=0.5, freq=2.0):
        """
        跟踪正弦参考轨迹控制，生成参考轨迹和实际轨迹并绘图。

        参考轨迹定义为：
            q_ref(t) = th_initial + amplitude * sin(2*pi*freq*t)
        其一阶和二阶导数分别为：
            dq_ref(t) = amplitude * 2*pi*freq * cos(2*pi*freq*t)
            ddq_ref(t) = -amplitude * (2*pi*freq)**2 * sin(2*pi*freq*t)
        """
        import matplotlib.pyplot as plt

        p.setRealTimeSimulation(False)
        dt = self.time_step
        t_vec = np.arange(0, final_time, dt)
        traj_points = len(t_vec)
        num_joints = len(th_initial)

        # 生成参考轨迹：对每个关节都采用相同正弦波
        q_ref = np.array([th_initial + amplitude * np.sin(2 * np.pi * freq * t) for t in t_vec])
        dq_ref = np.array([amplitude * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t) for t in t_vec])
        ddq_ref = np.array([- amplitude * (2 * np.pi * freq) ** 2 * np.sin(2 * np.pi * freq * t) for t in t_vec])

        print('#Trajectory points:', traj_points)

        # 关闭链节和关节的阻尼，避免干扰控制
        for link_idx in range(self.num_joints + 1):
            p.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)

        # 启用扭矩控制（关闭自带速度控制）
        p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                    p.VELOCITY_CONTROL,
                                    forces=np.zeros(len(self.controllable_joints)))

        # 控制参数
        Kp = controller_gain
        Kd = 2 * np.sqrt(Kp)
        n = 0
        t = 0

        # 记录实际关节轨迹和时间
        actual_history = []  # 每个时刻的实际关节角度
        time_history = []  # 时间戳

        while n < traj_points:
            # 获取当前关节状态
            q, dq, _ = self.getJointStates()
            actual_history.append(np.array(q))
            time_history.append(t)

            # 计算PD误差（参考 - 当前）
            q_e = q_ref[n] - np.asarray(q)
            dq_e = dq_ref[n] - np.asarray(dq)
            # 计算期望加速度加上PD补偿
            aq = ddq_ref[n] + Kp * q_e + Kd * dq_e

            # 逆动力学计算扭矩
            tau = p.calculateInverseDynamics(self.robot_id, list(q), list(dq), list(aq))

            # 施加扭矩控制
            p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=tau)

            print(f"t: {t:.3f} s, Step: {n} :: q: {q}")
            p.stepSimulation()
            time.sleep(dt)
            n += 1
            t += dt

        # 仿真结束后，将实际轨迹转换为numpy数组
        actual_history = np.array(actual_history)  # shape: (traj_points, num_joints)

        # ---- 将所有关节的参考与实际轨迹绘制到同一个图中，使用子图 (subplots) ----
        ncols = 2
        nrows = math.ceil(num_joints / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), sharex=True)
        axes = axes.flatten()  # 将二维数组转为一维便于循环

        for j in range(num_joints):
            axes[j].plot(time_history, actual_history[:, j], 'b-', label=f'Actual Joint {j + 1}')
            axes[j].plot(time_history, q_ref[:, j], 'r--', label=f'Desired Joint {j + 1}')
            axes[j].set_ylabel('Position (rad)')
            axes[j].legend(loc='best')
            axes[j].grid(True)

        # 隐藏多余的子图（如果总数不是偶数的话）
        for ax in axes[num_joints:]:
            ax.set_visible(False)

        # 为底部子图设置x轴标签
        # 这里简单给最后一个子图设置x轴标签，也可以循环设置底部行的所有子图
        axes[-1].set_xlabel('Time [s]')

        plt.tight_layout()
        plt.show()

        p.disconnect()

    def task_space_torque_control(self,
                                  final_time=5,
                                  Kp=200,
                                  Kd=None,
                                  amplitude=0.1,
                                  freq=1.0):
        """
        任务空间扭矩控制示例：
          D_e(q) * p_ddot + C_e(q,q_dot) * p_dot + G_e(q) = u_e
          -> tau = J^T * u_e
        在 x 方向做正弦振荡，y,z 保持不变。
        若需 y,z 同时振荡，可自行扩展。
        """

        if Kd is None:
            Kd = 2 * np.sqrt(Kp)  # 临界阻尼

        p.setRealTimeSimulation(False)
        dt = self.time_step
        t_vec = np.arange(0, final_time, dt)

        # 获取末端初始位置 (x0,y0,z0)
        link_state = p.getLinkState(self.robot_id, self.end_eff_index, computeLinkVelocity=True)
        x0 = np.array(link_state[0])  # 末端位置

        # 只在 x 方向做振荡，y,z 保持恒定
        # p_d(t) = [ x0 + A sin(2*pi*f*t), y0, z0 ]
        # p_dot_d, p_ddot_d 同理
        def desired_position(t):
            return np.array([
                x0[0] + 0.1 * np.sin(2*np.pi*freq*t),  # x
                x0[1] + 0.05 * np.cos(2*np.pi*freq*t),  # y
                x0[2]                                # z 不变
            ])
        def desired_velocity(t):
            return np.array([
                0.1 * (2*np.pi*freq) * np.cos(2*np.pi*freq*t),
                -0.05 * (2*np.pi*freq) * np.sin(2*np.pi*freq*t),
                0.0
            ])
        def desired_accel(t):
            return np.array([
                -0.1 * (2*np.pi*freq)**2 * np.sin(2*np.pi*freq*t),
                -0.05 * (2*np.pi*freq)**2 * np.cos(2*np.pi*freq*t),
                0.0
            ])
        # 关闭链节和关节的阻尼
        def compute_jacobian_dot_analytic(robot_id, link_index, q_list, dq_list, delta=1e-6):
            """
            通过数值微分来计算雅可比矩阵的时间导数 (J_dot)。
            J_dot 大小为 (6×n)，但若只关心位置部分，可取前3行。
            """
            # 计算原始的雅可比
            J_lin, J_ang = p.calculateJacobian(robot_id, link_index, [0, 0, 0],
                                               q_list, [0] * len(q_list), [0] * len(q_list))
            J_lin = np.array(J_lin)  # (3×n)
            J_ang = np.array(J_ang)  # (3×n)
            J = np.vstack((J_lin, J_ang))  # (6×n)

            # 初始化 J_dot
            J_dot = np.zeros_like(J)

            # 对每个关节做微小扰动，计算数值差分
            for i in range(len(q_list)):
                q_perturbed = q_list.copy()
                q_perturbed[i] += delta
                J_lin_p, J_ang_p = p.calculateJacobian(robot_id, link_index, [0, 0, 0],
                                                       q_perturbed, [0] * len(q_list), [0] * len(q_list))
                J_lin_p = np.array(J_lin_p)
                J_ang_p = np.array(J_ang_p)
                J_p = np.vstack((J_lin_p, J_ang_p))
                dJ_dqi = (J_p - J) / delta
                # 累加 dq_i * dJ/dq_i
                J_dot += dJ_dqi * dq_list[i]

            return J_dot
        for link_idx in range(self.num_joints + 1):
            p.changeDynamics(self.robot_id, link_idx, linearDamping=0.0,
                             angularDamping=0.0, jointDamping=0.0)

        # 启用扭矩控制
        p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                    p.VELOCITY_CONTROL,
                                    forces=[0] * len(self.controllable_joints))

        # 记录
        actual_history = []
        desired_history = []
        time_history = []

        for step, t in enumerate(t_vec):
            # 1. 读取当前关节状态
            q_list, dq_list, _ = self.getJointStates()
            q_array = np.array(q_list)
            dq_array = np.array(dq_list)

            # 2. 计算末端实际位置和速度
            link_state = p.getLinkState(self.robot_id,
                                        self.end_eff_index,
                                        computeLinkVelocity=True)
            p_curr = np.array(link_state[0])  # (3,)
            p_dot_curr = np.array(link_state[6])  # (3,)  index=6为线速度

            # 3. 获取期望位置/速度/加速度
            p_des = desired_position(t)
            p_dot_des = desired_velocity(t)
            p_ddot_des = desired_accel(t)

            # 4. 计算误差 e, e_dot，并加上 PD 补偿
            e = p_des - p_curr
            e_dot = p_dot_des - p_dot_curr
            # 令 p_ddot_cmd = p_ddot_des + Kp*e + Kd*e_dot
            a_p = p_ddot_des + Kp * e + Kd * e_dot

            # 5. 计算关节空间的 M, C, g
            M, g, C = self.calculateDynamicMatrices()

            # 6. 计算雅可比 J_lin (3×n) 及其导数
            J_lin, J_ang = p.calculateJacobian(
                bodyUniqueId=self.robot_id,
                linkIndex=self.end_eff_index,
                localPosition=[0, 0, 0],
                objPositions=q_list,
                objVelocities=dq_list,
                objAccelerations=[0] * (self.num_joints-1)
            )
            J_lin = np.array(J_lin)  # (3×n)
            # 任务空间速度 p_dot = J_lin * q_dot
            J = J_lin  # 仅位置控制
            J_dot_full = compute_jacobian_dot_analytic(self.robot_id, self.end_eff_index, q_list, dq_list)
            # 只取前3行(线速度部分)作为 J_dot
            J_dot = J_dot_full[:3, :]

            # 7. 计算任务空间质量矩阵、科氏力项、重力项
            #    D_e(q) = J^-T M J^-1
            #    C_e(q,q_dot)*p_dot = J^-T [ C - M J^-1 dotJ q_dot ] + ...
            #    G_e(q) = J^-T g
            # 先算一些常用逆
            J_inv = np.linalg.pinv(J)  # (n×3)
            J_T_inv = np.linalg.pinv(J.T)  # (3×n)


            # 8. 由公式 D_e p_ddot + C_e p_dot + G_e = u_e
            #    其中 C_e p_dot = C_x * (p_dot), 这里 p_dot = J * dq
            #    => C_e p_dot = C_x * (J dq)
            #    所以
            #    u_e = M_x p_ddot_cmd + C_x (J dq) + G_x
            p_dot_curr_3 = J @ dq_array  # 3×1

            tau = M @ J_inv @ a_p - M @ J_inv @ J_dot @ J_inv @ p_dot_curr_3 + C + g

            # 10. 施加扭矩
            p.setJointMotorControlArray(self.robot_id,
                                        self.controllable_joints,
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=tau)

            # 记录数据
            actual_history.append(p_curr)
            desired_history.append(p_des)
            time_history.append(t)

            p.stepSimulation()
            time.sleep(dt)

        # 转为 numpy 便于绘图
        actual_history = np.array(actual_history)  # shape: (steps, 3)
        desired_history = np.array(desired_history)  # shape: (steps, 3)

        # 画图
        x_min, x_max = actual_history[:, 0].min(), actual_history[:, 0].max()
        y_min, y_max = actual_history[:, 1].min(), actual_history[:, 1].max()
        z_min, z_max = actual_history[:, 2].min(), actual_history[:, 2].max()

        # 对 x, y 轴，我们可以使用等间隔的 tick
        x_ticks = np.linspace(x_min, x_max, 6)  # 比如 6 个 tick
        y_ticks = np.linspace(y_min, y_max, 6)

        # 对 z 轴，设置较大的刻度间隔，例如 0.02
        tick_interval = 0.02
        z_ticks = np.arange(z_min, z_max + tick_interval, tick_interval)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(actual_history[:, 0], actual_history[:, 1], actual_history[:, 2],
                'b-', label='Actual')
        ax.plot(desired_history[:, 0], desired_history[:, 1], desired_history[:, 2],
                'r--', label='Desired')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(loc='best')
        ax.set_title('End-Effector Trajectory (Elliptical in XY)')

        # 固定 box aspect（可选）
        ax.set_box_aspect([1, 1, 1])

        # 设置 tick 刻度
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)

        plt.tight_layout()
        plt.show()

        # 固定XYZ轴标尺，使比例保持一致
        ax.set_box_aspect([1, 1, 10])

        plt.tight_layout()
        plt.show()

        p.disconnect()

    # Function to do impedence control in task space
    def impedenceController(self, th_initial, desired_pose, controller_gain=300):
        p.setRealTimeSimulation(False)
        # forward dynamics simulation loop
        # for turning off link and joint damping
        for link_idx in range(self.num_joints+1):
            p.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)

        # Enable torque control
        p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                    p.VELOCITY_CONTROL, 
                                    forces=np.zeros(len(self.controllable_joints)))

        kd = 0.7 # from URDF file
        Kp = np.diag([controller_gain, controller_gain, controller_gain, controller_gain/5, controller_gain/5, controller_gain/5])
        Kd = 2 * np.sqrt(Kp)
        Md = 0.05*np.eye(6)

        # Target position and velcoity
        # xd = np.array([0.4499998573193256, 0.1, 1.95035834701983, 0.0, 1.5707963267948966, 0.0]) #1-link
        # xd = np.array([1.3499995719791142, 0.2, 2.9510750145148816, 0.0, 1.5707963267948966, 0.0]) #2-link
        # xd = np.array([2.199999302511422, 0.3, 2.9517518416742643, 0.0, 1.5707963267948966, 0.0]) #3-link
        # xd = np.array([1.8512362079506117, 0.30000000000000004, 4.138665008474901, -0.0, 1.0000000496605894, -0.0]) #3-link
        # xd = np.array([0.10972055742719365, -0.716441307051838, 1.44670878280948, -1.5700006464761673, 0.0007970376813496536, -1.570796326772595]) #ur5-link
        # xd = np.array([0.6811421738723965, -0.24773390188802563, 1.44670878280948, -1.5700006464761678, 0.0007970376813495148, -0.5007963267725951]) #ur5-link
        # xd = np.array([-0.10857937593446423, 0.7166151451748437, 1.4467087828094798, -1.5700006464761673, 0.0007970376813502642, 1.5692036732274044]) #ur5-link
        xd = desired_pose
        dxd = np.zeros(6)

        # define GUI sliders
        xdGUIids = self.TaskSpaceGUIcontrol(goal=xd)
        ForceInitial = np.zeros(len(self.controllable_joints))
        ForceGUIids = self.ForceGUIcontrol(forces=ForceInitial, max_limit=10, min_limit=-10) 

        while True:
            # read GUI values
            xd = self.readGUIparams(xdGUIids) # task space goal
            F_ext = self.readGUIparams(ForceGUIids) # applied external forces

            # get current joint states
            q, dq, _ = self.getJointStates()

            # Error in task space
            x = self.solveForwardPositonKinematics(q)
            x_e = xd - x
            dx = self.solveForwardVelocityKinematics(q, dq)
            dx_e = dxd - dx

            # Task space dynamics
            # Jacobian    
            J = self.getJacobian(q)
            J_inv = np.linalg.pinv(J)
            # Inertia matrix in the joint space
            Mq, G, _ = self.calculateDynamicMatrices()
            # Inertia matrix in the task space
            Mx = np.dot(np.dot(np.transpose(J_inv), Mq), J_inv)
            # Force in task space
            Fx = np.dot(np.dot(np.linalg.inv(Md), Mx),(np.dot(Kp, x_e) + np.dot(Kd, dx_e)))
            # External Force applied
            F_w_ext = np.dot((np.dot(np.linalg.inv(Md), Mx) - np.eye(6)), F_ext)
            Fx += F_w_ext
            # Force in joint space
            Fq = np.dot(np.transpose(J),Fx) 

            # Controlled Torque
            tau = G + Fq
            # tau += kd * np.asarray(dq) # if joint damping is turned off, this torque will not be required
            # print('tau:', tau)
            
            # Activate torque control  
            p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                        controlMode = p.TORQUE_CONTROL, 
                                        forces = tau)

            p.stepSimulation()
            time.sleep(self.time_step)
        p.disconnect()


    def impedenceController_null(self, th_initial, desired_pose, controller_gain=300):
        p.setRealTimeSimulation(False)

        for link_idx in range(self.num_joints + 1):
            p.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)

        p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                    p.VELOCITY_CONTROL,
                                    forces=np.zeros(len(self.controllable_joints)))

        kd = 0.7
        Kp = np.diag([controller_gain] * 3 + [controller_gain] * 3)
        Kd = 2 * np.sqrt(Kp)
        Md = 0.01 * np.eye(6)

        K_n = np.diag([1, 1, 1, 1, 1, 1, 1])  # Null space stiffness
        D_n = np.diag([1, 1, 1, 1, 1, 1, 1])  # Null space damping

        xd = desired_pose
        dxd = np.zeros(6)
        xdGUIids = self.TaskSpaceGUIcontrol(goal=xd)
        ForceInitial = np.zeros(len(self.controllable_joints))
        ForceGUIids = self.ForceGUIcontrol(forces=ForceInitial, max_limit=10, min_limit=-10)

        while True:
            xd = self.readGUIparams(xdGUIids)
            F_ext = self.readGUIparams(ForceGUIids)

            q, dq, _ = self.getJointStates()
            x = self.solveForwardPositonKinematics(q)
            pos_error, ori_error = self.computePoseError(q, xd)
            # Combine position and orientation error into a single vector
            x_e = np.hstack((pos_error, ori_error))
            dx = self.solveForwardVelocityKinematics(q, dq)
            dx_e = dxd - dx

            J = self.getJacobian(q)
            J_inv = np.linalg.pinv(J)
            Mq, G, _ = self.calculateDynamicMatrices()
            Mx = J_inv.T @ Mq @ J_inv
            Fx = np.linalg.inv(Md) @ Mx @ (Kp @ x_e + Kd @ dx_e)
            F_w_ext = (np.linalg.inv(Md) @ Mx - np.eye(6)) @ F_ext
            Fx += F_w_ext
            Fq = J.T @ Fx

            # Compute Null Space Matrix
            N = np.eye(len(q)) - J.T @ np.linalg.pinv(J.T)
            q=np.array(q)
            dq=np.array(dq)
            # Compute tau_2
            q_d = np.array([1.05568186e+00,  1.14822409e+00,  1.22098330e+00, -1.03651786e+00, 1.71922942e-01, -4.40386023e-01,  1.87819088e-17]) # Use initial configuration as a desired joint configuration
            tau2 = -K_n @ (q - q_d) - D_n @ dq

            # Compute total torque
            tau = G + Fq + N @ tau2
            #tau = G + Fq
            p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=tau)
            print('q',q)
            print('tau',tau)
            p.stepSimulation()
            time.sleep(self.time_step)

        p.disconnect()

    def joint_space_impedance_control(self, th_initial, final_time=5, K_val=100, amplitude=0.2, freq=0.5):
        """
        利用阻抗控制跟踪正弦参考轨迹，生成参考轨迹和实际关节轨迹并绘图。

        参考轨迹（对于每个关节）：
            q_d(t) = th_initial + amplitude * sin(2*pi*freq*t)
            dq_d(t) = amplitude * 2*pi*freq * cos(2*pi*freq*t)
            ddq_d(t) = -amplitude * (2*pi*freq)**2 * sin(2*pi*freq*t)

        阻抗控制律：
            τ = τ_ff + τ_imp
         其中：
            τ_ff = M(q)*ddq_d + C(q, dq)*dq_d + g(q)
            τ_imp = K*(q_d - q) + D*(dq_d - dq)
        """
        p.setRealTimeSimulation(False)
        dt = self.time_step
        t_vec = np.arange(0, final_time, dt)
        traj_points = len(t_vec)
        num_joints = len(th_initial)

        # 生成参考轨迹（各关节均采用相同正弦波）
        q_ref = np.array([th_initial + amplitude * np.sin(2 * np.pi * freq * t) for t in t_vec])
        dq_ref = np.array([amplitude * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t) for t in t_vec])
        ddq_ref = np.array([-amplitude * (2 * np.pi * freq) ** 2 * np.sin(2 * np.pi * freq * t) for t in t_vec])

        print("#Trajectory points:", traj_points)

        # 设置各链节动力学参数：去除阻尼、设置最大关节速度
        total_joints = p.getNumJoints(self.robot_id)
        for link_idx in range(total_joints):
            p.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)

        # 关闭默认控制，使后续由力矩控制接管
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.VELOCITY_CONTROL,
            forces=np.zeros(len(self.controllable_joints))
        )

        # 定义阻抗控制增益矩阵
        D_val = 2 * np.sqrt(K_val)
        K_mat = np.eye(num_joints) * K_val
        D_mat = np.eye(num_joints) * D_val

        n = 0
        t = 0

        # 用于记录实际关节轨迹和时间
        actual_history = []  # 每个时刻的实际关节角度
        time_history = []  # 时间戳

        while n < traj_points:
            # 获取当前关节状态
            joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
            q_current = np.array([s[0] for s in joint_states])
            dq_current = np.array([s[1] for s in joint_states])
            actual_history.append(q_current)
            time_history.append(t)

            # 计算误差
            q_err = q_ref[n] - q_current
            dq_err = dq_ref[n] - dq_current
            ddq_d = [ddq_ref[n]] * len(self.controllable_joints)
            # 前馈补偿力矩（逆动力学计算）： M(q)*ddq_d + C(q, dq)*dq_d + g(q)
            tau_ff = p.calculateInverseDynamics(
                self.robot_id,
                list(q_current),
                list(dq_current),
                ddq_d
            )
            M = np.array(p.calculateMassMatrix(self.robot_id, list(q_current)))
            Cg = np.array(p.calculateInverseDynamics(self.robot_id, list(q_current), list(dq_current), [0] * 7)).reshape(7,1)
            g = np.array(p.calculateInverseDynamics(self.robot_id, list(q_current), [0] * 7, [0] * 7)).reshape(7, 1)
            # 合成最终控制力矩
            tau_ff = np.array(tau_ff)

            # 阻抗反馈力矩
            tau_imp = K_mat @ q_err + D_mat @ dq_err

            # 合成总扭矩
            tau = tau_ff + tau_imp
            #tau = np.array(tau_imp).reshape(7, 1) + M @ np.array(ddq_d).reshape(7, 1) + Cg
            #tau=np.array(tau_imp).reshape(7, 1)+g
            # 施加扭矩控制
            p.setJointMotorControlArray(
                self.robot_id,
                self.controllable_joints,
                controlMode=p.TORQUE_CONTROL,
                forces=tau
            )

            print(f"t: {t:.3f} s, Step: {n}, q: {q_current}")
            print('tau',tau)
            p.stepSimulation()
            time.sleep(dt)
            n += 1
            t += dt

        # 将记录的实际轨迹转换为 numpy 数组
        actual_history = np.array(actual_history)  # shape: (traj_points, num_joints)

        # ---- 将所有关节的参考与实际轨迹绘制到同一张图中（2列显示） ----
        ncols = 2
        nrows = math.ceil(num_joints / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        for j in range(num_joints):
            axes[j].plot(time_history, actual_history[:, j], 'b-', label=f'Actual Joint {j + 1}')
            axes[j].plot(time_history, q_ref[:, j], 'r--', label=f'Desired Joint {j + 1}')
            axes[j].set_ylabel('Position (rad)')
            axes[j].legend(loc='best')
            axes[j].grid(True)
        for ax in axes[num_joints:]:
            ax.set_visible(False)
        axes[-1].set_xlabel('Time [s]')

        plt.tight_layout()
        plt.show()

        p.disconnect()

    def task_space_impedance_control(self, th_initial, desired_pose, controller_gain=300):
        """
        任务空间阻抗控制（不忽略姿态控制），目标使末端任务空间位姿（[x,y,z,roll,pitch,yaw]）跟踪 desired_pose，
        同时利用 null space 投影使关节向 th_initial 收敛。
        控制律：
            先利用 compute_torque 按您原来方式计算 tau，
            再计算 tau_null = -K_n*(q - q_d) - D_n*q_dot，
            最终 tau = tau + N*tau_null，其中 N = I - J^T*pinv(J^T)
        """
        p.setRealTimeSimulation(False)
        dt = self.time_step
        total_joints = p.getNumJoints(self.robot_id)
        for link_idx in range(total_joints):
            p.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)
        p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                    p.VELOCITY_CONTROL,
                                    forces=np.zeros(len(self.controllable_joints)))

        # 任务空间刚度与阻尼（6×6对角矩阵，对位置与姿态均使用 controller_gain）
        K_d = np.diag([controller_gain] * 3 + [controller_gain/100] * 3)
        D_d = 2 * np.sqrt(K_d)
        # Null space 控制参数（7×7 对角矩阵）
        K_n = np.diag([0.1] * len(self.controllable_joints))
        D_n = np.diag([0.1] * len(self.controllable_joints))

        # 期望任务空间位姿：包含位置和姿态（单位： [x,y,z,roll,pitch,yaw]）
        xd = desired_pose
        xdGUIids = self.TaskSpaceGUIcontrol(goal=xd)
        p_dot_d = np.zeros((6, 1))
        p_ddot_des = np.zeros((6, 1))

        # 期望关节配置，用于 null space 收敛，采用 th_initial
        q_d = np.array(th_initial)

        # 获取非固定关节索引，用于发送控制命令
        total_joints_num = p.getNumJoints(self.robot_id)
        joint_indices = [i for i in range(total_joints_num) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]

        J_prev = np.zeros((6, len(self.controllable_joints)))
        ext_force_x_slider = p.addUserDebugParameter("Ext Force X", -100, 100, 0)
        ext_force_y_slider = p.addUserDebugParameter("Ext Force Y", -100, 100, 0)
        ext_force_z_slider = p.addUserDebugParameter("Ext Force Z", -100, 100, 0)

        def compute_jacobian_dot_analytic(robot_id, link_index, q_list, dq_list, delta=1e-6):
            """
            通过数值微分来计算雅可比矩阵的时间导数 (J_dot)。
            J_dot 大小为 (6×n)，但若只关心位置部分，可取前3行。
            """
            # 计算原始的雅可比
            J_lin, J_ang = p.calculateJacobian(robot_id, link_index, [0, 0, 0],
                                               q_list, [0] * len(q_list), [0] * len(q_list))
            J_lin = np.array(J_lin)  # (3×n)
            J_ang = np.array(J_ang)  # (3×n)
            J = np.vstack((J_lin, J_ang))  # (6×n)

            # 初始化 J_dot
            J_dot = np.zeros_like(J)

            # 对每个关节做微小扰动，计算数值差分
            for i in range(len(q_list)):
                q_perturbed = q_list.copy()
                q_perturbed[i] += delta
                J_lin_p, J_ang_p = p.calculateJacobian(robot_id, link_index, [0, 0, 0],
                                                       q_perturbed, [0] * len(q_list), [0] * len(q_list))
                J_lin_p = np.array(J_lin_p)
                J_ang_p = np.array(J_ang_p)
                J_p = np.vstack((J_lin_p, J_ang_p))
                dJ_dqi = (J_p - J) / delta
                # 累加 dq_i * dJ/dq_i
                J_dot += dJ_dqi * dq_list[i]

            return J_dot
        def compute_torque(q, q_dot, q_ddot, p_ddot_des, current_pos, current_quat, xd,
                           p_dot, p_dot_d, J, J_prev, M, C, g, K_d, D_d, time_step):
            """
            按照您原来的方式计算 tau：
              1. 通过 compute_pose_error 得到位置与姿态误差，
              2. 计算 6D 误差 e = -[pos_error; ori_error] 及其速度误差，
              3. 利用全维雅可比导数 J_dot（用有限差分计算）得到末端加速度：p_ddot = J*q_ddot + J_dot*q_dot，
              4. 最终控制律：
                     tau = M*J_inv*p_ddot_des - M*J_inv*J_dot*J_inv*p_dot + C + g - J^T*(K_d*e + D_d*e_dot)
            """
            pos_error, ori_error = self.computePoseError(q, xd)
            e = -np.hstack((pos_error, ori_error)).reshape(6,1)
            e_dot = np.array(p_dot - p_dot_d)
            J_inv = np.linalg.pinv(J)
            J_T_inv = np.linalg.pinv(J.T)
            # 这里将当前 q 与 q_dot 转为列表供 compute_jacobian_dot_analytic 使用
            # robot_id 全局变量（在主函数中定义）
            J_dot = compute_jacobian_dot_analytic(self.robot_id, 7, q_list, dq_list)
            p_ddot = np.array(J @ q_ddot + J_dot @ q_dot).reshape(6, 1)
            e_ddot = p_ddot - p_ddot_des
            C_x = np.array(J_T_inv @ C)
            H_x = - J_T_inv @ M @ J_inv @ J_dot @ J_inv
            M_x = J_T_inv @ M @ J_inv
            g_x = np.array(J_T_inv @ g)
            M_d = 0.01 * np.eye(6)
            f_x = M_x @ p_ddot_des + H_x @ p_dot + C_x + g_x - K_d @ e - D_d @ e_dot
            tau = M @ J_inv @ p_ddot_des - M @ J_inv @ J_dot @ J_inv @ p_dot + C + g - J.T @ (K_d @ e + D_d @ e_dot)

            return tau, J

        def get_joint_states_and_acceleration(robot_id, controlled_joints, time_step):
            """
            获取关节位置、速度以及利用有限差分计算的加速度，并返回列表与数组形式。
            """
            joint_states = [p.getJointState(robot_id, j) for j in controlled_joints]
            q_list = [s[0] for s in joint_states]
            dq_list = [s[1] for s in joint_states]
            q = np.array(q_list)
            q_dot = np.array(dq_list)
            if not hasattr(get_joint_states_and_acceleration, "prev_q_dot"):
                get_joint_states_and_acceleration.prev_q_dot = q_dot.copy()
            q_ddot = (q_dot - get_joint_states_and_acceleration.prev_q_dot) / time_step
            get_joint_states_and_acceleration.prev_q_dot = q_dot.copy()
            return q_list, dq_list, q, q_dot, q_ddot
        while True:
            # 获取当前关节状态
            xd = self.readGUIparams(xdGUIids)
            q_list, dq_list, q, q_dot, q_ddot = get_joint_states_and_acceleration(self.robot_id,
                                                                                  self.controllable_joints, dt)
            M = np.array(p.calculateMassMatrix(self.robot_id, q_list))

            g = np.array(p.calculateInverseDynamics(self.robot_id, q_list, [0] * 7, [0] * 7)).reshape(7, 1)

            Cg = np.array(p.calculateInverseDynamics(self.robot_id, q_list, dq_list, [0] * 7)).reshape(7, 1)
            C = Cg - g
            # 获取末端状态（使用末端 link 索引为 total_joints_num-1）
            ee_state = p.getLinkState(self.robot_id, total_joints_num - 1, computeLinkVelocity=1,
                                      computeForwardKinematics=1)
            pos = np.array(ee_state[0]).reshape(3, 1)
            quat = ee_state[1]
            # 计算任务空间雅可比（6×n）
            J_lin, J_ang = p.calculateJacobian(self.robot_id, total_joints_num - 1, [0, 0, 0],
                                               q_list, dq_list, [0] * len(q_list))
            J_lin = np.array(J_lin)
            J_ang = np.array(J_ang)
            J = np.vstack([J_lin, J_ang])
            p_dot = np.array(J @ q_dot).reshape(6, 1)

            # 按您原来方式计算 tau（含位置与姿态误差）

            tau, J_prev = compute_torque(q, q_dot, q_ddot, p_ddot_des, pos, quat, np.array(xd), p_dot, p_dot_d,
                                         J, J_prev, M, C, g, K_d, D_d, dt)
            # 计算 null space 补偿
            tau_null = np.array(-K_n @ (q - q_d) - D_n @ q_dot).reshape(len(self.controllable_joints), 1)
            N = np.eye(len(q)) - J.T @ np.linalg.pinv(J.T)
            #tau = tau + N @ tau_null
            ext_force_x = p.readUserDebugParameter(ext_force_x_slider)
            ext_force_y = p.readUserDebugParameter(ext_force_y_slider)
            ext_force_z = p.readUserDebugParameter(ext_force_z_slider)
            ext_force = [ext_force_x, ext_force_y, ext_force_z]
            # 将外力施加在末端执行器所在的连杆上，这里使用 p.LINK_FRAME 施加在连杆原点
            p.applyExternalForce(objectUniqueId=self.robot_id,
                                 linkIndex=total_joints_num - 1,
                                 forceObj=ext_force,
                                 posObj=pos ,
                                 flags=p.WORLD_FRAME)
            print("q:", q)
            print("tau:", tau.flatten())
            print("End-effector pose:", pos.flatten(), quat)
            p.setJointMotorControlArray(self.robot_id, joint_indices,
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=tau)
            p.stepSimulation()
            time.sleep(dt)
        p.disconnect()

    def impedance_control_opt(self, th_initial, desired_pose, controller_gain=300):
        """
        任务空间阻抗控制（不忽略姿态控制），目标使末端任务空间位姿（[x,y,z,roll,pitch,yaw]）跟踪 desired_pose，
        同时利用 null space 投影使关节向 th_initial 收敛。
        控制律：
            先利用 compute_torque 按您原来方式计算 tau，
            再计算 tau_null = -K_n*(q - q_d) - D_n*q_dot，
            最终 tau = tau + N*tau_null，其中 N = I - J^T*pinv(J^T)
        """
        p.setRealTimeSimulation(False)
        dt = self.time_step
        total_joints = p.getNumJoints(self.robot_id)
        for link_idx in range(total_joints):
            p.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot_id, link_idx, maxJointVelocity=200)
        p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
                                    p.VELOCITY_CONTROL,
                                    forces=np.zeros(len(self.controllable_joints)))

        # 任务空间刚度与阻尼（6×6对角矩阵，对位置与姿态均使用 controller_gain）
        K_d = np.diag([controller_gain] * 3 + [controller_gain/100] * 3)
        D_d = 2 * np.sqrt(K_d)
        # Null space 控制参数（7×7 对角矩阵）
        K_n = np.diag([0.1] * len(self.controllable_joints))
        D_n = np.diag([0.1] * len(self.controllable_joints))

        # 期望任务空间位姿：包含位置和姿态（单位： [x,y,z,roll,pitch,yaw]）
        xd = desired_pose
        xdGUIids = self.TaskSpaceGUIcontrol(goal=xd)
        p_dot_d = np.zeros((6, 1))
        p_ddot_des = np.zeros((6, 1))

        # 期望关节配置，用于 null space 收敛，采用 th_initial
        q_d = np.array(th_initial)

        # 获取非固定关节索引，用于发送控制命令
        total_joints_num = p.getNumJoints(self.robot_id)
        joint_indices = [i for i in range(total_joints_num) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]

        J_prev = np.zeros((6, len(self.controllable_joints)))
        ext_force_x_slider = p.addUserDebugParameter("Ext Force X", -100, 100, 0)
        ext_force_y_slider = p.addUserDebugParameter("Ext Force Y", -100, 100, 0)
        ext_force_z_slider = p.addUserDebugParameter("Ext Force Z", -100, 100, 0)

        def compute_jacobian_dot_analytic(robot_id, link_index, q_list, dq_list, delta=1e-6):
            """
            通过数值微分来计算雅可比矩阵的时间导数 (J_dot)。
            J_dot 大小为 (6×n)，但若只关心位置部分，可取前3行。
            """
            # 计算原始的雅可比
            J_lin, J_ang = p.calculateJacobian(robot_id, link_index, [0, 0, 0],
                                               q_list, [0] * len(q_list), [0] * len(q_list))
            J_lin = np.array(J_lin)  # (3×n)
            J_ang = np.array(J_ang)  # (3×n)
            J = np.vstack((J_lin, J_ang))  # (6×n)

            # 初始化 J_dot
            J_dot = np.zeros_like(J)

            # 对每个关节做微小扰动，计算数值差分
            for i in range(len(q_list)):
                q_perturbed = q_list.copy()
                q_perturbed[i] += delta
                J_lin_p, J_ang_p = p.calculateJacobian(robot_id, link_index, [0, 0, 0],
                                                       q_perturbed, [0] * len(q_list), [0] * len(q_list))
                J_lin_p = np.array(J_lin_p)
                J_ang_p = np.array(J_ang_p)
                J_p = np.vstack((J_lin_p, J_ang_p))
                dJ_dqi = (J_p - J) / delta
                # 累加 dq_i * dJ/dq_i
                J_dot += dJ_dqi * dq_list[i]

            return J_dot

        def compute_torque(q, q_dot, q_ddot, p_ddot_des, current_pos, current_quat, xd,
                           p_dot, p_dot_d, J, J_prev, M, C, g, K_d, D_d):
            # 计算所需变量
            J_inv = np.linalg.pinv(J)
            J_dot = compute_jacobian_dot_analytic(self.robot_id, 7, q_list, dq_list)

            # 位置与姿态误差
            pos_error, ori_error = self.computePoseError(q, xd)
            e = -np.hstack((pos_error, ori_error)).reshape(6, 1)
            e_dot = p_dot - p_dot_d

            # 任务空间控制力矩项 (b中第一行)
            tau_task = M @ J_inv @ (p_ddot_des - J_dot @ J_inv @ p_dot) + C + g - J.T @ (K_d @ e + D_d @ e_dot)
            Fx = p.readUserDebugParameter(ext_force_x_slider)
            Fy = p.readUserDebugParameter(ext_force_y_slider)
            Fz = p.readUserDebugParameter(ext_force_z_slider)
            # 力矩约束示例 (你应根据实际情况调整)
            F_ext = np.vstack((Fx, Fy, Fz, 0, 0, 0)).reshape(6, 1)
            tau_ext = J.T @ F_ext  # shape=(n_joints,1)
            # H矩阵，这里我们假设为质量矩阵M

            n_joints = len(q)
            H = np.eye(n_joints)
            # 定义优化变量
            u = cp.Variable((n_joints, 1))
            tau_2 = cp.Variable((n_joints, 1))

            # 约束定义
            # A矩阵构建 (严格与图片中定义一致)
            A_upper = np.hstack([np.eye(n_joints), -np.eye(n_joints)])
            J_T_inv=np.linalg.pinv(J.T)
            A_lower = np.hstack([np.zeros((J.shape[0], n_joints)), np.linalg.pinv(J.T)])
            A = np.vstack([A_upper, A_lower])

            # b向量定义 (严格与图片中定义一致)
            b_upper = tau_task
            b_lower = np.zeros((J.shape[0], 1))
            b = np.vstack([b_upper, b_lower])


            # —— 3) 加速度上下界 ddq_min, ddq_max ——
            T = dt  # 你的控制周期
            q_min = np.array([-2/3 * np.pi, -2/3 * np.pi, -2/3 * np.pi, -2/3 * np.pi, -2/3 * np.pi, -2/3 * np.pi, -2/3 * np.pi])
            q_max = np.array([2/3 * np.pi, 2/3 * np.pi, 2/3 * np.pi, 2/3 * np.pi, 2/3 * np.pi, 2/3 * np.pi, 2/3 * np.pi])
            #V_max = np.array(self.joint_velocity_limits)
            #A_max = np.array(self.joint_acceleration_limits)

            term1 = 2 * (q_min - q - q_dot * T) / T ** 2
            term2 = 2 * (q_max - q - q_dot * T) / T ** 2
            #term3 = (-V_max - q_dot) / T
            #term4 = (V_max - q_dot) / T

            #ddq_min = np.maximum.reduce([term1, term3, -A_max])
            #ddq_max = np.minimum.reduce([term2, term4, A_max])
            ddq_min = term1
            ddq_max = term2
            # —— 4) 映射到扭矩上下界 u_lower,u_upper ——
            # u = M·ddq + C + g - τ_ext
            other_torque = (C + g - tau_ext).flatten()

            # 枚举 2^n 个顶点
            vertices_ddq = []
            for bits in itertools.product([0, 1], repeat=n_joints):
                v = np.array([ddq_min[i] if bit == 0 else ddq_max[i]
                              for i, bit in enumerate(bits)])
                vertices_ddq.append(v)

            # 对应的 u 顶点
            vertices_u = [M.dot(v) + other_torque for v in vertices_ddq]
            u_stack = np.stack(vertices_u, axis=1)  # shape (n, 2^n)

            # 分量最小/最大
            u_lower = u_stack.min(axis=1)
            u_upper = u_stack.max(axis=1)

            # —— 5) 构造 CVXPY 的 constant bound ——
            d_lower = cp.Constant(u_lower.reshape(-1, 1))
            d_upper = cp.Constant(u_upper.reshape(-1, 1))

            # Cu约束矩阵定义（这里示例为单位矩阵）
            C_mat = np.eye(n_joints)
            d_lower = -np.ones(n_joints) * 100000
            d_upper = np.ones(n_joints) * 100000

            d_lower = cp.Constant(d_lower.reshape(-1, 1))
            d_upper = cp.Constant(d_upper.reshape(-1, 1))
            # 构建QP问题
            constraints = [
                A @ cp.vstack([u, tau_2]) == b,
                d_lower <= C_mat @ u,
                C_mat @ u <= d_upper
            ]

            # 定义QP目标函数
            objective = cp.Minimize(0.5 * cp.quad_form(u, H))

            # 求解优化问题
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP)
            if u.value is None or tau_2.value is None:
                raise ValueError("QP Optimization failed!")
            print('tau_2',tau_2.value)

            null=J @ tau_2.value
            null1 = np.linalg.pinv(J.T) @ tau_2.value
            tau_optimal = u.value
            norm_optimal=tau_optimal.T@tau_optimal
            norm=tau_task.T@tau_task
            return tau_optimal

        def get_joint_states_and_acceleration(robot_id, controlled_joints, time_step):
            """
            获取关节位置、速度以及利用有限差分计算的加速度，并返回列表与数组形式。
            """
            joint_states = [p.getJointState(robot_id, j) for j in controlled_joints]
            q_list = [s[0] for s in joint_states]
            dq_list = [s[1] for s in joint_states]
            q = np.array(q_list)
            q_dot = np.array(dq_list)
            if not hasattr(get_joint_states_and_acceleration, "prev_q_dot"):
                get_joint_states_and_acceleration.prev_q_dot = q_dot.copy()
            q_ddot = (q_dot - get_joint_states_and_acceleration.prev_q_dot) / time_step
            get_joint_states_and_acceleration.prev_q_dot = q_dot.copy()
            return q_list, dq_list, q, q_dot, q_ddot

        while True:
            # 获取当前关节状态
            xd = self.readGUIparams(xdGUIids)
            q_list, dq_list, q, q_dot, q_ddot = get_joint_states_and_acceleration(self.robot_id,
                                                                                  self.controllable_joints, dt)
            M = np.array(p.calculateMassMatrix(self.robot_id, q_list))

            g = np.array(p.calculateInverseDynamics(self.robot_id, q_list, [0] * 7, [0] * 7)).reshape(7, 1)

            Cg = np.array(p.calculateInverseDynamics(self.robot_id, q_list, dq_list, [0] * 7)).reshape(7, 1)
            C = Cg - g
            # 获取末端状态（使用末端 link 索引为 total_joints_num-1）
            ee_state = p.getLinkState(self.robot_id, total_joints_num - 1, computeLinkVelocity=1,
                                      computeForwardKinematics=1)
            pos = np.array(ee_state[0]).reshape(3, 1)
            quat = ee_state[1]
            # 计算任务空间雅可比（6×n）
            J_lin, J_ang = p.calculateJacobian(self.robot_id, total_joints_num - 1, [0, 0, 0],
                                               q_list, dq_list, [0] * len(q_list))
            J_lin = np.array(J_lin)
            J_ang = np.array(J_ang)
            J = np.vstack([J_lin, J_ang])
            p_dot = np.array(J @ q_dot).reshape(6, 1)

            # 按您原来方式计算 tau（含位置与姿态误差）

            tau = compute_torque(q, q_dot, q_ddot, p_ddot_des, pos, quat, np.array(xd), p_dot, p_dot_d,
                                 J, J_prev, M, C, g, K_d, D_d)
            # 计算 null space 补偿

            print("q:", q)
            print("tau:", tau)
            print("End-effector pose:", pos.flatten(), quat)
            ext_force_x = p.readUserDebugParameter(ext_force_x_slider)
            ext_force_y = p.readUserDebugParameter(ext_force_y_slider)
            ext_force_z = p.readUserDebugParameter(ext_force_z_slider)
            ext_force = [ext_force_x, ext_force_y, ext_force_z]
            # 将外力施加在末端执行器所在的连杆上，这里使用 p.LINK_FRAME 施加在连杆原点
            p.applyExternalForce(objectUniqueId=self.robot_id,
                                 linkIndex=total_joints_num - 1,
                                 forceObj=ext_force,
                                 posObj=pos,
                                 flags=p.WORLD_FRAME)
            p.setJointMotorControlArray(self.robot_id, joint_indices,
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=tau)
            p.stepSimulation()
            time.sleep(dt)
        p.disconnect()