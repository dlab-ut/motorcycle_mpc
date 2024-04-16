"""
Copyright (c) 2024 Michikuni Eguchi
Released under the MIT license
"""

import pybullet as p
import pybullet_data

import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from path_generate import make_track, make_side_lane, make_spline_course
from plotter.plot_utils import plot_robot, plot_arrow


# pybullet parameter
# joint id
REARWHEEL_ID = 0
FRONTWHEEL_ID = 2
STEER_ID = 1
# number of visualize path
est_paths = 3
# position offset
X_OFFSET = -0.53
Z_OFFSET = 0.37

# initial velocity
INITIAL_VEL = 5

PATH_INTERVAL = 10 # for debug

class PathTrackEnv:
    def __init__(self, step_size, center_path, lane_width, GUI):

        self.GUI = GUI
        self.step_size = step_size
        self.center_path = center_path
        self.lane_width = lane_width

        # make side lane
        right_lane, left_lane = make_side_lane(center_path, lane_width=lane_width)

        self.right_lane = right_lane
        self.left_lane = left_lane

        # pybullet setup
        if GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # load model
        ini_pos = [center_path[0][0], center_path[0][1], 0.003]
        ini_orn = p.getQuaternionFromEuler([0, 0, -math.pi/2+center_path[0][2]])
        self.bicycle = p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../../model/bicycle.urdf", ini_pos, ini_orn)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(False)
        p.setTimeStep(step_size)
        # initial velocity
        p.resetBaseVelocity(self.bicycle, [INITIAL_VEL*math.cos(center_path[0][2]), INITIAL_VEL*math.sin(center_path[0][2]), 0])

        # visualize lane
        if GUI:
            for i in range(0, len(center_path)-PATH_INTERVAL, PATH_INTERVAL):
                p.addUserDebugLine([center_path[i][0], center_path[i][1], 0.05], [center_path[i+PATH_INTERVAL][0], center_path[i+PATH_INTERVAL][1], 0.05], [0, 0, 1], 3, 0)
                p.addUserDebugLine([right_lane[i][0], right_lane[i][1], 0.05], [right_lane[i+PATH_INTERVAL][0], right_lane[i+PATH_INTERVAL][1], 0.05], [0, 1, 0], 3, 0)
                p.addUserDebugLine([left_lane[i][0], left_lane[i][1], 0.05], [left_lane[i+PATH_INTERVAL][0], left_lane[i+PATH_INTERVAL][1], 0.05], [0, 1, 0], 3, 0)

        # debug bar
        self.cameraYawId = p.addUserDebugParameter("cameraYaw", -180, 180, -90)
        self.cameraPitchId = p.addUserDebugParameter("cameraPitch", -90, 0, -25)
        self.cameraDistanceId = p.addUserDebugParameter("cameraDistance", 2.5, 10, 3.5)

        # plt init
        plt.figure(figsize=[12,10])
        plt.rcParams["font.size"] = 18

        # variable initialize
        self.turns = 0
        self.pre_yaw = 0
        self.pre_roll = 0
        self.current_index = 0
        self.t = 0
        self.state = []
        self.action = []
        self.states = []

    def get_state(self):
        # get state
        pos, orn = p.getBasePositionAndOrientation(self.bicycle)
        velocity_vector = p.getBaseVelocity(self.bicycle)
        angle = p.getEulerFromQuaternion(orn)
        pitch = angle[0]
        roll = angle[1]
        yaw = angle[2] + math.pi/2 + 2*math.pi*self.turns
        pos = [pos[0]+X_OFFSET*math.cos(yaw)-Z_OFFSET*math.sin(roll)*math.sin(yaw), 
               pos[1]+X_OFFSET*math.sin(yaw)+Z_OFFSET*math.sin(roll)*math.cos(yaw), 
               0.05]

        # yaw fix
        if (yaw-self.pre_yaw)>math.pi:
            self.turns -= 1
            yaw = angle[2] + math.pi/2 + 2*math.pi*self.turns
        elif(yaw-self.pre_yaw)<-math.pi:
            self.turns += 1
            yaw = angle[2] + math.pi/2 + 2*math.pi*self.turns

        droll = (roll - self.pre_roll)/self.step_size
        dyaw = velocity_vector[1][2]
        self.pre_roll = roll
        self.pre_yaw = yaw

        steer_info = p.getJointState(self.bicycle, STEER_ID)
        steer_angle = steer_info[0]

        v_r = velocity_vector[0][0]*math.cos(yaw)+velocity_vector[0][1]*math.sin(yaw)
        v_l = -velocity_vector[0][0]*math.sin(yaw)+velocity_vector[0][1]*math.cos(yaw)
        velocity = math.sqrt(velocity_vector[0][0]**2 + velocity_vector[0][1]**2)

        state = np.array([pos[0], pos[1], v_r, yaw, dyaw, roll, droll, steer_angle]) # [x, y, v, yaw, dyaw, roll, droll, steer]

        return state
    
    def reset(self):
        # variable initialize
        self.turns = 0
        self.pre_yaw = 0
        self.pre_roll = 0
        self.t = 0

        self.state = self.get_state()

        return self.state
    
    def step(self, action):

        # camera update
        if self.GUI:
            cameraYaw = p.readUserDebugParameter(self.cameraYawId)
            cameraPitch = p.readUserDebugParameter(self.cameraPitchId)
            cameraDistance = p.readUserDebugParameter(self.cameraDistanceId)

        # input
        # rear wheel torque
        p.setJointMotorControl2(bodyUniqueId=self.bicycle,
                                jointIndex=REARWHEEL_ID,
                                controlMode=p.TORQUE_CONTROL,
                                force=action[1])

        # front wheel(free)
        p.setJointMotorControl2(bodyUniqueId=self.bicycle,
                                jointIndex=FRONTWHEEL_ID,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=0,
                                force=0)
        # steer velocity
        p.setJointMotorControl2(bodyUniqueId=self.bicycle,
                                jointIndex=STEER_ID,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=action[0],
                                force=999999)


        # camera move
        if self.GUI:
            pos = [self.state[0], self.state[1], 0.05]
            p.resetDebugVisualizerCamera(cameraDistance=cameraDistance, cameraYaw=cameraYaw+self.state[3]*180/math.pi, cameraPitch=cameraPitch, cameraTargetPosition=pos)

        # simulation update
        p.stepSimulation()

        # state update
        self.state = self.get_state()
        self.action = action

        self.t += self.step_size
        self.states.append(self.state)

        return self.state

    def render(self, estimated_trajectory, reference_path):

        est_traj_length = len(estimated_trajectory[0])
        for i in range(est_paths):
            p.addUserDebugLine([estimated_trajectory[0][int(est_traj_length*i/est_paths)], estimated_trajectory[1][int(est_traj_length*i/est_paths)], 0.05], 
                               [estimated_trajectory[0][int(est_traj_length*(i+1)/est_paths-1)], estimated_trajectory[1][int(est_traj_length*(i+1)/est_paths-1)], 0.05], 
                               [1, 0, 0], 3, 0.2)

        traj_x = [state[0] for state in self.states]
        traj_y = [state[1] for state in self.states]

        # visualize track and runned trajectory
        plt.cla()
        # track
        plt.plot(self.center_path[:, 0], self.center_path[:, 1], linestyle="dashed")
        plt.plot(self.right_lane[:, 0], self.right_lane[:, 1], color="green")
        plt.plot(self.left_lane[:, 0], self.left_lane[:, 1], color="green")
        # robot
        plot_robot(self.state[0], self.state[1], self.state[3], 0.6)
        plot_arrow(plt, self.state[0], self.state[1], self.state[3])
        # reference_path
        plt.scatter(reference_path[:, 0], reference_path[:, 1], color="b")
        ref_right_bound, ref_left_bound = make_side_lane(reference_path, lane_width=self.lane_width)
        plt.scatter(ref_right_bound[:, 0], ref_right_bound[:, 1], color="b")
        plt.scatter(ref_left_bound[:, 0], ref_left_bound[:, 1], color="b")
        # estimated path
        plt.plot(estimated_trajectory[0], estimated_trajectory[1], color="r", lw=4)
        # runned trajectory
        plt.plot(traj_x, traj_y, color='purple', lw=4)
        plt.title(f"motorcycle path track\n v: {self.state[2]:.2f} , w: {self.state[4]:.2f} , roll: {self.state[5]:.2f} , steerW: {self.action[0]:.2f} , Torque: {self.action[1]:.2f} , t: {self.t:.2f}")
        plt.axis("equal")
        plt.pause(0.0001)