"""
Copyright (c) 2024 Michikuni Eguchi
Released under the MIT license
"""

import math
import numpy as np
from casadi import *

class Bicycle:
    def __init__(self, DT):
        self.MASS = 29 # [kg]
        self.L = 1.21 # wheelbase[m]
        self.H = 0.37 # height of the motorcycle center of mass[m]
        self.B = 0.52 # distance of rear wheel point to center of mass[m]
        self.N = 1.37 # caster angle[rad]
        self.R = 0.37 # wheel radius[m]
        self.TRAIL = self.R/math.tan(self.N) # motorcycle trail[m]
        self.DT = DT #[s]
        self.GRAVITY = 9.81 #gravity constant

    def dynamics(self, x, u):
        MASS = self.MASS
        L = self.L
        H = self.H
        B = self.B
        N = self.N
        R = self.R
        TRAIL = self.TRAIL
        DT = self.DT
        GRAVITY = self.GRAVITY

        x_pos = x[0] # x position[m]
        y_pos = x[1] # y position[m]
        v = x[2] # cart velocity[m/s]
        yaw = x[3] # yaw angle[rad]
        dyaw = x[4] # yaw angle velocity[rad/s]
        roll = x[5] # roll angle[rad]
        droll = x[6] # roll angle velocity[rad/s]
        steer = x[7] # steer angle[rad]

        w = u[0] # angle velocity[rad/s]
        torque = u[1] # [N.m]
        F = torque/R# [N]

        sigma = tan(steer)/L
        beta = atan(tan(steer)*sin(N)/cos(roll))

        M00 = H*H
        M01 = -B*H*sigma*cos(roll)
        M10 = -B*H*sigma*cos(roll)
        M11 = B*B*sigma*sigma + pow(1+H*sigma*sin(roll), 2)

        K0 = GRAVITY*(H*sin(roll)+B*TRAIL*sigma*sin(N)*cos(roll)) + (1+H*sigma*sin(roll))*H*sigma*v*v*cos(roll)
        K1 = -2*H*sigma*v*droll*(1+H*sigma*sin(roll))*cos(roll) - B*H*sigma*droll*droll*sin(roll)

        A00 = B*H*v*cos(roll)
        A01 = 0
        A10 = -(B*B*sigma + H*sin(roll)*(1+H*sigma*sin(roll)))*v
        A11 = 1/MASS

        dx = v*cos(yaw)
        dy = v*sin(yaw)
        dv = (-M10/M00*K0+K1+(-M10/M00*A00 + A10)*w+A11*F)/(M11 - M10/M00*M01)
        ddroll = (K0+A00*w)/M00 - ((M01/M00)*(-M10/M00*K0+K1+(-M10/M00*A00+A10)*w+A11*F)/(M11 - M10/M00*M01))

        x_pos = x_pos + dx*DT
        y_pos = y_pos + dy*DT
        v = v + dv*DT
        yaw = yaw + dyaw*DT
        dyaw = v*tan(beta)/L
        roll = roll + droll*DT
        droll = droll + ddroll*DT
        steer = steer + w*DT

        return x_pos, y_pos, v, yaw, dyaw, roll, droll, steer