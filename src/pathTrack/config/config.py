"""
Copyright (c) 2024 Michikuni Eguchi
Released under the MIT license
"""

import numpy as np
import math

############## mpc parameter ##################

WARM_ITER_NUM = 30 # number of iterations for warm start
sampling_time = 0.04 # [s]
HORIZON = 50  # horizon length
MAX_TORQUE = 100 # [N.m]
MAX_DSTEER = np.deg2rad(90.0) # [rad]
MAX_V = 8 # [m/s]
MAX_STEER = np.deg2rad(50.0) # [rad]
MAX_ROLL = math.pi/5 # [rad]
FRICTION_COEF = 0.6 # friction coefficient

# input cost weights (steering angular velocity, rear wheel torque)
R = [0.01, 0.001]
# state cost weights (contour, lag, v, roll, roll_rate)
Q = [2, 0.2, 2.0, 20.0, 25.0]
# final state cost weights
Qf = [Q[0], Q[1], Q[2], Q[3], Q[4]]
# offset of contour error gain
Kec = 3
###############################################


############## simulation parameter ##################

# initial velocity
INITIAL_VEL = MAX_V
# lane width
LANE_WIDTH = 5
# diff longtitudinal distance
DL = MAX_V * sampling_time
REFERENCE_RATE = 0.1

# path parameter
TRACK_LENGTH = 20
TRACK_RADIUS = 10

# course2 point
ax1 = [0.0,5.0, 25.0, 32.0, 35.0, 50.0, 58.0, 55.0, 40.0, 30.0, -5.0, -5.0,0.0]
ay1 = [0.0,0.0, 0.0, 5.0, 30.0,  50.0, 45.0, 30.0, 30.0, 50.0, 40.0, 0.0,0.0]
###############################################