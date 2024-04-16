"""
Copyright (c) 2024 Michikuni Eguchi
Released under the MIT license
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import time
import argparse

import config.config as config
from MPC import MPC
from Bicycle import Bicycle
from reference_trajectory import *
from path_generate import make_track, make_spline_course
from env.path_track_env import PathTrackEnv


def main():
    # parser arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('course', help='course type', choices=['course1', 'course2'], default='course1')
    parser.add_argument('-g', '--gui', action='store_true', help='show gui')
    args = parser.parse_args()

    # make course
    if args.course == 'course1':
        track_path = make_track(circle_radius=config.TRACK_RADIUS, linelength=config.TRACK_LENGTH, dl = config.DL)
    elif args.course == 'course2':
        track_path = make_spline_course(config.ax1, config.ay1, dl = config.DL)
    elif args.course == 'course3':
        track_path = make_spline_course(config.ax2, config.ay2, dl = config.DL)

    # set environment
    env = PathTrackEnv(config.sampling_time, track_path, config.LANE_WIDTH, args.gui)

    # set mpc
    #mpc config
    mpc_config = {
              'MAX_TORQUE':config.MAX_TORQUE,
              'MAX_STEER_DELTA':config.MAX_DSTEER,
              'MAX_V':config.MAX_V,
              'MAX_STEER':config.MAX_STEER,
              'MAX_ROLL':config.MAX_ROLL,
              'LANE_WIDTH':config.LANE_WIDTH,
              'FRICTION_COEF':config.FRICTION_COEF,
              'DT':config.sampling_time,
              'HORIZON':config.HORIZON,
              'R0':config.R[0],
              'R1':config.R[1],
              'Q0':config.Q[0],
              'Q1':config.Q[1],
              'Q2':config.Q[2],
              'Q3':config.Q[3],
              'Q4':config.Q[4],
              'Qf0':config.Qf[0],
              'Qf1':config.Qf[1],
              'Qf2':config.Qf[2],
              'Qf3':config.Qf[3],
              'Qf4':config.Qf[4],
              'Kec':config.Kec,
                }
    mpc = MPC(mpc_config)

    # variables initialized
    current_index = 0
    t = 0
    mpc_warm = True

    state = env.reset() # [x, y, v, yaw, dyaw, roll, droll, steer]

    while True:

        # control reference
        state_2d = np.array([state[0], state[1], state[3], state[2]])
        reference_path, current_index = calc_ref_trajectory(state_2d, track_path, current_index, config.sampling_time, config.HORIZON, config.DL, reference_rate=config.REFERENCE_RATE)
        
        # mpc warm
        if mpc_warm:
            mpc_warm = False
            # dispose control value
            for i in range(config.WARM_ITER_NUM):
                mpc.solve(state, reference_path)

        # solve mpc
        current_time = time.time()
        u = mpc.solve(state, reference_path)
        # estimated trajectory
        estimated_trajectory = mpc.get_trajectory()
        
        print("solve time:" + str(round((time.time() - current_time)*1000, 4)) + "ms")

        # state update
        state = env.step(u)

        # render
        env.render(estimated_trajectory, reference_path)
        

        # fail(roll)
        if abs(state[5])>math.pi/2 - 0.6:
            print("fall down!!\n")
            break

        # goal
        if current_index >= len(track_path)-1:
            print("finish!! time "+str(t)+"[s]\n\n")
            break

        print("")


if __name__ == '__main__':
    main()