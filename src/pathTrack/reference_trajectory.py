"""
Copyright (c) 2024 Michikuni Eguchi
Released under the MIT license
"""
import numpy as np
import config.config as config


def calc_nearest_index(state, cx, cy, cind, N_IND_SEARCH = 10):

    dx = [state[0] - icx for icx in cx[cind:(cind + N_IND_SEARCH)]]
    dy = [state[1] - icy for icy in cy[cind:(cind + N_IND_SEARCH)]]

    distanceList = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    min_distance = min(distanceList)

    nearest_index = distanceList.index(min_distance) + cind

    return nearest_index

def calc_ref_trajectory(state, path, cind, sampling_time, T, DL, reference_rate = 0.3):
    ncourse = len(path)
    npath_state = len(path[0])
    xref = np.zeros((T+1, npath_state))
    

    ind = calc_nearest_index(state, path[:, 0], path[:, 1], cind)

    if cind >= ind:
        ind = cind

    for i in range(npath_state):
        xref[0, i] = path[ind, i]


    travel = abs(state[3]) * reference_rate
    end_travel = 0

    dl = abs(config.MAX_V) * sampling_time
    for i in range(T+1):
        travel += dl
        dind = int(round(travel / DL))

        if (ind + dind) < ncourse:
            for j in range(npath_state):
                xref[i, j] = path[ind + dind, j]
        else:
            end_travel += dl
            end_p = path[ncourse - 1, :]
            xref[i, 0] = end_travel * np.cos(end_p[2]) + end_p[0]
            xref[i, 1] = end_travel * np.sin(end_p[2]) + end_p[1]
            xref[i, 2] = path[ncourse - 1, 2]
            xref[i, 3] = 0

    return xref, ind
