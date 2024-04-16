"""
Michikuni Eguchi, 2024.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import cubic_spline_planner
from plotter.plot_utils import plot_arrow, circle



def make_track(circle_radius, linelength, dl):
    """ make track
    Input parameters:
        circle_radius (float): circle radius
        linelength (float): line length

    Returns:
        road (numpy.ndarray): shape(n_point, 3) x, y, angle, curvature
    """
    line_points = round(linelength/dl)

    line = np.linspace(-linelength/2, linelength/2, num=line_points+1, endpoint=False)[1:]
    line_1 = np.stack((line, np.zeros(line_points)), axis=1)
    line_2 = np.stack((line[::-1], np.zeros(line_points)+circle_radius*2.), axis=1)

    # circle
    circle_1_x, circle_1_y = circle(linelength/2., circle_radius,
                                    circle_radius, start=-np.pi/2., end=np.pi/2., dl=dl)
    circle_1 = np.stack((circle_1_x, circle_1_y), axis=1)

    circle_2_x, circle_2_y = circle(-linelength/2., circle_radius,
                                    circle_radius, start=np.pi/2., end=3*np.pi/2., dl=dl)
    circle_2 = np.stack((circle_2_x, circle_2_y), axis=1)

    road_pos = np.concatenate((line_1, circle_1, line_2, circle_2), axis=0)

    # calc road angle
    road_diff = road_pos[1:] - road_pos[:-1]
    road_angle = np.arctan2(road_diff[:, 1], road_diff[:, 0])
    road_angle = np.concatenate((np.zeros(1), road_angle))

    road = np.concatenate((road_pos, road_angle[:, np.newaxis]), axis=1)

    # calc road curvature
    road_curvature = calc_curvature_range_kutta(road[:, 0], road[:, 1])

    road = np.concatenate((road, np.array(road_curvature)[:, np.newaxis]), axis=1)

    # start offset
    road[:, 0] = road[:, 0] + linelength/2

    return road

#def make_course1():


def make_spline_course(ax, ay, dl):
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    road = np.stack((cx, cy, cyaw), axis=1)

    # calc road curvature
    road_curvature = calc_curvature_range_kutta(road[:, 0], road[:, 1])

    road = np.concatenate((road, np.array(road_curvature)[:, np.newaxis]), axis=1)

    return road

def calc_curvature_range_kutta(x, y):
    dists = np.array([np.hypot(dx, dy) for dx, dy in zip(np.diff(x), np.diff(y))])
    curvatures = [0.0, 0.0]
    for i in np.arange(2, len(x)-1):
        dx = (x[i+1] - x[i])/dists[i]
        dy = (y[i+1] - y[i])/dists[i]
        ddx = (x[i-2] - x[i-1] - x[i] + x[i+1])/(2*dists[i]**2)
        ddy = (y[i-2] - y[i-1] - y[i] + y[i+1])/(2*dists[i]**2)
        curvature = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** 1.5)
        curvatures.append(curvature)
    curvatures.append(0.0)
    return curvatures

def make_side_lane(road, lane_width):
    """make_side_lane
    Input parameters:
        road (numpy.ndarray): shape(n_point, 3) x, y, angle, curvature
        lane_width (float): width of the lane
    Output:
        right_lane (numpy.ndarray): shape(n_point, 3) x, y, angle
        left_lane  (numpy.ndarray): shape(n_point, 3) x, y, angle
    """
    right_lane_x = lane_width/2*np.cos(road[:,2]-np.pi/2) +road[:,0]
    right_lane_y = lane_width/2*np.sin(road[:,2]-np.pi/2) +road[:,1]
    right_lane_pos = np.stack((right_lane_x, right_lane_y), axis=1)

    left_lane_x = lane_width/2*np.cos(road[:,2]+np.pi/2) +road[:,0]
    left_lane_y = lane_width/2*np.sin(road[:,2]+np.pi/2) +road[:,1]
    left_lane_pos = np.stack((left_lane_x, left_lane_y), axis=1)

    road_angle = road[:,2]

    right_lane = np.concatenate((right_lane_pos, road_angle[:, np.newaxis]), axis=1)
    left_lane = np.concatenate((left_lane_pos, road_angle[:, np.newaxis]), axis=1)

    return right_lane, left_lane

if __name__ == '__main__':
    # parser arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('course_name', help='course name')
    parser.add_argument('lane_width', help='lane width')
    args = parser.parse_args()

    lane_width = float(args.lane_width)


    # make track
    """
    road = make_track(circle_radius = 10, linelength = 17, dl=0.1)
    right_lane, left_lane = make_side_lane(road, lane_width=lane_width)
    #print(road)
    # track
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(road[:, 0], road[:, 1])
    ax1.plot(right_lane[:, 0], right_lane[:, 1])
    ax1.plot(left_lane[:, 0], left_lane[:, 1])
    ax1.set_title("track")
    # arrow plot
    for i in range(int(road[:,0].size/10)):
        plot_arrow(ax1, road[i*10, 0], road[i*10, 1], road[i*10, 2], 1, 0.5)

    # track curvature
    l = np.linspace(0, road[:,0].size, num=road[:,0].size)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.plot(l, road[:, 3])
    ax2.set_title('track curvature')
    """

    # make spline course
    zx = [0,5,30,30,30,-20,-20, -10, -4, 5, 24, 24,  20,10, -20, -20,   -5,0]
    zy = [-30,-30,-30,0,20,20,10, 5,  4, 15, 15, 0, -20,-20,   -5, -30, -30,-30]
    road = make_spline_course(zx, zy, dl=0.1)
    right_lane, left_lane = make_side_lane(road, lane_width=lane_width)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(road[:, 0], road[:, 1])
    ax1.plot(right_lane[:, 0], right_lane[:, 1])
    ax1.plot(left_lane[:, 0], left_lane[:, 1])
    for i in range(int(road[:,0].size/30)):
        plot_arrow(ax1, road[i*30, 0], road[i*30, 1], road[i*30, 2], 2, 1)
    ax1.set_title('spline course')

    # track curvature
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    l = np.linspace(0, road[:,0].size, num=road[:,0].size)
    ax2.plot(l, road[:, 3])
    ax2.set_title('spline course curvature')
    

    plt.show()

    # csv
    df = pd.DataFrame(columns=['x_m',
                               'y_m',
                               'w_tr_right_m',
                               'w_tr_left_m'])
    df2 = pd.DataFrame(columns=['x',
                               'y',
                               'angle',
                               'w_tr_right_m',
                               'w_tr_left_m'])
    for i in range(len(road[:, 0])):
        df.loc[i] = [round(road[i][0], 3), round(road[i][1], 3), lane_width/2, lane_width/2]
        df2.loc[i] = [round(road[i][0], 3), round(road[i][1], 3), round(road[i][2], 3), lane_width/2, lane_width/2]
        df.to_csv('course_data/'+args.course_name+'.csv', header=True, index=False)
        df2.to_csv('course_data/'+args.course_name+'_all.csv', header=True, index=False)
        if i%100 == 0:
            print('...')

    print('done')
