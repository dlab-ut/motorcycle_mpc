"""
Copyright (c) 2024 Michikuni Eguchi
Released under the MIT license
"""

from casadi import *
import math
import numpy as np

from Bicycle import Bicycle

class MPC:
    def __init__(self, config):
        self.T = config['DT']*config['HORIZON'] # horizon length
        self.N = config['HORIZON'] # discreate grid number
        self.dt = config['DT'] # minute time
        self.nx = 8 # state variable number [x, y, v, yaw, dyaw, roll, droll, steer]
        self.nu = 2 # input variable number
        self.npath = 4 # path variable number
        self.nvar = self.nx + self.nu
        self.Model = Bicycle(self.dt)

        #  weights[ec,  el,  v, roll, roll rate]
        self.Q  = [config['Q0'], config['Q1'], config['Q2'], config['Q3'], config['Q4']]
        self.Qf  = [config['Qf0'], config['Qf1'], config['Qf2'], config['Qf3'], config['Qf4']]
        self.R  = [config['R0'], config['R1']]       # input weights[delta steer, rear wheel torque]
        # offset of contour error gain
        Kec = config['Kec']

        max_torque_input = config['MAX_TORQUE']
        max_delta_input = config['MAX_STEER_DELTA']
        self.max_v = config['MAX_V']
        max_roll = config['MAX_ROLL']
        max_steer = config['MAX_STEER']
        lane_width = config['LANE_WIDTH']
        friction_coef = config['FRICTION_COEF']
        max_friction_force = self.Model.MASS/2 * self.Model.GRAVITY * friction_coef

        w = [] # contain optimal variable
        w0 = [] # contain initial optimal variable
        lbw = [] # lower bound optimal variable
        ubw = [] # upper bound optimal variable
        J = 0 # cost function
        g  = [] # constrain
        lbg = [] # lower bound constrain
        ubg = [] # upper bound constrain
        lam_x0 = [] # Lagrangian multiplier
        lam_g0 = [] # Lagrangian multiplier

        Xk = MX.sym('X0', self.nx) # initial time state vector x0
        Pathref = MX.sym('path_ref', self.N+1, self.npath) # path reference

        w += [Xk]
        # equality constraint
        lbw += [0] * self.nx    # constraints are set by setting lower-bound and upper-bound to the same value
        ubw += [0] * self.nx    # constraints are set by setting lower-bound and upper-bound to the same value
        w0 +=  [0] * self.nx    # x0 initial estimate
        lam_x0 += [0] * self.nx # Lagrangian multiplier initial estimate

        ############## stage ##################
        for k in range(self.N):
            Xref = Pathref[k,:]
            Uk = MX.sym('U_' + str(k), self.nu)
            w += [Uk]
            lbw += [-max_delta_input, -max_torque_input]
            ubw += [max_delta_input, max_torque_input]
            w0 += [0] * self.nu
            lam_x0 += [0] * self.nu

            # contouring error of path
            Ec = sin(Xref[2])*(Xk[0] - Xref[0]) - cos(Xref[2])*(Xk[1] - Xref[1])
            #lag error of path
            El = -cos(Xref[2])*(Xk[0] - Xref[0]) - sin(Xref[2])*(Xk[1] - Xref[1])

            # offset contouring error of path
            Ec_ref = -((lane_width/2*0.9)) * tanh(Kec * Xref[3])

            #stage cost
            J += self.stage_cost(Xk, Uk, Ec, El, Ec_ref)

            # Discretized equation of state by forward Euler
            Xk_ = self.Model.dynamics(Xk, Uk)
            Xk_next = vertcat(Xk_[0],
                              Xk_[1],
                              Xk_[2],
                              Xk_[3],
                              Xk_[4],
                              Xk_[5],
                              Xk_[6],
                              Xk_[7])

            Xk1 = MX.sym('X_' + str(k+1), self.nx)
            w   += [Xk1]
            # [x, y, v, yaw, dyaw, roll, droll, steer]
            lbw += [-inf, -inf, 0, -inf, -inf, -max_roll, -inf, -max_steer]
            ubw += [ inf,  inf,  inf,  inf,  inf,  max_roll,  inf,  max_steer]
            w0 += [0.0] * self.nx
            lam_x0 += [0.0] * self.nx

            # (xk+1=xk+fk*dt) is introduced as an equality constraint
            g   += [Xk_next-Xk1]
            lbg += [0] * self.nx     # Equality constraints are set by setting lower-bound and upper-bound to the same value
            ubg += [0] * self.nx     # Equality constraints are set by setting lower-bound and upper-bound to the same value
            lam_g0 += [0] * self.nx
            Xk = Xk1

            # contouring error constraint
            Xref_next = Pathref[k+1,:]
            Ec_next = sin(Xref_next[2])*(Xk[0] - Xref_next[0]) - cos(Xref_next[2])*(Xk[1] - Xref_next[1])
            g   += [Ec_next]
            lbg += [-lane_width/2*1.2]
            ubg += [lane_width/2*1.2]
            lam_g0 += [0]

            # lateral force of rear wheel constraint
            vel = Xk[2]
            steer = Xk[7]
            beta = atan(tan(steer)*sin(self.Model.N)/cos(Xk[5]))
            curvature = tan(beta)/self.Model.L
            centForce = self.Model.MASS * curvature * vel*vel

            g   += [centForce]
            lbg += [-max_friction_force]
            ubg += [max_friction_force]
            lam_g0 += [0]
        #######################################

        ################ final ################
        Xref = Pathref[self.N,:]

        # contouring error of path
        Ec = sin(Xref[2])*(Xk[0] - Xref[0]) - cos(Xref[2])*(Xk[1] - Xref[1])
        #lag error of path
        El = -cos(Xref[2])*(Xk[0] - Xref[0]) - sin(Xref[2])*(Xk[1] - Xref[1])

        # offset contouring error of path
        Ec_ref = -((lane_width/2*0.9)) * tanh(Kec * Xref[3])

        #stage cost
        J += self.final_cost(Xk, Ec, El, Ec_ref)

        ########################################


        self.J = J
        self.w = vertcat(*w)
        self.g = vertcat(*g)
        self.x = w0
        self.lam_x = lam_x0
        self.lam_g = lam_g0
        self.lbx = lbw
        self.ubx = ubw
        self.lbg = lbg
        self.ubg = ubg

        # NonLinearProblem
        self.nlp = {'f': self.J, 'x': self.w, 'p': Pathref, 'g': self.g}
        # Ipopt solver，min bariar parameter0.1，maximum iteration time 5, warm start ON
        self.solver = nlpsol('solver', 'ipopt', self.nlp, {'calc_lam_p':True, 'calc_lam_x':True, 'print_time':False, 'ipopt':{'max_iter':4, 'hessian_approximation':'exact', 'mu_target':0.5, 'mu_min':0.001, 'warm_start_init_point':'yes', 'print_level':0, 'print_timing_statistics':'no'}})

    def stage_cost(self, x, u, ec, el, ec_ref):
        cost = 0

        # path contour error
        cost += self.Q[0] * (ec-ec_ref)**2
        # path lag error
        cost += self.Q[1] * el**2
        # maximum velocity
        cost -= self.Q[2] * x[2]
        # roll angle error
        cost += self.Q[3] * x[5]**2
        #minimum roll angular velocity
        cost += self.Q[4] * x[6]**2
        # input cost
        for i in range(self.nu):
            cost += self.R[i] * u[i]**2

        return cost

    def final_cost(self, x, ec, el, ec_ref):
        cost = 0

        # path contour error
        cost += self.Qf[0] * (ec-ec_ref)**2
        # path lag error
        cost += self.Qf[1] * el**2
        # maximum velocity
        cost -= self.Qf[2] * x[2]
        # roll angle error
        cost += self.Qf[3] * x[5]**2
        #minimum roll angular velocity
        cost += self.Qf[4] * x[6]**2

        return cost

    def init(self, x0=None, path_ref=None):
        if x0 is not None:
            # set constraint for initial state
            self.lbx[0:self.nx] = x0
            self.ubx[0:self.nx] = x0
        if path_ref is None:
            path_ref = np.zeros((self.N, self.npath))
        # primal variables (x) dual variables solve（warm start）
        sol = self.solver(x0=self.x, p=path_ref, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # save solution for next warm start
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()

    """
    x0 = np.array([x_current, y_current, v_current, yaw_current, dyaw_current, roll_current, droll_current, steer_current])
    path_ref = np.array([x_ref(0), y_ref(0), v_ref(0)]
                                 ...
                        [x_ref(N), y_ref(N), v_ref(N)]])
    """
    def solve(self, x0, path_ref):
        # set constraint for initial state
        self.lbx[0:self.nx] = x0
        self.ubx[0:self.nx] = x0
        # primal variables (x) dual variables solve（warm start）
        sol = self.solver(x0=self.x, p=path_ref, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # save solution for next warm start
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()

        return np.array([self.x[self.nx], self.x[self.nx+1]]) # return control input

    def get_trajectory(self):
        path_x = []
        path_y = []

        for i in range(self.N):
            path_x.append(self.x[(self.nvar)*i])
            path_y.append(self.x[(self.nvar)*i+1])

        return path_x, path_y