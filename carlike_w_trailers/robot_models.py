# Author: Mohamed Naveed Gul Mohamed
# email:mohdnaveed96@gmail.com
# Date: Oct 11th 2019
#
# Dynamic models.

import casadi as c
import math as m
import numpy as np


class youBot_model(object):

    """
    model specification and dynamics
    """

    def __init__(self, length=0.58, breadth = 0.38, ang_vel_max=m.pi/6, vel_max=.8, dt=0.1, nx=3, nu=3):

        self.length = length
        self.breadth = breadth
        self.ang_vel_max = ang_vel_max
        #self.ang_accel_max = ang_accel_max
        self.vel_max = vel_max
        #self.accel_max = accel_max
        self.dt = dt
        self.nx = nx
        self.nu = nu
        self.A = c.DM.eye(self.nx) #youBot has a linear model
        self.B = c.DM.eye(self.nu)*self.dt
        self.G = c.DM.eye(self.nu)*self.dt
        self.x = c.MX.sym('x', self.nx,1)
        self.u = c.MX.sym('u', self.nu,1)
        self.Sigma_w = c.DM([[self.vel_max**2,0,0],[0,self.vel_max**2,0],[0,0,self.ang_vel_max**2]])

    def proc_model(self):

        f = c.Function('f',[self.x,self.u],[self.x + self.u*self.dt])
        A = c.Function('A',[self.x,self.u],[c.jacobian(f(self.x,self.u),self.x)]) #linearization
        B = c.Function('B',[self.x,self.u],[c.jacobian(f(self.x,self.u),self.u)])

        return f,A, B

    def kinematics(self, state, vx, vy, vtheta, epsilon=0):

        f,_,_ = self.proc_model()

        #vx = vx + epsilon*self.vel_max*np.random.normal(0,1)
        #vy = vy + epsilon*self.vel_max*np.random.normal(0,1)
        #vtheta = vtheta + epsilon*self.ang_vel_max*np.random.normal(0,1)

        w0 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[0,0]))
        w1 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[1,1]))
        w2 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[2,2]))

        w = c.DM([[w0],[w1],[w2]])

        #state_n =  state + self.dt*c.blockcat([[vx],[vy],[vtheta]])

        state_n = f(state,c.blockcat([[vx],[vy],[vtheta]])) + c.mtimes(self.G,w)

        state_n[2] = c.atan2(c.sin(state_n[2]),c.cos(state_n[2]))

        return state_n

class car_model(object):

    """
    model specification and dynamics
    """

    def __init__(self, length=0.58,  breadth = 0.38, ang_vel_max=m.pi/6, vel_max=.8, dt=0.1, nx=4, nu=2):

        self.length = length
        self.breadth = breadth
        self.ang_vel_max = ang_vel_max
        #self.ang_accel_max = ang_accel_max
        self.vel_max = vel_max
        #self.accel_max = accel_max
        self.dt = dt
        self.nx = nx
        self.nu = nu
        # self.A = c.DM.eye(self.nx) #youBot has a linear model
        # self.B = c.DM.eye(self.nu)*self.dt
        # self.G = c.DM.eye(self.nu)*self.dt
        self.x = c.MX.sym('x', self.nx,1)
        self.u = c.MX.sym('u', self.nu,1)
        self.Sigma_w = c.DM([[self.vel_max**2,0],[0,self.ang_vel_max**2]])

    def proc_model(self):

        # f = c.Function('f',[self.x,self.u],[self.x[0] + self.u[0]*c.cos(self.x[2])*self.dt,
        #                                         self.x[1] + self.u[0]*c.sin(self.x[2])*self.dt,
        #                                         self.x[2] + self.u[0]*c.tan(self.x[3])*self.dt/self.length,
        #                                         self.x[3] + self.u[1]*self.dt])

        g = c.MX(self.nx,self.nu)
        g[0,0] = c.cos(self.x[2]); g[0,1] = 0;
        g[1,0] = c.sin(self.x[2]); g[1,1] = 0;
        g[2,0] = c.tan(self.x[3])/self.length; g[2,1] = 0
        g[3,0] = 0; g[3,1] = 1;

        # f = c.Function('f',[self.x,self.u],[self.x[0] + self.u[0]*c.cos(self.x[2])*self.dt,
        #                                         self.x[1] + self.u[0]*c.sin(self.x[2])*self.dt,
        #                                         self.x[2] + self.u[0]*c.tan(self.x[3])*self.dt/self.length,
        #                                         self.x[3] + self.u[1]*self.dt])

        f =  c.Function('f',[self.x,self.u],[self.x + c.mtimes(g,self.u)*self.dt])

        # A = c.Function('A',[self.x,self.u],[c.jacobian(f(self.x,self.u)[0],self.x),
        #                                         c.jacobian(f(self.x,self.u)[1],self.x),
        #                                         c.jacobian(f(self.x,self.u)[2],self.x),
        #                                         c.jacobian(f(self.x,self.u)[3],self.x)]) #linearization

        # B = c.Function('B',[self.x,self.u],[c.jacobian(f(self.x,self.u)[0],self.u),
        #                                         c.jacobian(f(self.x,self.u)[1],self.u),
        #                                         c.jacobian(f(self.x,self.u)[2],self.u),
        #                                         c.jacobian(f(self.x,self.u)[3],self.u)])

        A = c.Function('A',[self.x,self.u],[c.jacobian(f(self.x,self.u),self.x)])

        B = c.Function('B',[self.x,self.u],[c.jacobian(f(self.x,self.u),self.u)])

        return f,A, B

    def kinematics(self, state, U, epsilon=0):

        f,_,_ = self.proc_model()

        w0 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[0,0]))
        w1 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[1,1]))


        w = c.DM([[w0],[w1]])

        #state_n =  state + self.dt*c.blockcat([[vx],[vy],[vtheta]])

        state_n = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))

        # state_n = c.MX(self.nx,1)
        # state_n[0] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[0]
        # state_n[1] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[1]
        # state_n[2] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[2]
        # state_n[3] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[3]

        state_n[2] = c.atan2(c.sin(state_n[2]),c.cos(state_n[2]))

        return state_n

    # def kinematics_DM(self, state, U, epsilon=0):
    #
    #     f,_,_ = self.proc_model()
    #
    #     w0 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[0,0]))
    #     w1 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[1,1]))
    #
    #
    #     w = c.DM([[w0],[w1]])
    #
    #     #state_n =  state + self.dt*c.blockcat([[vx],[vy],[vtheta]])
    #
    #     state_n = c.DM(self.nx,1)
    #     state_n[0] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[0]
    #     state_n[1] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[1]
    #     state_n[2] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[2]
    #     state_n[3] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[3]
    #
    #     state_n[2] = c.atan2(c.sin(state_n[2]),c.cos(state_n[2]))
    #
    #     return state_n

class car_w_trailers(object):

    """
    model specification and dynamics
    """

    def __init__(self, length=0.58,  breadth = 0.38, trailer_length = 0.3, ang_vel_max=m.pi/3, vel_max=3, dt=0.1, nx=6, nu=2):

        self.length = length
        self.breadth = breadth
        self.trailer_length = trailer_length
        self.ang_vel_max = ang_vel_max
        #self.ang_accel_max = ang_accel_max
        self.vel_max = vel_max
        #self.accel_max = accel_max
        self.dt = dt
        self.nx = nx
        self.nu = nu
        # self.A = c.DM.eye(self.nx) #youBot has a linear model
        # self.B = c.DM.eye(self.nu)*self.dt
        # self.G = c.DM.eye(self.nu)*self.dt
        self.x = c.MX.sym('x', self.nx,1)
        self.u = c.MX.sym('u', self.nu,1)
        self.Sigma_w = c.DM([[self.vel_max**2,0],[0,self.ang_vel_max**2]])

    def proc_model(self):

        # f = c.Function('f',[self.x,self.u],[self.x[0] + self.u[0]*c.cos(self.x[2])*self.dt,
        #                                         self.x[1] + self.u[0]*c.sin(self.x[2])*self.dt,
        #                                         self.x[2] + self.u[0]*c.tan(self.x[3])*self.dt/self.length,
        #                                         self.x[3] + self.u[1]*self.dt])

        g = c.MX(self.nx,self.nu)
        g[0,0] = c.cos(self.x[2]); g[0,1] = 0;
        g[1,0] = c.sin(self.x[2]); g[1,1] = 0;
        g[2,0] = c.tan(self.x[3])/self.length; g[2,1] = 0
        g[3,0] = 0; g[3,1] = 1;
        g[4,0] = c.sin(self.x[2] - self.x[4])/self.trailer_length; g[4,1] = 0
        g[5,0] = c.cos(self.x[2] - self.x[4])*c.sin(self.x[4] - self.x[5])/self.trailer_length; g[5,1] = 0

        # f = c.Function('f',[self.x,self.u],[self.x[0] + self.u[0]*c.cos(self.x[2])*self.dt,
        #                                         self.x[1] + self.u[0]*c.sin(self.x[2])*self.dt,
        #                                         self.x[2] + self.u[0]*c.tan(self.x[3])*self.dt/self.length,
        #                                         self.x[3] + self.u[1]*self.dt])

        f =  c.Function('f',[self.x,self.u],[self.x + c.mtimes(g,self.u)*self.dt])

        # A = c.Function('A',[self.x,self.u],[c.jacobian(f(self.x,self.u)[0],self.x),
        #                                         c.jacobian(f(self.x,self.u)[1],self.x),
        #                                         c.jacobian(f(self.x,self.u)[2],self.x),
        #                                         c.jacobian(f(self.x,self.u)[3],self.x)]) #linearization

        # B = c.Function('B',[self.x,self.u],[c.jacobian(f(self.x,self.u)[0],self.u),
        #                                         c.jacobian(f(self.x,self.u)[1],self.u),
        #                                         c.jacobian(f(self.x,self.u)[2],self.u),
        #                                         c.jacobian(f(self.x,self.u)[3],self.u)])

        A = c.Function('A',[self.x,self.u],[c.jacobian(f(self.x,self.u),self.x)])

        B = c.Function('B',[self.x,self.u],[c.jacobian(f(self.x,self.u),self.u)])

        return f,A, B

    def kinematics(self, state, U, epsilon=0):

        f,_,_ = self.proc_model()

        w0 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[0,0]))
        w1 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[1,1]))


        w = c.DM([[w0],[w1]])

        #state_n =  state + self.dt*c.blockcat([[vx],[vy],[vtheta]])

        state_n = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))

        # state_n = c.MX(self.nx,1)
        # state_n[0] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[0]
        # state_n[1] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[1]
        # state_n[2] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[2]
        # state_n[3] = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]]]))[3]

        state_n[2] = c.atan2(c.sin(state_n[2]),c.cos(state_n[2]))
        state_n[4] = c.atan2(c.sin(state_n[4]),c.cos(state_n[4]))
        state_n[5] = c.atan2(c.sin(state_n[5]),c.cos(state_n[5]))

        return state_n
