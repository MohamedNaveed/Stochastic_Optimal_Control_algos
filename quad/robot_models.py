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

class quad(object):

        def __init__(self,torque_max=.05, thrust_max=1.5, dt=0.1, nx=12, nu=4):

                self.dt = dt
                self.nx = nx
                self.nu = nu
                self.thrust_max = thrust_max
                self.thrust_min = 0
                self.torque_max = torque_max
                self.x = c.MX.sym('x', self.nx,1)
                self.u = c.MX.sym('u', self.nu,1)
                self.Sigma_w = c.DM([[self.thrust_max**2,0,0,0],[0,self.torque_max**2,0,0],
                                        [0,0,self.torque_max**2,0],[0,0,0,self.torque_max**2]])

                self.Ic = 10**(-3)*c.DM([[4.86,0,0],[0,4.86,0],[0,0,8.8]])
                self.m = .5
                self.g = 9.81
                self.kd = .25

        def proc_model(self):

                g1 = c.MX.zeros(self.nx,self.nx)
                #assigning zeros
                # for i in range(self.nx):
                #         for j in range(self.nx):
                #                 g1[i,j] = 0

                g1[0,3] = 1; g1[1,4] = 1; g1[2,5] = 1;
                g1[6,9] = 1; g1[6,10] = c.sin(self.x[6])*c.tan(self.x[7]); g1[6,11] = c.cos(self.x[6])*c.tan(self.x[7]);
                g1[7,10] = c.cos(self.x[6]); g1[7,11] = -c.sin(self.x[6]);
                g1[8,10] = c.sin(self.x[6])/c.cos(self.x[7]); g1[8,11] = c.cos(self.x[6])/c.cos(self.x[7]);

                g2 = c.MX.zeros(self.nx,self.nu)

                g2[3,0] = (c.cos(self.x[8])*c.sin(self.x[7])*c.cos(self.x[6]) + c.sin(self.x[8])*c.sin(self.x[6]))/self.m
                g2[4,0] = (c.sin(self.x[8])*c.sin(self.x[7])*c.cos(self.x[6]) - c.cos(self.x[8])*c.sin(self.x[6]))/self.m
                g2[5,0] = c.cos(self.x[7])*c.cos(self.x[6])/self.m
                g2[9:,1:] = c.inv(self.Ic)

                g3 = c.MX.zeros(self.nx,1)

                f =  c.Function('f',[self.x,self.u],[self.x + c.mtimes(g1,self.x)*self.dt + c.mtimes(g2,self.u)*self.dt - g3*self.g*self.dt])

                A = c.Function('A',[self.x,self.u],[c.jacobian(f(self.x,self.u),self.x)])

                B = c.Function('B',[self.x,self.u],[c.jacobian(f(self.x,self.u),self.u)])

                return f,A, B

        def kinematics(self, state, U, epsilon=0):

            f,_,_ = self.proc_model()

            w0 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[0,0]))
            w1 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[1,1]))
            w2 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[2,2]))
            w3 = epsilon*np.random.normal(0,np.sqrt(self.Sigma_w[3,3]))

            w = c.DM([[w0],[w1],[w2],[w3]])

            state_n = f(state,c.blockcat([[U[0] + w[0]],[U[1] + w[1]],[U[2] + w[2]],[U[3] + w[3]]]))

            state_n[6] = c.atan2(c.sin(state_n[6]),c.cos(state_n[6]))
            state_n[7] = c.atan2(c.sin(state_n[7]),c.cos(state_n[7]))
            state_n[8] = c.atan2(c.sin(state_n[8]),c.cos(state_n[8]))

            return state_n
