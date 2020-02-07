# Author: Mohamed Naveed Gul Mohamed
# email:mohdnaveed96@gmail.com
# Date: Sept 24th 2019
#
# Classes and functions for Stochastic optimal control.

import casadi as c
import math as m
import numpy as np
import simulation_params as params

#State space planning
class SSP(object):

    """
    classes and functions for implementing optimal controller
    """

    def __init__(self, time_horizon, X0, Xg, R, Q, Qf, Xmin, Xmax):

        self.T = time_horizon
        self.X0 = X0
        self.Xg = Xg
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.R = R
        self.Q = Q
        self.Qf = Qf

    def solve_SSP(self, robot, X0, Ui, U_guess):

        opti = c.Opti()

        U = opti.variable(robot.nu*self.T,1)
        opti.set_initial(U, U_guess)

        opti.minimize(self.cost_func_SSP(robot, U, X0))

        #control constraints
        opti.subject_to(U <= self.U_upper_bound(robot))
        opti.subject_to(U >= self.U_lower_bound(robot))

        #state constraints
        #opti.subject_to(self.state_contraints(robot,U) > 0)

        p_opts = {}
        p_opts['ipopt.print_level'] = 0
        s_opts = {"max_iter": 100}
        opti.solver('ipopt', p_opts,s_opts)

        try:
                sol = opti.solve()
                U_opti = sol.value(U)

        except RuntimeError:
                U_opti = opti.debug.value(U)
                print "debug value used."

        # sol = opti.solve()
        # U_opti = sol.value(U)
        U_opti = c.reshape(U_opti,robot.nu,self.T)

        return U_opti

    def cost_func_SSP(self, robot, U, X0):

        cost = 0

        U = c.reshape(U, robot.nu, self.T)

        X = c.MX(robot.nx,self.T+1)
        X[:,0] = X0

        for i in range(self.T):

                cost = (cost + c.mtimes(c.mtimes(U[:,i].T,self.R),U[:,i]) +
                         c.mtimes(c.mtimes((self.Xg - X[:,i]).T,self.Q),(self.Xg - X[:,i])))

                X_temp = X[0:2,i]


                if params.OBSTACLES:

                        obstacle_cost =  params.M*(c.exp(-(c.mtimes(c.mtimes((params.c_obs_1 - X_temp).T, params.E_obs_1),(params.c_obs_1 - X_temp))))+ \
                                                c.exp(-(c.mtimes(c.mtimes((params.c_obs_2 - X_temp).T, params.E_obs_2),(params.c_obs_2 - X_temp)))) + \
                                                c.exp(-(c.mtimes(c.mtimes((params.c_obs_3 - X_temp).T, params.E_obs_3),(params.c_obs_3 - X_temp)))) + \
                                                c.exp(-(c.mtimes(c.mtimes((params.c_obs_4 - X_temp).T, params.E_obs_4),(params.c_obs_4 - X_temp)))))


                        cost = cost + obstacle_cost
                X[:,i+1] = robot.kinematics(X[:,i],U[:,i])

        cost = cost + c.mtimes(c.mtimes((self.Xg - X[:,self.T]).T,self.Qf),(self.Xg - X[:,self.T]))

        return cost

    def U_upper_bound(self, robot):
        ones = c.DM.ones(robot.nu,self.T)
        ub = c.blockcat([[robot.thrust_max*ones[0,:]],[robot.torque_max*ones[1,:]],
                [robot.torque_max*ones[2,:]],[robot.torque_max*ones[3,:]]])
        ub = c.reshape(ub,robot.nu*self.T,1)

        return ub

    def U_lower_bound(self, robot):
        ones = c.DM.ones(robot.nu,self.T)
        lb = c.blockcat([[robot.thrust_min*ones[0,:]],[-robot.torque_max*ones[1,:]],
                [-robot.torque_max*ones[2,:]],[-robot.torque_max*ones[3,:]]])
        lb = c.reshape(lb,robot.nu*self.T,1)

        return lb

    def state_contraints(self, robot, U):

        constraintVar = c.MX(2*2,self.T) #skipping orientation & steer angle. 2* to include min and max
        U = c.reshape(U,robot.nu,self.T)
        X = c.MX(robot.nx,self.T+1)
        X[:,0] = self.X0

        for i in range(self.T):
            X[:,i+1] = robot.kinematics(X[:,i],U[:,i])
            constraintVar[:,i] = c.blockcat([[self.Xmax - X[0:2,i+1]],[X[0:2,i+1] - self.Xmin]])

        constraintVar = c.reshape(constraintVar,1,2*2*self.T)

        return constraintVar

#Multi-agent State space planning
class M_SSP(object):

    """
    classes and functions for implementing optimal controller
    """

    def __init__(self, time_horizon, N, X0, cov_X0, Xg, R, Q, Qf, Xmin, Xmax):

        self.T = time_horizon
        self.N = N
        self.X0 = X0
        self.cov_X0 = cov_X0
        self.Xg = Xg
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.R = R
        self.Q = Q
        self.Qf = Qf

    def solve_SSP(self, robot, X0, Ui, U_guess):

        opti = c.Opti()

        U = opti.variable(robot.nu*self.T*self.N,1)
        opti.set_initial(U, U_guess)

        opti.minimize(self.cost_func_SSP(robot, U, X0))

        #control constraints
        opti.subject_to(U <= self.U_upper_bound(robot))
        opti.subject_to(U >= self.U_lower_bound(robot))

        #state constraints
        opti.subject_to(self.state_contraints(robot,U) > 0)

        opts = {}
        opts['ipopt.print_level'] = 0
        opti.solver('ipopt', opts)
        sol = opti.solve()
        U_opti = sol.value(U)
        U_opti = c.reshape(U_opti,robot.nu*self.N,self.T)

        return U_opti

    def cost_func_SSP(self, robot, U, X0):

        cost = 0

        U = c.reshape(U, robot.nu*self.N, self.T)

        X = c.MX(robot.nx*self.N,self.T+1)

        X[:,0] = np.reshape(X0,(robot.nx*self.N,))

        for i in range(self.T):

                for n in range(self.N):

                        cost = (cost + c.mtimes(c.mtimes(U[robot.nu*n:robot.nu*(n+1),i].T,self.R),U[robot.nu*n:robot.nu*(n+1),i]) +
                                 c.mtimes(c.mtimes((self.Xg[n,:] - X[robot.nx*n:robot.nx*(n+1),i]).T,self.Q),(self.Xg[n,:] - X[robot.nx*n:robot.nx*(n+1),i])))

                        X_temp = X[robot.nx*n:robot.nx*(n+1)-1,i] #getting x,y position


                        if params.OBSTACLES:

                                obstacle_cost =  self.obstacle_cost_func(X_temp)

                                cost = cost + obstacle_cost

                        X[robot.nx*n:robot.nx*(n+1),i+1] = robot.kinematics(X[robot.nx*n:robot.nx*(n+1),i],U[robot.nu*n,i],U[robot.nu*n+1,i],U[robot.nu*n+2,i])

                if self.N > 1:

                        inter_agent_cost = self.inter_agent_cost_func(robot,X[:,i])
                        cost = cost + inter_agent_cost


        for n in range(self.N):

                cost = cost + c.mtimes(c.mtimes((self.Xg[n,:] - X[robot.nx*n:robot.nx*(n+1),self.T]).T,self.Qf),(self.Xg[n,:] - X[robot.nx*n:robot.nx*(n+1),self.T]))

        return cost

    def U_upper_bound(self, robot):
        ones = c.DM.ones(self.N*self.T)
        ub = c.blockcat([[robot.vel_max*ones],[robot.vel_max*ones],[robot.ang_vel_max*ones]])
        ub = c.reshape(ub,robot.nu*self.N*self.T,1)

        return ub

    def U_lower_bound(self, robot):
        ones = c.DM.ones(self.N*self.T)
        lb = c.blockcat([[-robot.vel_max*ones],[-robot.vel_max*ones],[-robot.ang_vel_max*ones]])
        lb = c.reshape(lb,robot.nu*self.N*self.T,1)

        return lb

    def state_contraints(self, robot, U):

        constraintVar = c.MX(2*(robot.nx-1)*self.N,self.T) #skipping orientation. 2* to include min and max
        U = c.reshape(U,robot.nu*self.N,self.T)
        X = c.MX(robot.nx*self.N,self.T+1)
        X[:,0] = np.reshape(self.X0,(robot.nx*self.N,))

        for i in range(self.T):
            for n in range(self.N):
                X[robot.nx*n:robot.nx*(n+1),i+1] = robot.kinematics(X[robot.nx*n:robot.nx*(n+1),i],U[robot.nu*n,i],U[robot.nu*n+1,i],U[robot.nu*n+2,i])
                constraintVar[2*(robot.nx-1)*n:2*(robot.nx-1)*(n+1),i] = c.blockcat([[self.Xmax - X[robot.nx*n:robot.nx*(n+1)-1,i+1]],[X[robot.nx*n:robot.nx*(n+1)-1,i+1] - self.Xmin]])

        constraintVar = c.reshape(constraintVar,1,2*(robot.nx-1)*self.N*self.T)

        return constraintVar

    def obstacle_cost_func(self, X_temp):

        cost =  params.M*(c.exp(-(c.mtimes(c.mtimes((params.c_obs_1 - X_temp).T, params.E_obs_1),(params.c_obs_1 - X_temp))))+ \
                                c.exp(-(c.mtimes(c.mtimes((params.c_obs_2 - X_temp).T, params.E_obs_2),(params.c_obs_2 - X_temp)))) + \
                                c.exp(-(c.mtimes(c.mtimes((params.c_obs_3 - X_temp).T, params.E_obs_3),(params.c_obs_3 - X_temp)))) + \
                                c.exp(-(c.mtimes(c.mtimes((params.c_obs_4 - X_temp).T, params.E_obs_4),(params.c_obs_4 - X_temp)))))

        return cost

    def inter_agent_cost_func(self,robot,X):

        inter_agent_cost = 0
        for j in range(self.N-1):

            for h in range(self.N-j-1):

                agent_1 = X[robot.nx*j:robot.nx*(j+1)]
                agent_2 = X[robot.nx*(j+(h+1)):robot.nx*(j+(h+1)+1)]

                inter_agent_cost = inter_agent_cost + (params.SF*c.exp(-((agent_1[0] - agent_2[0])**2 +
                                    (agent_1[1] - agent_2[1])**2 - (2*params.r_th)**2)))

        return inter_agent_cost

#Belief space planning
class BSP(object):

        def __init__(self, robot, time_horizon, X0, cov_X0, Xg, R, Q, Qf, Xmin, Xmax, gamma, epsilon=0):

            self.T = time_horizon
            self.X0 = X0
            self.cov_X0 = cov_X0
            self.Xg = Xg
            self.Xmin = Xmin
            self.Xmax = Xmax
            self.R = R
            self.Q = Q
            self.Qf = Qf
            self.gamma = gamma
            self.epsilon = epsilon
            self.Sigma_w = robot.Sigma_w
            self.Sigma_nu = c.DM([[.1,0,0],[0,.1,0],[0,0,.1]]) #+ c.DM.eye(robot.nx)
            self.x = c.MX.sym('x', robot.nx,1)
            self.u = c.MX.sym('u', robot.nu,1)

        def solve_BSP(self, robot, X0, cov_X0, Ui, U_guess): #Belief space planning

            opti = c.Opti()

            U = opti.variable(robot.nu*self.T,1)
            opti.set_initial(U, U_guess)

            opti.minimize(self.cost_func_BSP(robot, U, X0, cov_X0))

            #control constraints
            opti.subject_to(U <= self.U_upper_bound(robot))
            opti.subject_to(U >= self.U_lower_bound(robot))

            #state constraints
            opti.subject_to(self.state_contraints(robot,U) > 0)

            opts = {}
            opts['ipopt.print_level'] = 0
            opti.solver('ipopt',opts)
            sol = opti.solve()
            U_opti = sol.value(U)
            U_opti = c.reshape(U_opti,robot.nu,self.T)

            return U_opti

        def cost_func_BSP(self, robot, U, X0, cov_X0):

            cost = 0

            U = c.reshape(U, robot.nu, self.T)

            X = c.MX(robot.nx,self.T+1)
            X[:,0] = X0
            P = cov_X0

            for i in range(self.T):

                    X[:,i+1] = robot.kinematics(X[:,i],U[0,i],U[1,i],U[2,i])

                    H, M = self.light_dark_MX(X[:,i+1])

                    Sigma_w = (self.epsilon**2)*self.Sigma_w
                    Sigma_nu = (self.epsilon**2)*self.Sigma_nu

                    P = c.mtimes(c.mtimes(robot.A,P),robot.A.T) + c.mtimes(c.mtimes(robot.G,Sigma_w),robot.G.T)
                    S = c.mtimes(c.mtimes(H,P),H.T) + c.mtimes(c.mtimes(M,Sigma_nu),M.T)
                    K = c.mtimes(c.mtimes(P,H.T),c.inv(S))
                    P = c.mtimes(c.DM.eye(robot.nx) - c.mtimes(K,H), P)

                    cost = cost + self.gamma*c.trace(c.mtimes(c.mtimes(self.Q,P),self.Q.T)) + c.mtimes(c.mtimes(U[:,i].T,self.R),U[:,i])

                    X_temp = X[0:2,i]
                    if params.OBSTACLES:

                            obstacle_cost =  self.obstacle_cost_func(X_temp)
                            cost = cost + obstacle_cost

            cost = cost + c.mtimes(c.mtimes((self.Xg - X[:,self.T]).T,self.Qf),(self.Xg - X[:,self.T]))

            return cost


        def U_upper_bound(self, robot):
            ones = c.DM.ones(self.T)
            ub = c.blockcat([[robot.vel_max*ones],[robot.vel_max*ones],[robot.ang_vel_max*ones]])
            ub = c.reshape(ub,robot.nu*self.T,1)

            return ub

        def U_lower_bound(self, robot):
            ones = c.DM.ones(self.T)
            lb = c.blockcat([[-robot.vel_max*ones],[-robot.vel_max*ones],[-robot.ang_vel_max*ones]])
            lb = c.reshape(lb,robot.nu*self.T,1)

            return lb

        def state_contraints(self, robot, U):

            constraintVar = c.MX(2*(robot.nx-1),self.T) #skipping orientation. 2* to include min and max
            U = c.reshape(U,robot.nu,self.T)
            X = c.MX(robot.nx,self.T+1)
            X[:,0] = self.X0

            for i in range(self.T):
                X[:,i+1] = robot.kinematics(X[:,i],U[0,i],U[1,i],U[2,i])
                constraintVar[:,i] = c.blockcat([[self.Xmax - X[0:2,i+1]],[X[0:2,i+1] - self.Xmin]])

            constraintVar = c.reshape(constraintVar,1,2*(robot.nx-1)*self.T)

            return constraintVar

        def obstacle_cost_func(self, X_temp):

            cost =  params.M*(c.exp(-(c.mtimes(c.mtimes((params.c_obs_1 - X_temp).T, params.E_obs_1),(params.c_obs_1 - X_temp))))+ \
                                    c.exp(-(c.mtimes(c.mtimes((params.c_obs_2 - X_temp).T, params.E_obs_2),(params.c_obs_2 - X_temp)))) + \
                                    c.exp(-(c.mtimes(c.mtimes((params.c_obs_3 - X_temp).T, params.E_obs_3),(params.c_obs_3 - X_temp)))) + \
                                    c.exp(-(c.mtimes(c.mtimes((params.c_obs_4 - X_temp).T, params.E_obs_4),(params.c_obs_4 - X_temp)))))

            return cost

        def solve_K(self, robot, X, U, Wx, Wxf, Wu):

            K = c.DM.zeros(robot.nu**2,self.T+1)

            for i in range(self.T,0,-1):

                if i == self.T:
                        P = Wxf

                else:
                        P = c.mtimes(c.mtimes(robot.A.T,P),robot.A) - c.mtimes(c.mtimes(c.mtimes(robot.A.T,P),robot.B),K_mat) + Wx

                K_mat = c.mtimes(c.inv(Wu + c.mtimes(c.mtimes(robot.B.T,P),robot.B)),c.mtimes(c.mtimes(robot.B.T,P),robot.A))
                #print("P:",P)
                #print("K:",K_mat)
                K[:,i] = c.reshape(K_mat,robot.nu**2,1)

            return K

        def light_dark(self,X):

                h = c.Function('h',[self.x],[self.x])
                H = c.Function('H',[self.x],[c.jacobian(h(self.x),self.x)])

                M = c.DM([[(X[0] - 3)**2,0,0],[0,1,0],[0,0,1]])

                return h, H(X), M

        def light_dark_MX(self,X):

                h = c.Function('h',[self.x],[self.x])
                H = c.Function('H',[self.x],[c.jacobian(h(self.x),self.x)])

                #H = c.Function('H',[self.x],[c.MX.eye(X.shape[0])])

                M = c.MX(X.shape[0],X.shape[0])
                M[0,0] = (X[0] - 3)**2
                M[0,1] = 0; M[0,2] = 0;
                M[1,0] = 0; M[1,1] = 1; M[1,2] = 0;
                M[2,0] = 0; M[2,1] = 0; M[2,2] = 1;

                return H(X), M


        def observation(self,obs_model, X, Sigma_nu):

                h, _, M = obs_model(X)

                nu0 = np.random.normal(0,np.sqrt(Sigma_nu[0,0]))
                nu1 = np.random.normal(0,np.sqrt(Sigma_nu[1,1]))
                nu2 = np.random.normal(0,np.sqrt(Sigma_nu[2,2]))

                nu = c.DM([[nu0],[nu1],[nu2]])

                Y = h(X) + c.mtimes(M,nu)  #observation model

                return Y

        def Kalman_filter(self, robot, obs_model, X_prev_est, X_prev_act, U, P_prev):

            #Prediction
            X_prior = robot.kinematics(X_prev_est,U[0],U[1],U[2],0)
            X_prior = np.reshape(X_prior,(robot.nx,))

            Sigma_w = (self.epsilon**2)*self.Sigma_w
            Sigma_nu = (self.epsilon**2)*self.Sigma_nu

            _,A,B = robot.proc_model()
            robot.A = A(X_prev_est,U)
            #print("A:",robot.A," K:",K_fb)
            P_prior = c.mtimes(c.mtimes(robot.A,P_prev),robot.A.T) + c.mtimes(c.mtimes(robot.G,Sigma_w),robot.G.T)

            #Update
            X_act = robot.kinematics(X_prev_act,U[0],U[1],U[2],self.epsilon)
            X_act = np.reshape(X_act,(robot.nx,)) #to suit the receiving array

            Y_act = self.observation(obs_model,X_act,Sigma_nu)
            Y_est = self.observation(obs_model,X_prior,np.zeros((robot.nx,robot.nx)))

            _, H, M = obs_model(X_prior)

            S = c.mtimes(c.mtimes(H,P_prior),H.T) + c.mtimes(c.mtimes(M,Sigma_nu),M.T)
            K_gain = c.mtimes(c.mtimes(P_prior,H.T),c.inv(S))
            P_post = c.mtimes(c.DM.eye(robot.nx)-c.mtimes(K_gain,H), P_prior)

            X_est = X_prior + c.mtimes(K_gain,Y_act - Y_est)
            X_est = np.reshape(X_est,(robot.nx,))

            return X_est, X_act, P_post

        def U_bounds(self,robot,U):

            if U[0] > robot.vel_max:
                U[0] = robot.vel_max

            elif U[0] < -robot.vel_max:
                U[0] = -robot.vel_max

            if U[1] > robot.vel_max:
                U[1] = robot.vel_max

            elif U[1] < -robot.vel_max:
                U[1] = -robot.vel_max

            if U[2] > robot.ang_vel_max:
                U[2] = robot.ang_vel_max

            elif U[2] < -robot.ang_vel_max:
                U[2] = -robot.ang_vel_max

            return U

class M_BSP(object):

        def __init__(self, robot, N, time_horizon, X0, cov_X0, Xg, R, Q, Qf, Xmin, Xmax, gamma, epsilon=0):

            self.T = time_horizon
            self.N = N
            self.X0 = X0
            self.cov_X0 = cov_X0
            self.Xg = Xg
            self.Xmin = Xmin
            self.Xmax = Xmax
            self.R = R
            self.Q = Q
            self.Qf = Qf
            self.gamma = gamma
            self.epsilon = epsilon
            self.Sigma_w = c.DM([[robot.vel_max**2,0,0],[0,robot.vel_max**2,0],[0,0,robot.ang_vel_max**2]])
            self.Sigma_nu = c.DM([[.1,0,0],[0,.1,0],[0,0,.1]]) #+ c.DM.eye(robot.nx)

        def solve_BSP(self, robot, X0, cov_X0, Ui, U_guess): #Belief space planning

            opti = c.Opti()

            U = opti.variable(robot.nu*self.T*self.N,1)
            opti.set_initial(U, U_guess)

            opti.minimize(self.cost_func_BSP(robot, U, X0, cov_X0))

            #control constraints
            opti.subject_to(U <= self.U_upper_bound(robot))
            opti.subject_to(U >= self.U_lower_bound(robot))

            #state constraints
            opti.subject_to(self.state_contraints(robot,U) > 0)

            opts = {}
            opts['ipopt.print_level'] = 0
            opti.solver('ipopt',opts)
            sol = opti.solve()
            U_opti = sol.value(U)
            U_opti = c.reshape(U_opti,robot.nu*self.N,self.T)

            return U_opti

        def cost_func_BSP(self, robot, U, X0, cov_X0):

            cost = 0

            U = c.reshape(U, robot.nu*self.N, self.T)

            X = c.MX(robot.nx*self.N,self.T+1)
            X[:,0] = np.reshape(X0,(robot.nx*self.N,))

            for i in range(self.T):

                for n in range(self.N):

                    X[robot.nx*n:robot.nx*(n+1),i+1] = robot.kinematics(X[robot.nx*n:robot.nx*(n+1),i],U[robot.nu*n,i],U[robot.nu*n+1,i],U[robot.nu*n+2,i])

                    X_temp = X[robot.nx*n:robot.nx*(n+1),i+1]

                    if i==0:
                        P_temp = cov_X0[robot.nx*n:robot.nx*(n+1),:]

                        if n == 0:
                            P = c.MX(robot.nx*self.N,robot.nx)

                    else:
                        P_temp = P[robot.nx*n:robot.nx*(n+1),:]


                    M = c.MX(robot.nx,robot.nx)
                    M[0,0] = (X_temp[0] - 3)**2 #1/(c.mtimes(X[0,i+1],X[0,i+1])+1);
                    M[0,1] = 0; M[0,2] = 0;
                    M[1,0] = 0; M[1,1] = 1; M[1,2] = 0;
                    M[2,0] = 0; M[2,1] = 0; M[2,2] = 1;
                    #M = c.DM.eye(robot.nx)

                    Sigma_w = (self.epsilon**2)*self.Sigma_w
                    Sigma_nu = (self.epsilon**2)*self.Sigma_nu

                    P_temp = P_temp + c.mtimes(c.mtimes(robot.G,Sigma_w),robot.G.T)
                    S = P_temp + c.mtimes(c.mtimes(M,Sigma_nu),M.T)
                    K = c.mtimes(P_temp,c.inv(S))
                    P_temp = c.mtimes(c.DM.eye(robot.nx) - K, P_temp)


                    P[robot.nx*n:robot.nx*(n+1),:] = P_temp

                    cost = cost + self.gamma*c.trace(c.mtimes(c.mtimes(self.Q,P_temp),self.Q.T)) + \
                                c.mtimes(c.mtimes(U[robot.nu*n:robot.nu*(n+1),i].T,self.R),U[robot.nu*n:robot.nu*(n+1),i])


                    if params.OBSTACLES:

                            obstacle_cost =  self.obstacle_cost_func(X_temp[0:2])
                            cost = cost + obstacle_cost

                if self.N > 1:

                    inter_agent_cost = self.inter_agent_cost_func(robot,X[:,i])
                    cost = cost + inter_agent_cost

            for n in range(self.N):
                cost = cost + c.mtimes(c.mtimes((self.Xg[n,:] - X[robot.nx*n:robot.nx*(n+1),self.T]).T,self.Qf),(self.Xg[n,:] - X[robot.nx*n:robot.nx*(n+1),self.T]))

            return cost


        def U_upper_bound(self, robot):
            ones = c.DM.ones(self.N*self.T)
            ub = c.blockcat([[robot.vel_max*ones],[robot.vel_max*ones],[robot.ang_vel_max*ones]])
            ub = c.reshape(ub,robot.nu*self.N*self.T,1)

            return ub

        def U_lower_bound(self, robot):
            ones = c.DM.ones(self.N*self.T)
            lb = c.blockcat([[-robot.vel_max*ones],[-robot.vel_max*ones],[-robot.ang_vel_max*ones]])
            lb = c.reshape(lb,robot.nu*self.N*self.T,1)

            return lb

        def state_contraints(self, robot, U):

            constraintVar = c.MX(2*(robot.nx-1)*self.N,self.T) #skipping orientation. 2* to include min and max
            U = c.reshape(U,robot.nu*self.N,self.T)
            X = c.MX(robot.nx*self.N,self.T+1)
            X[:,0] =np.reshape(self.X0,(robot.nx*self.N,))

            for i in range(self.T):
                for n in range(self.N):
                    X[robot.nx*n:robot.nx*(n+1),i+1] = robot.kinematics(X[robot.nx*n:robot.nx*(n+1),i],U[robot.nu*n,i],U[robot.nu*n+1,i],U[robot.nu*n+2,i])
                    constraintVar[2*(robot.nx-1)*n:2*(robot.nx-1)*(n+1),i] = c.blockcat([[self.Xmax - X[robot.nx*n:robot.nx*(n+1)-1,i+1]],[X[robot.nx*n:robot.nx*(n+1)-1,i+1] - self.Xmin]])

            constraintVar = c.reshape(constraintVar,1,2*(robot.nx-1)*self.N*self.T)

            return constraintVar

        def obstacle_cost_func(self, X_temp):

            cost =  params.M*(c.exp(-(c.mtimes(c.mtimes((params.c_obs_1 - X_temp).T, params.E_obs_1),(params.c_obs_1 - X_temp))))+ \
                                    c.exp(-(c.mtimes(c.mtimes((params.c_obs_2 - X_temp).T, params.E_obs_2),(params.c_obs_2 - X_temp)))) + \
                                    c.exp(-(c.mtimes(c.mtimes((params.c_obs_3 - X_temp).T, params.E_obs_3),(params.c_obs_3 - X_temp)))) + \
                                    c.exp(-(c.mtimes(c.mtimes((params.c_obs_4 - X_temp).T, params.E_obs_4),(params.c_obs_4 - X_temp)))))

            return cost

        def inter_agent_cost_func(self,robot,X):

                inter_agent_cost = 0
                for j in range(self.N-1):

                    for h in range(self.N-j-1):

                        agent_1 = X[robot.nx*j:robot.nx*(j+1)]
                        agent_2 = X[robot.nx*(j+(h+1)):robot.nx*(j+(h+1)+1)]

                        inter_agent_cost = inter_agent_cost + (params.SF*c.exp(-((agent_1[0] - agent_2[0])**2 +
                                            (agent_1[1] - agent_2[1])**2 - (2*params.r_th)**2)))

                return inter_agent_cost


        def solve_K(self, robot, X, U, Wx, Wxf, Wu):

            K = c.DM.zeros(robot.nu**2,self.T+1)

            for i in range(self.T,0,-1):

                if i == self.T:
                        P = Wxf

                else:
                        P = c.mtimes(c.mtimes(robot.A.T,P),robot.A) - c.mtimes(c.mtimes(c.mtimes(robot.A.T,P),robot.B),K_mat) + Wx

                K_mat = c.mtimes(c.inv(Wu + c.mtimes(c.mtimes(robot.B.T,P),robot.B)),c.mtimes(c.mtimes(robot.B.T,P),robot.A))
                #print("P:",P)
                #print("K:",K_mat)
                K[:,i] = c.reshape(K_mat,robot.nu**2,1)

            return K

        def Kalman_filter(self, robot, X_prev_est, X_prev_act, U, P_prev):

            #Prediction
            X_prior = robot.kinematics(X_prev_est,U[0],U[1],U[2],0)

            Sigma_w = (self.epsilon**2)*self.Sigma_w
            Sigma_nu = (self.epsilon**2)*self.Sigma_nu

            P_prior = c.mtimes(c.mtimes(robot.A,P_prev),robot.A.T) + c.mtimes(c.mtimes(robot.G,Sigma_w),robot.G.T)

            #Update
            M = c.DM([[(X_prev_act[0] - 3)**2,0,0],[0,1,0],[0,0,1]])

            S = P_prior + c.mtimes(c.mtimes(M,Sigma_nu),M.T)
            K_gain = c.mtimes(P_prior,c.inv(S))
            P_post = c.mtimes(c.DM.eye(robot.nx)-K_gain, P_prior)

            X_act = robot.kinematics(X_prev_act,U[0],U[1],U[2],self.epsilon)
            X_act = np.reshape(X_act,(robot.nx,)) #to suit the receiving array

            nu0 = np.random.normal(0,np.sqrt(Sigma_nu[0,0]))
            nu1 = np.random.normal(0,np.sqrt(Sigma_nu[1,1]))
            nu2 = np.random.normal(0,np.sqrt(Sigma_nu[2,2]))

            nu = c.DM([[nu0],[nu1],[nu2]])

            Y = X_act + c.mtimes(M,nu)

            X_est = X_prior + c.mtimes(K_gain,Y - X_prior)
            X_est = np.reshape(X_est,(robot.nx,))

            return X_est, X_act, P_post

        def U_bounds(self,robot,U):

            if U[0] > robot.vel_max:
                U[0] = robot.vel_max

            elif U[0] < -robot.vel_max:
                U[0] = -robot.vel_max

            if U[1] > robot.vel_max:
                U[1] = robot.vel_max

            elif U[1] < -robot.vel_max:
                U[1] = -robot.vel_max

            if U[2] > robot.ang_vel_max:
                U[2] = robot.ang_vel_max

            elif U[2] < -robot.ang_vel_max:
                U[2] = -robot.ang_vel_max

            return U

class LQR(object):

        def __init__(self, Wx, Wu, Wxf, T):

                self.T = T
                self.Wx = Wx
                self.Wu = Wu
                self.Wxf = Wxf

        def solve_K(self, robot, X, U):

            K = c.DM.zeros(robot.nu*robot.nx,self.T+1)

            _,A,B = robot.proc_model()


            for i in range(self.T,0,-1):

                At = A(X[:,i],U[:,i-1])
                Bt = B(X[:,i],U[:,i-1])

                #print "At:", At
                #print "Bt:", Bt
                if i == self.T:
                        P = self.Wxf

                else:

                        P = c.mtimes(c.mtimes(At.T,P),At) - c.mtimes(c.mtimes(c.mtimes(At.T,P),Bt),K_mat) + self.Wx

                K_mat = c.mtimes(c.inv(self.Wu + c.mtimes(c.mtimes(Bt.T,P),Bt)),c.mtimes(c.mtimes(Bt.T,P),At))

                K[:,i] = c.reshape(K_mat,robot.nu*robot.nx,1)

            return K

        def U_bounds(self,robot,U):

            if U[0] > robot.thrust_max:
                U[0] = robot.thrust_max

            elif U[0] < robot.thrust_min:
                U[0] = robot.thrust_min

            for i in range(1,robot.nu):
                if U[i] > robot.torque_max:
                        U[i] = robot.torque_max

                elif U[i] < -robot.torque_max:
                        U[i] = -robot.torque_max

            return U

        def obstacle_cost_func(self, X_temp):

            cost =  params.M*(c.exp(-(c.mtimes(c.mtimes((params.c_obs_1 - X_temp).T, params.E_obs_1),(params.c_obs_1 - X_temp))))+ \
                                    c.exp(-(c.mtimes(c.mtimes((params.c_obs_2 - X_temp).T, params.E_obs_2),(params.c_obs_2 - X_temp)))) + \
                                    c.exp(-(c.mtimes(c.mtimes((params.c_obs_3 - X_temp).T, params.E_obs_3),(params.c_obs_3 - X_temp)))) + \
                                    c.exp(-(c.mtimes(c.mtimes((params.c_obs_4 - X_temp).T, params.E_obs_4),(params.c_obs_4 - X_temp)))))

            return cost

class PFC(object):

        def __init__(self, Wx, Wu, Wxf, T, Xg):

                self.T = T
                self.Wx = Wx
                self.Wu = Wu
                self.Wxf = Wxf
                self.Xg = Xg

        def solve_K(self, robot, X, U):

            K = c.DM.zeros(robot.nu*robot.nx,self.T+1)

            f,A,B = robot.proc_model()

            f_xx = c.Function('f_xx',[robot.x,robot.u],[c.jacobian(c.jacobian(f(robot.x,robot.u),robot.x).T,robot.x)])
            f_xu = c.Function('f_xu',[robot.x,robot.u],[c.jacobian(c.jacobian(f(robot.x,robot.u),robot.x).T,robot.u)])
            # B = c.Function('B',[self.x,self.u],[c.jacobian(f(self.x,self.u),self.u)])

            for i in range(self.T,0,-1):

                At = A(X[:,i],U[:,i-1])
                Bt = B(X[:,i],U[:,i-1])

                f_xxt = f_xx(X[:,i],U[:,i-1])
                f_xut = f_xu(X[:,i],U[:,i-1])
                #print "At:", At
                #print "Bt:", Bt
                if i == self.T:

                        P = 2*self.Wxf #2*
                        G = -2*c.mtimes(self.Wxf,self.Xg - X[:,i]) #2*
                        G = G.T
                else:
                        Lt = -2*c.mtimes(self.Wx,self.Xg - X[:,i]) #2*
                        Lt = Lt.T
                        Ltt = 2*self.Wxf #2*


                        tensor_product_xx = c.mtimes(G,f_xxt[0:robot.nx,:])

                        for p in range(1,robot.nx):
                                tensor_product_xx = c.vertcat(tensor_product_xx,c.mtimes(G,f_xxt[p*robot.nx:(p+1)*robot.nx,:]))

                        #print "tensor_product_xx:", tensor_product_xx

                        P = Ltt + c.mtimes(c.mtimes(At.T,P),At) - c.mtimes(c.mtimes(K_mat.T,S),K_mat) + tensor_product_xx
                        G = Lt + c.mtimes(G,At)

                #print "G:", G
                #print "P:", P
                S = self.Wu + c.mtimes(c.mtimes(Bt.T,P),Bt)

                tensor_product_xu = c.mtimes(G,f_xut[0:robot.nx,:])

                for p in range(1,robot.nx):
                        tensor_product_xu = c.vertcat(tensor_product_xu,c.mtimes(G,f_xut[p*robot.nx:(p+1)*robot.nx,:]))

                #print "tensor_product_xu:", tensor_product_xu

                K_mat = c.mtimes(c.inv(S),c.mtimes(c.mtimes(Bt.T,P),At))# + tensor_product_xu.T)

                #print "K:", K_mat
                K[:,i] = c.reshape(K_mat,robot.nu*robot.nx,1)

            return K

        def U_bounds(self,robot,U):

            if U[0] > robot.vel_max:
                U[0] = robot.vel_max

            elif U[0] < -robot.vel_max:
                U[0] = -robot.vel_max

            # if U[1] > robot.vel_max:
            #     U[1] = robot.vel_max
            #
            # elif U[1] < -robot.vel_max:
            #     U[1] = -robot.vel_max

            if U[1] > robot.ang_vel_max:
                U[1] = robot.ang_vel_max

            elif U[1] < -robot.ang_vel_max:
                U[1] = -robot.ang_vel_max

            return U
