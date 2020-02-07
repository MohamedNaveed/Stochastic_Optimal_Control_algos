from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
import pylab
from random import uniform
import time

params = {'axes.labelsize':12,
            'font.size':10,
            'legend.fontsize':12,
            'xtick.labelsize':10,
            'ytick.labelsize':10,
            'text.usetex':True,
            'figure.figsize':[4.5,4.5]}
pylab.rcParams.update(params)

filename = "/home/naveed/Dropbox/Research/Data/T_LQR_files/TLQR_replan_1_agent_wo_obs_x/TLQR_replan_fast_1_agent_wo_obs_epsi8_all_dummy.csv"
time_filename = "/home/naveed/Documents/Optimal_trajectory/T_LQR_files/TLQR_replan_1_agent_wo_obs_x/TLQR_replan_3_agent_wo_obs_02_time1.csv"
file = open(filename,"a")
#time_file = open(time_filename,"w")
file.write('epsilon' + ',' + 'Average Cost' + ',' + 'Cost variance' + ','
                + 'Average Time' + ',' + '# Replan mean' + ',' + 'Replan variance' + ',' + 'Replan bound' + ',' + 'Hc'+'\n' )

N = 1#no. of agents

dt = .1
L = 0.5
replan_bound_vec = np.array([.02]) #L #percent/100
replan = False
no_iters = 1
#alloting start points
x_g = DM.zeros(4,N)


a_linear_lim = 2
a_rot_lim = pi/24

v_max = 4
w_max = pi/12


x_g[0,0] = [3.5]
x_g[1,0] = [7]
x_g[2,0] = math.pi/2

if N > 1:
    x_g[0,1]= [2]
    x_g[1,1]= [8]

if N > 2:
    x_g[0,2]= [8]
    x_g[1,2]= [1.5]

#obstacle location and sizes
obs_1 = [7,3]; r1 = 1.0
obs_2 = [1.5,6]; r2 = 1.0
obs_3 = [6,6]; r3 = 1.0

X_max = [20,20]

OBSTACLES = False
SAVE_FIG = False

def cost_func(U, K, Hc, x_i):

    cost = 0

    U = reshape(U,2*N,Hc)

    R = DM([[20,0],[0,200]])
    Q = DM([[20,0,0,0],[0,20,0,0],[0,0,0,0],[0,0,0,0]])
    Qf = 1000*DM([[7,0,0,0],[0,7,0,0],[0,0,10,0],[0,0,0,1]])#terminal cost
    X = MX(4*N,K)


    for i in range(K):

        if i==0:

            for j in range(N):
                X[4*j:4*(j+1),i] = car_robot_dynamics(x_i[:,j],U[2*j:2*(j+1),i]) #calculate new state
        elif i < Hc:

            for j in range(N):
                X[4*j:4*(j+1),i] = car_robot_dynamics(X[4*j:4*(j+1),i-1],U[2*j:2*(j+1),i]) #calculate new state

        else:

            for j in range(N):
                X[4*j:4*(j+1),i] = car_robot_dynamics(X[4*j:4*(j+1),i-1],U[2*j:2*(j+1),Hc - 1]) #calculate new state

        obstacle_cost = 0

        if OBSTACLES:
            for j in range(N):

                x_t = X[4*j:4*(j+1),i]
                obstacle_cost = obstacle_cost + 1000*(exp(-((x_t[0]-obs_1[0])**2 + (x_t[1]-obs_1[1])**2 - (r1)**2)) +
                                exp(-((x_t[0]-obs_2[0])**2 + (x_t[1]-obs_2[1])**2 - (r2)**2)) +
                                exp(-((x_t[0]-obs_3[0])**2 + (x_t[1]-obs_3[1])**2 - (r3)**2))) #+
                                #exp(-((x_t[0]-obs_4[0])**2 + (x_t[1]-obs_4[1])**2 - (r4)**2)))+
                                # exp(-((x_t[0]-obs_5[0])**2 + (x_t[1]-obs_5[1])**2 - (r5)**2))+
                                # exp(-((x_t[0]-obs_6[0])**2 + (x_t[1]-obs_6[1])**2 - (r6)**2)))


        inter_agent_cost = 0

        #inter agent collision cost
        if N > 1:
            for j in range(N-1):

                for h in range(N-j-1):

                    agent_1 = X[4*j:4*(j+1),i]
                    agent_2 = X[4*(j+(h+1)):4*(j+(h+1)+1),i]

                    inter_agent_cost = inter_agent_cost + (1000*exp(-((agent_1[0] - agent_2[0])**2 +
                                        (agent_1[1] - agent_2[1])**2 - (2*L)**2)))

        if i < Hc:
            for j in range(N):
                cost = cost + (mtimes(mtimes(U[2*j:2*(j+1),i].T,R),U[2*j:2*(j+1),i]) +
                    mtimes(mtimes((x_g[:,j] - X[4*j:4*(j+1),i]).T,Q),(x_g[:,j] - X[4*j:4*(j+1),i]))
                    + obstacle_cost + inter_agent_cost)

        else:

            for j in range(N):
                cost = cost + (mtimes(mtimes(U[2*j:2*(j+1),Hc-1].T,R),U[2*j:2*(j+1),Hc-1]) +
                    mtimes(mtimes((x_g[:,j] - X[4*j:4*(j+1),i]).T,Q),(x_g[:,j] - X[4*j:4*(j+1),i]))
                    + obstacle_cost + inter_agent_cost)


    for j in range(N): #terminal cost
        cost = cost +  (mtimes(mtimes((x_g[:,j] - X[4*j:4*(j+1),i]).T,Qf),(x_g[:,j] - X[4*j:4*(j+1),i])))

    return cost

def car_robot_dynamics(X_prev,U):

    #X_new = MX(4,1)
    X_new = X_prev + dt*blockcat([[mtimes(U[0],cos(X_prev[2]))],
            [mtimes(U[0],sin(X_prev[2]))],
            [mtimes(U[0],tan(X_prev[3])*1/L)],[U[1]]])
    #print "X_new:", X_new
    #print "X_newT:", reshape(X_new,1,4)
    return X_new

def dynamics_uncertain(X_prev,U):

    w1 = epsilon*v_max*np.random.normal(0,1,1)
    w2 = epsilon*w_max*np.random.normal(0,1,1)

    X_new = X_prev + dt*blockcat([[mtimes(U[0]+w1,cos(X_prev[2]))],
            [mtimes(U[0]+w1,sin(X_prev[2]))],
            [mtimes(U[0]+w1,tan(X_prev[3])*1/L)],[U[1]+w2]])
    #print "X_new:", X_new

    return X_new

def constraints(U):

    c = MX(N*N,K) #constraint equation
    U = reshape(U,2*N,K)
    X = MX(4*N,K)

    for i in range(K):

        if i==0:

            m=0; n=0 #variables to specify each agent. m=0:4 corresponds to 1 agent
            for j in range(N):
                X[m:m+4,i] = car_robot_dynamics(x_i[:,j],U[n:n+2,i]) #calculate new state
                m = m + 4; n = n + 2 #robot update
        else:

            m = 0; n = 0;
            for j in range(N):
                X[m:m+4,i] = car_robot_dynamics(X[m:m+4,i-1],U[n:n+2,i]) #calculate new state
                m = m + 4; n = n + 2 #robot update

        #collision constraints
        m1 = 0
        for j in range(N):
            m2 = 0

            for h in range(N):

                if m1!=m2:    #to exclude finding for same robot
                    c[j*N+h,i] = (mtimes((X[m1:m1+4,i] - X[m2:m2+4,i]).T,(X[m1:m1+4,i] - X[m2:m2+4,i])) -0.01 ) #robots should not come within a distance sqrt(.01)
                m2 = m2 +4;
            m1 = m1+4;

    c = reshape(c,1,N*N*K)

    return c

def lower_bound(K):
    ones = DM.ones(2,N*K)
    lb = blockcat([[-4*ones[0,:]],[(-w_max)*ones[1,:]]])
    lb = reshape(lb,2*N*K,1)
    #print "lb", lb
    return lb

def upper_bound(K):
    ones = DM.ones(2,N*K)
    ub = blockcat([[4*ones[0,:]],[(w_max)*ones[1,:]]])
    ub = reshape(ub,2*N*K,1)

    return ub

def calculate_P(P,j,X,K,U):


    if j == K: #K
        Q_f = 100*DM([[150,0,0,0],[0,150,0,0],[0,0,30,0],[0,0,0,30]])
        for i in range(N):
            P[i*16:(i+1)*16,j] = reshape(Q_f,16,1)
        #print "j=",j, P
        P = calculate_P(P,j-1,X,K,U)
        return P

    else:
        Q = DM([[10,0,0,0],[0,10,0,0],[0,0,100,0],[0,0,0,100]])
        R = 100*DM([[10,0],[0,100]])
        #A = DM.eye(4)

        for n in range(N):

            A02 =-U[2*n,j]*math.sin(X[4*n+2,j])*dt

            A12 = U[2*n,j]*math.cos(X[4*n+2,j])*dt

            A23 = U[2*n,j]*dt/(((math.cos(X[4*n + 3,j]))**2)*L)

            A = DM.eye(4)
            A[0,2] = A02
            A[1,2] = A12
            A[2,3] = A23

            B = DM([[math.cos(X[4*n + 2,j]),0],[math.sin(X[4*n + 2,j]),0],[math.tan(X[4*n + 3,j])/L,0],[0,1]])*dt

            P[n*16:(n+1)*16,j] = reshape(Q + mtimes(mtimes(A.T,reshape(P[n*16:(n+1)*16,j+1],4,4)),A) - mtimes(mtimes(mtimes(A.T,reshape(P[n*16:(n+1)*16,j+1],4,4)),
                    mtimes(B,inv(R + mtimes(mtimes(B.T,reshape(P[n*16:(n+1)*16,j+1],4,4)),B)))),mtimes(mtimes(B.T,reshape(P[n*16:(n+1)*16,j+1],4,4)),A)),16,1)
        #print "j=",j, P[:,j]

        if(j!=0):
            P = calculate_P(P,j-1,X,K,U)

        return P

def calculate_K(P,X,K,U):

    K_lqr = DM(8*N,K)
    R = 100*DM([[10,0],[0,100]])
    #A = DM.eye(4)
    i=0
    for i in range(K):
        for n in range(N):
            A02 =-U[2*n,i]*math.sin(X[4*n+2,i])*dt

            A12 = U[2*n,i]*math.cos(X[4*n+2,i])*dt

            A23 = U[2*n,i]*dt/(((math.cos(X[4*n+3,i]))**2)*L)

            A = DM.eye(4)
            A[0,2] = A02
            A[1,2] = A12
            A[2,3] = A23

            B = DM([[math.cos(X[4*n+2,i]),0],[math.sin(X[4*n+2,i]),0],[math.tan(X[4*n+3,i])/L,0],[0,1]])*dt

            K_lqr[n*8:(n+1)*8,i] = reshape(mtimes(inv(R+mtimes(mtimes(B.T,reshape(P[n*16:(n+1)*16,i+1],4,4)),B)),
                                    mtimes(mtimes(B.T,reshape(P[n*16:(n+1)*16,i+1],4,4)),A)),8,1)
        #K_lqr[:,i] = reshape(-mtimes(mtimes(inv(R),B.T),reshape(P[:,i+1],4,4)),8,1)
        #print "i=",i,K_lqr[:,i]
    return K_lqr

def limit(U,U_prev):

    if abs(U[0]-U_prev[0]) > a_linear_lim:
        if U[0]>U_prev[0] :
            U[0] = U_prev[0] + a_linear_lim
        else:
            U[0] = U_prev[0] - a_linear_lim

    if abs(U[1]-U_prev[1]) > a_rot_lim:
        if U[1]>U_prev[1] :
            U[1] = U_prev[1] + a_rot_lim
        else:
            U[1] = U_prev[1] - a_rot_lim

    if U[0]>v_max:
        U[0] = v_max
    elif U[0]<-v_max:
        U[0] = -v_max
    if U[1]>w_max:
        U[1] = w_max
    elif U[1]<-w_max:
        U[1] = -w_max

    return U


def boundary_env(U,K,x_i):

    c = MX(2*N,K) #constraint equation
    U = reshape(U,2*N,K)
    X = MX(4*N,K)
    for i in range(K):
        #print "i:",i
        if i==0:
            #print "at if"
            m=0; n=0 #variables to specify each agent. m=0:4 corresponds to 1 agent
            for j in range(N):
                X[m:m+4,i] = car_robot_dynamics(x_i[:,j],U[n:n+2,i]) #calculate new state
                c[2*j:2*(j+1),i] = X_max - X[m:m+2,i]
        else:
            #print "at else",i
            #print X
            m = 0; n = 0;
            for j in range(N):
                X[m:m+4,i] = car_robot_dynamics(X[m:m+4,i-1],U[n:n+2,i]) #calculate new state
                c[2*j:2*(j+1),i] = X_max - X[m:m+2,i]

        m = m + 4; n = n + 2 #robot update
    c = reshape(c,1,2*N*K)

    return c

def acceleration_limit(U, U_i, K):

    c = MX(2*N,K) #constraint equation
    U = reshape(U,2*N,K)

    i=0;j=0;

    for j in range(N):
        for i in range(K):

            if i == 0:

                c[2*j,i] = mtimes((U[2*j,i]-U_i[2*j]).T,(U[2*j,i]-U_i[2*j])) - a_linear_lim**2
                c[2*j+1,i] = mtimes((U[2*j+1,i]-U_i[2*j+1]).T,(U[2*j+1,i]-U_i[2*j+1])) - a_rot_lim**2
                U_prev = U[2*j:2*(j+1),i]

            else:
                c[2*j,i] = mtimes((U[2*j,i]-U_prev[0]).T,(U[2*j,i]-U_prev[0])) - a_linear_lim**2
                c[2*j+1,i] = mtimes((U[2*j+1,i]-U_prev[1]).T,(U[2*j+1,i]-U_prev[1])) - a_rot_lim**2
                U_prev = U[2*j:2*(j+1),i]

    c = reshape(c,1,2*N*K)
    return c

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def plan(K, Hc, x_i, U_i, U_guess):

    opti = casadi.Opti()

    U = opti.variable(2*N*Hc,1) #control variable

    opti.set_initial(U, U_guess)
    opti.minimize(cost_func(U, K, Hc, x_i))

    # if N!=1:
    #     opti.subject_to(constraints(U) >= 0) #constraint to avoid collision with other robots
    #opti.subject_to(boundary_env(U,K,x_i) > 0) #constraint for wall
    opti.subject_to(U <= upper_bound(Hc))
    opti.subject_to(U >= lower_bound(Hc))
    opti.subject_to(acceleration_limit(U, U_i, Hc)<=0)

    opti.solver('ipopt')
    sol = opti.solve()
    U_opti = sol.value(U)

    U_opti = reshape(U_opti,2*N,Hc)

    X = np.zeros((4*N,Hc))

    for i in range(Hc):#nominal trajectory

        if i==0:

            for j in range(N):
                if j == 0:
                    x_t = car_robot_dynamics(x_i[:,j],U_opti[2*j:2*(j+1),i]) #calculate new state
                else:
                    x_t = np.append(x_t, car_robot_dynamics(x_i[:,j],U_opti[2*j:2*(j+1),i]), axis=0)

            X = x_t
        else:

            for j in range(N):
                if j == 0:
                    x_t = car_robot_dynamics(X[4*j:4*(j+1),i-1],U_opti[2*j:2*(j+1),i]) #calculate new state
                else:
                    x_t = np.append(x_t, car_robot_dynamics(X[4*j:4*(j+1),i-1],U_opti[2*j:2*(j+1),i]), axis=0)

            X = np.append(X, x_t, axis=1)


    #check collisions
    # for i in range(Hc):
    #     if N > 1:
    #         for j in range(N-1):
    #
    #             for h in range(N-j-1):
    #
    #                 agent_1 = X[4*j:4*(j+1), i]
    #                 agent_2 = X[4*(j+(h+1)):4*(j+(h+1)+1), i]
    #                 enablePrint()
    #                 if ((agent_1[0] - agent_2[0])**2 + (agent_1[1] - agent_2[1])**2 - (2*L)**2) < 0:
    #                     print "Collision detected", i
    #                 blockPrint()
    #                 #print " dist:", (agent_1[0] - agent_2[0])**2 + (agent_1[1] - agent_2[1])**2

    P = DM(16*N,Hc+1)
    K_lqr = DM(8*N,Hc)
    j = Hc

    P = calculate_P(P,j,X,Hc,U_opti)
    K_lqr = calculate_K(P,X,Hc,U_opti)

    return U_opti, X, K_lqr

def inter_agent_cost_func(X):

    inter_agent_cost = 0
    if N > 1:
        for j in range(N-1):

            for h in range(N-j-1):

                agent_1 = X[4*j:4*(j+1)]
                agent_2 = X[4*(j+(h+1)):4*(j+(h+1)+1)]

                inter_agent_cost = inter_agent_cost + (1000*exp(-((agent_1[0] - agent_2[0])**2 +
                                    (agent_1[1] - agent_2[1])**2 - (2*L)**2)))

    return inter_agent_cost

if __name__=='__main__':

    global epsilon, replan
    K = 35 #time horizon


    #initial positions.
    x_i = DM.zeros(4,N)

    x_i[0,0] = [3]; x_i[1,0] = [1];
    if N > 1 :
        x_i[0,1]= [5]; x_i[1,1] = [1];
    if N > 2 :
        x_i[0,2]= [6]; x_i[1,2] = [8];

    #print "Iteration:",iter

    R = DM([[20,0],[0,200]])
    Q = DM([[20,0,0,0],[0,20,0,0],[0,0,0,0],[0,0,0,0]])
    Qf = 1000*DM([[7,0,0,0],[0,7,0,0],[0,0,10,0],[0,0,0,1]])#terminal cost

    Hc_array = np.linspace(7,7,1,dtype=np.int16) #control horizon
    epsi_range = np.linspace(0.2,0.2,1)
    #epsi_range = np.append(epsi_range, np.linspace(.2,1,5), axis=0)
    epsi = 20

    print "Epsilon range:", epsi_range

    pylab.figure(1)

    pylab.xlim(-1, 6)
    pylab.ylim(0.5, 8)

    blockPrint()
    for Hc_ini in Hc_array:

        start_opti = time.time()
        U_opti_ini, X_ini, K_lqr_ini = plan(Hc_ini, Hc_ini, x_i, DM(np.zeros((2*N,1))), DM(np.zeros((2*N*Hc_ini,1))))
        end_opti = time.time()
        time_array = end_opti - start_opti

        #Nominal cost

        #pylab.plot(X[0,:],X[1,:],'r',linewidth=2,label="Nominal")
        Cost = 0
        for i in range(Hc_ini):
            for j in range(N):

                U = U_opti_ini[2*j:2*(j+1),i]
                x_t = X_ini[4*j:4*(j+1), i]
                Cost = Cost + (mtimes(mtimes(U.T,R),U) +
                        mtimes(mtimes((x_g[:,j] - x_t).T,Q),(x_g[:,j] - x_t)))

            inter_agent_cost = inter_agent_cost_func(X_ini[:,i])
            Cost = Cost + inter_agent_cost

            if i == 0:
                Cost_nominal_ini = Cost

            else:
                Cost_nominal_ini = np.append(Cost_nominal_ini, Cost)

        #Terminal cost
        for j in range(N):
            x_t = X_ini[4*j:4*(j+1), Hc_ini-1]
            Cost = Cost + mtimes(mtimes((x_g[:,j] - x_t).T,Qf),(x_g[:,j] - x_t))

        Cost_nominal_ini = np.append(Cost_nominal_ini, Cost)
        #enablePrint()
        print "Cost Nominal:", Cost_nominal_ini
        blockPrint()

        nav = 0
        for replan_bound in replan_bound_vec:
            for epsilon in epsi_range:
                np.random.seed(4)
                start = time.time()

                if nav != 0:
                    epsi = epsi + 40#+ int(epsi_range[1]*100 - epsi_range[0]*100)
                # else:
                #     epsi = epsi + 1
                Cost_iter = np.zeros(no_iters) #store cost for each iteration
                Time_iter = np.zeros(no_iters) #store time for each iteration
                Replan_iter = np.zeros(no_iters) #store no. of replans for each iteration

                for iter in range(no_iters): #no of iterations

                    U_opti = DM(2*N,K)
                    K_lqr = DM(8*N,K)
                    X = np.zeros((4*N,K))
                    Cost_nominal = np.zeros(K+1)
                    U_opti[:,0:Hc_ini] = U_opti_ini #Storing in new variable since replan can the new variable instead of altering the old
                    K_lqr[:,0:Hc_ini] = K_lqr_ini
                    X[:,0:Hc_ini] = X_ini
                    Cost_nominal[0:Hc_ini+1] = Cost_nominal_ini
                    Hc = Hc_ini

                    Cost = 0
                    obstacle_cost = 0
                    replan_count = 0

                    replan_index = []

                    for i in range(K):#actual trajectory with LQR
                        start_time = time.time()
                        if i==0:

                            for j in range(N):
                                if j == 0:

                                    x_t = dynamics_uncertain(x_i[:,j],U_opti[2*j:2*(j+1),i])
                                    x_cur = x_t
                                    U_prev = U_opti[2*j:2*(j+1),i]

                                else:

                                    x_t = dynamics_uncertain(x_i[:,j],U_opti[2*j:2*(j+1),i])
                                    x_cur = np.append(x_cur, x_t, axis = 0)

                                    U_prev = np.append(U_prev, U_opti[2*j:2*(j+1),i], axis = 0)


                                if OBSTACLES :
                                    obstacle_cost = 1000*(exp(-((x_t[0]-obs_1[0])**2 + (x_t[1]-obs_1[1])**2 - (r1)**2)) +
                                                    exp(-((x_t[0]-obs_2[0])**2 + (x_t[1]-obs_2[1])**2 - (r2)**2)) +
                                                    exp(-((x_t[0]-obs_3[0])**2 + (x_t[1]-obs_3[1])**2 - (r3)**2)))


                                U_lqr = U_opti[2*j:2*(j+1),i]
                                Cost = Cost + (mtimes(mtimes(U_lqr.T,R),U_lqr) +
                                        mtimes(mtimes((x_g[:,j] - x_t).T,Q),(x_g[:,j] - x_t)) + obstacle_cost)

                            #if ((x_t[0] - X[4*j,i])**2 + (x_t[1] - X[4*j+1,i])**2) > replan_bound**2 :
                            if abs(Cost - Cost_nominal[i])/Cost_nominal[i] > replan_bound:
                                replan = True

                            X_a = x_cur


                        else:



                            if replan:

                                if K - i < Hc:
                                    Hc = K - i

                                #enablePrint()
                                print i
                                blockPrint()
                                replan_count += 1
                                Cost_b4replan = Cost

                                replan_index.append(i-1)

                                xi_new = np.reshape(X_a[:,i-1], (N,4))

                                U_guess = reshape(U_opti[:,i:i+Hc],2*N*Hc,1)

                                U_opti[:,i:i+Hc], X[:,i:i+Hc], K_lqr[:,i:i+Hc] = plan(Hc, Hc, xi_new.T, U_prev, U_guess)

                                #Nominal cost
                                Cost_temp = Cost_nominal[i-1] #change
                                for m in range(i,i+Hc):
                                    for n in range(N):

                                        U = U_opti[2*n:2*(n+1),m]
                                        x_t = X[4*n:4*(n+1), m]
                                        Cost_temp = Cost_temp + (mtimes(mtimes(U.T,R),U) +
                                                mtimes(mtimes((x_g[:,n] - x_t).T,Q),(x_g[:,n] - x_t)))

                                    inter_agent_cost = inter_agent_cost_func(X[:,m])
                                    Cost_temp = Cost_temp + inter_agent_cost

                                    Cost_nominal[m] = Cost_temp


                                #Terminal cost
                                for n in range(N):
                                    x_t = X[4*n:4*(n+1), K-1]
                                    Cost_temp = Cost_temp + mtimes(mtimes((x_g[:,n] - x_t).T,Qf),(x_g[:,n] - x_t))

                                Cost_nominal[i+Hc] = Cost_temp

                                replan = False
                                replan_done = True #flag used to ignore LQR feedback for first step after replan


                            else:
                                replan_done = False

                            for j in range(N):

                                if replan_done:

                                    U_lqr = U_opti[2*j:2*(j+1),i]

                                else:
                                    U_lqr = U_opti[2*j:2*(j+1),i] + mtimes(reshape(K_lqr[j*8:(j+1)*8,i],2,4),(X[4*j:4*(j+1),i-1]-X_a[4*j:4*(j+1),i-1]))

                                    U_lqr = limit(U_lqr, U_prev[2*j:2*(j+1)])

                                if j == 0:

                                    x_t = dynamics_uncertain(X_a[4*j:4*(j+1), i-1],U_lqr)
                                    x_cur = x_t

                                    U_temp = U_lqr #temp variable for assigning U_prev

                                else:

                                    x_t = dynamics_uncertain(X_a[4*j:4*(j+1), i-1],U_lqr)
                                    x_cur = np.append(x_cur, x_t, axis = 0)

                                    U_temp = np.append(U_temp, U_lqr, axis = 0) #stack vertically

                                if OBSTACLES :
                                    obstacle_cost = 1000*(exp(-((x_t[0]-obs_1[0])**2 + (x_t[1]-obs_1[1])**2 - (r1)**2)) +
                                                    exp(-((x_t[0]-obs_2[0])**2 + (x_t[1]-obs_2[1])**2 - (r2)**2)) +
                                                    exp(-((x_t[0]-obs_3[0])**2 + (x_t[1]-obs_3[1])**2 - (r3)**2)))


                                Cost = Cost + (mtimes(mtimes(U_lqr.T,R),U_lqr) +
                                        mtimes(mtimes((x_g[:,j] - x_t).T,Q),(x_g[:,j] - x_t)) + obstacle_cost)


                            if len(replan_index) > 0:

                                # enablePrint()
                                print "i:", i," Cost:", Cost, " Cost_nominal:", Cost_nominal[i], " deviation:", abs((Cost - Cost_b4replan) - (Cost_nominal[i] - Cost_nominal[replan_index[-1]]))/(Cost_nominal[i] - Cost_nominal[replan_index[-1]])
                                print "cost - costb4replan:",Cost - Cost_b4replan, "nominal change:",Cost_nominal[i] - Cost_nominal[replan_index[-1]],"index:",replan_index[-1]
                                blockPrint()

                                if abs((Cost - Cost_b4replan) - (Cost_nominal[i] - Cost_nominal[replan_index[-1]]))/(Cost_nominal[i] - Cost_nominal[replan_index[-1]]) > replan_bound:
                                    replan = True

                                elif i - replan_index[-1] == Hc:
                                    replan = True

                            else:

                                # enablePrint()
                                print "i:", i," Cost:", Cost, " Cost_nominal:", Cost_nominal[i], " deviation:", abs(Cost - Cost_nominal[i])/Cost_nominal[i]
                                blockPrint()

                                if abs(Cost - Cost_nominal[i])/Cost_nominal[i] > replan_bound:
                                    replan = True

                                elif i == Hc-1:
                                    replan = True


                            X_a = np.append(X_a, x_cur, axis = 1)
                            U_prev = U_temp
                        #inter agent collision cost

                        inter_agent_cost = inter_agent_cost_func(X_a[:,i])

                        Cost = Cost + inter_agent_cost
                        end_time = time.time()
                        time_array = np.append(time_array, end_time-start_time)
                    for j in range(N):
                        Cost = Cost + mtimes(mtimes((x_g[:,j] - X_a[4*j:4*(j+1),K-1]).T,Qf),(x_g[:,j] - X_a[4*j:4*(j+1),K-1]))

                    Cost_iter[iter] = Cost
                    Replan_iter[iter] = replan_count

                    # for j in range(N):
                    #     pylab.plot(X[4*j,:],X[4*j+1,:],'r',linewidth=2,label="Nominal")


                    for j in range(N):
                        pylab.plot(X_a[4*j,:],X_a[4*j+1,:],'--',linewidth=2,label="T-LQR2 " + "$\epsilon =$"+ str(epsi/100.0))
                    nav = 1
                    """
                    pylab.plot(x_i[0,1:N], x_i[1,1:N], 'co')
                    pylab.plot(x_g[0,1:N], x_g[1,1:N], 'g*')
                    #replan markers

                    if len(replan_index) > 0:
                        for i in replan_index:
                            for j in range(N):
                                pylab.plot(X_a[4*j,i], X_a[4*j+1,i], 'mX', markersize=7)

                    if OBSTACLES:
                        circle1 = pylab.Circle(obs_1,r1,color="r")
                        circle2 = pylab.Circle(obs_2,r2,color="r")
                        circle3 = pylab.Circle(obs_3,r3,color="r")

                        ax = pylab.gca()

                        ax.add_artist(circle1)
                        ax.add_artist(circle2)
                        ax.add_artist(circle3)

                    pylab.xlabel('X (meters)')
                    pylab.ylabel('Y (meters)')



                    if SAVE_FIG:
                        #pylab.savefig('/home/naveed/Documents/'+'run'+str(epsi)+'_'+str(iter))
                        #pylab.close()
                        pass
                    else:
                        #pylab.show()
                        pass
                    """
                end = time.time()

                Time_total = end - start
                Avg_time = (Time_total/no_iters) + end_opti - start_opti

                enablePrint()
                print "Hc:",Hc_ini," replan_bound",replan_bound," epsilon:",epsilon , "   Average Cost:",np.mean(Cost_iter), " Cost Variance:", np.var(Cost_iter), " Average replans:", np.mean(Replan_iter), " Avg time:", Avg_time, " Replan Var:", np.var(Replan_iter)
                blockPrint()
                file.write(str(epsilon) + ',' + str(np.mean(Cost_iter)) + ',' + str(np.var(Cost_iter))
                                + ',' + str(Avg_time) + ',' + str(np.mean(Replan_iter)) + ',' + str(np.var(Replan_iter)) + ',' + str(replan_bound) + ',' + str(Hc_ini) + '\n')

                # for t in range(K+1):
                #     time_file.write(str(time_array[t]) + ',')

    """
    pylab.plot(x_i[0,0], x_i[1,0], 'co',label='Start',markersize=10)
    pylab.plot(x_g[0,0], x_g[1,0], 'g*',label='Goal',markersize=10)


    if len(replan_index) > 0:
        for j in range(N):
            pylab.plot(X_a[4*j,replan_index[0]], X_a[4*j+1,replan_index[0]], 'mX', markersize=7,label='Replan point')

    legend = pylab.legend()

    frame = legend.get_frame()
        #frame.set_facecolor('0.9')
        #frame.set_edgecolor('0.9')

        #pylab.show()

    pylab.savefig('/home/naveed/Dropbox/Research/Manuscripts/Media/Single_agent_plots/'+'1_agent_replan_test_cases_high3')
    #pylab.close()
    """
    file.close()
    #time_file.close()
