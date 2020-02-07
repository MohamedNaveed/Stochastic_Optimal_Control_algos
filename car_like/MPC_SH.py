"""
Model predictive control (short horizon) implementation for trajectory control of a system car-like robots

Author: Mohamed Naveed G.
20th July 2019.
"""


from casadi import *
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
import numpy as np
import pylab
import sys
import time

params = {'axes.labelsize':12,
            'font.size':10,
            'legend.fontsize':12,
            'xtick.labelsize':10,
            'ytick.labelsize':10,
            'text.usetex':True,
            'figure.figsize':[4.5,4.5]}
pylab.rcParams.update(params)


filename = "/home/naveed/Dropbox/Research/Data/MPC_files/MPC_1_agent_wo_obs_hx/MPC_fast_1_agent_wo_obs_epsi1_all.csv"
file = open(filename,"a")
file.write('epsilon' + ',' + 'Average Cost' + ',' + 'Cost variance' + ',' + 'Average Time' + ',' + 'Hc' + '\n' )

N = 1 #no. of agents
#K = 30 #time horizon
dt = .1
L = 0.5

no_iters = 20

a_linear_lim = 2
a_rot_lim = pi/24

v_max = 4
w_max = pi/12

epsilon = 0.10 #noise percent

x_g = DM.zeros(4,N)
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

#boundary
X_max = [20,20]

OBSTACLES = False
SAVE_FIG = False

def cost_func(U,K,Hc,x_i):
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


    X_new = X_prev + dt*blockcat([[mtimes(U[0],cos(X_prev[2]))],
            [mtimes(U[0],sin(X_prev[2]))],
            [mtimes(U[0],tan(X_prev[3])*1/L)],[U[1]]])

    return X_new

def dynamics_uncertain(X_prev,U):

    w1 = epsilon*v_max*np.random.normal(0,1,1)
    w2 = epsilon*w_max*np.random.normal(0,1,1)

    X_new = X_prev + dt*blockcat([[mtimes(U[0]+w1,cos(X_prev[2]))],
            [mtimes(U[0]+w1,sin(X_prev[2]))],
            [mtimes(U[0]+w1,tan(X_prev[3])*1/L)],[U[1]+w2]])
    #print "X_new:", X_new

    return X_new

def constraints(U,K,x_i):

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
    lb = blockcat([[-v_max*ones[0,:]],[(-w_max)*ones[1,:]]])
    lb = reshape(lb,2*N*K,1)
    #print "lb", lb
    return lb

def upper_bound(K):
    ones = DM.ones(2,N*K)
    ub = blockcat([[v_max*ones[0,:]],[(w_max)*ones[1,:]]])
    ub = reshape(ub,2*N*K,1)

    return ub

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

        if i==0:

            m=0; n=0 #variables to specify each agent. m=0:4 corresponds to 1 agent
            for j in range(N):
                X[m:m+4,i] = car_robot_dynamics(x_i[:,j],U[n:n+2,i]) #calculate new state
                c[2*j:2*(j+1),i] = X_max - X[m:m+2,i]
        else:

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

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def plan(K, Hc, x_i, U_i, U_guess):

    opti = casadi.Opti()

    U = opti.variable(2*N*Hc,1) #control variable

    opti.set_initial(U, U_guess)
    opti.minimize(cost_func(U,K,Hc,x_i))

    # if N!=1:
    #     opti.subject_to(constraints(U,K,x_i) >= 0) #constraint to avoid collision with other robots
    #opti.subject_to(boundary_env(U,K,x_i) > 0) #constraint for wall
    opti.subject_to(U <= upper_bound(Hc))
    opti.subject_to(U >= lower_bound(Hc))
    opti.subject_to(acceleration_limit(U, U_i, Hc) <= 0)

    opti.solver('ipopt')
    sol = opti.solve()
    U_opti = sol.value(U)

    U_opti = reshape(U_opti,2*N,Hc)

    return U_opti

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

    global epsilon

    K = 35 #horizon

    #Hp = 25#prediction horizon

    epsi_range = np.linspace(.1,0.1,1)
    #epsi_range = np.append(epsi_range, np.linspace(.2,1,5), axis=0)
    epsi = 20 #var used just to name the resultant figures. (decimal names are not accepted)
    nav = 0
    print "Epsilon range:", epsi_range
    #allocating start points
    x_i = DM.zeros(4,N)


    x_i[0,0] = [3]; x_i[1,0] = [1];

    if N > 1:
        x_i[0,1]= [5]; x_i[1,1]= [1];

    if N > 2:
        x_i[0,2]= [6]; x_i[1,2]= [8];


    R = DM([[20,0],[0,200]])
    Q = DM([[20,0,0,0],[0,20,0,0],[0,0,0,0],[0,0,0,0]])
    Qf = 1000*DM([[7,0,0,0],[0,7,0,0],[0,0,10,0],[0,0,0,1]])#terminal cost

    blockPrint()

    Hc_array = np.linspace(5,35,7,dtype=np.int16) #control horizon
    for Hc_ini in Hc_array:
        start_opti = time.time()
        U_opti_ini = plan(Hc_ini, Hc_ini, x_i, DM(np.zeros((2*N,1))), DM(np.zeros((2*N*Hc_ini,1))))
        end_opti = time.time()

        enablePrint()
        #print "U:", U_opti_ini
        blockPrint()

        #pylab.figure(1)

        #pylab.xlim(-1, 6)
        #pylab.ylim(0.5, 8)


        for epsilon in epsi_range:
            np.random.seed(4)
            if nav!=0:
                epsi = epsi + 40# int(epsi_range[1]*100 - epsi_range[0]*100)
            # else:
            #     epsi = epsi_range[0]*100

            Cost_iter = np.zeros(no_iters) #store cost for each iteration
            Time_iter = np.zeros(no_iters) #store time for each iteration

            iter = 0

            while iter < no_iters: #no of iterations
                #print "Iteration:",iter

                Hc = Hc_ini #control horizon
                #Hp = 25 #prediction horizon

                try:
                    start = time.time()
                    Cost = 0


                    U_opti = U_opti_ini
                    #MPC
                    for t in range(K): #t - time step

                        if t == 0:

                            U = U_opti[:,0]

                            j = 0
                            for j in range(N):

                                X_next = dynamics_uncertain(x_i[:,j],U_opti[2*j:2*(j+1),0])
                                x_t = X_next
                                U_mpc = U_opti[2*j:2*(j+1),0]

                                obstacle_cost = 0

                                if OBSTACLES:
                                    obstacle_cost = 1000*(exp(-((x_t[0]-obs_1[0])**2 + (x_t[1]-obs_1[1])**2 - (r1)**2)) +
                                                    exp(-((x_t[0]-obs_2[0])**2 + (x_t[1]-obs_2[1])**2 - (r2)**2)) +
                                                    exp(-((x_t[0]-obs_3[0])**2 + (x_t[1]-obs_3[1])**2 - (r3)**2)))

                                Cost = Cost + (mtimes(mtimes(U_mpc.T,R),U_mpc) +
                                        mtimes(mtimes((x_g[:,j] - x_t).T,Q),(x_g[:,j] - x_t)) + obstacle_cost)

                                if j == 0:

                                    X_cur = X_next
                                    X_temp = X_next

                                else:

                                    X_cur = np.append(X_cur,X_next,axis=1)  #stack state horizontally
                                    X_temp = np.append(X_temp,X_next,axis=0) #stack state vertically

                            X = X_temp
                            U_prev = U_opti[:,0]

                        else:

                            # if K - t < Hp:
                            #     Hp = K - t

                            if K - t < Hc:
                                Hc = K - t

                            else:
                                U_opti = np.append(U_opti, DM(np.zeros((2*N,1))), axis=1) #last input is not known, so adding 0 for guess

                            U_guess = reshape(U_opti[:,1:],2*N*(Hc),1)

                            U_opti = plan(Hc, Hc, X_cur, U_prev, U_guess)

                            U = np.append(U,U_opti[:,0],axis=1)
                            X_prev = X_cur

                            j = 0

                            for j in range(N):

                                X_next = dynamics_uncertain(X_prev[:,j],U_opti[2*j:2*(j+1),0])

                                x_t = X_next
                                U_mpc = U_opti[2*j:2*(j+1),0]

                                obstacle_cost = 0

                                if OBSTACLES:

                                    obstacle_cost = 1000*(exp(-((x_t[0]-obs_1[0])**2 + (x_t[1]-obs_1[1])**2 - (r1)**2)) +
                                                    exp(-((x_t[0]-obs_2[0])**2 + (x_t[1]-obs_2[1])**2 - (r2)**2)) +
                                                    exp(-((x_t[0]-obs_3[0])**2 + (x_t[1]-obs_3[1])**2 - (r3)**2)))

                                #if t != K-1:
                                Cost = Cost + (mtimes(mtimes(U_mpc.T,R),U_mpc) +
                                        mtimes(mtimes((x_g[:,j] - x_t).T,Q),(x_g[:,j] - x_t)) + obstacle_cost)
                                # else:
                                #     Cost = Cost + (mtimes(mtimes(U_mpc.T,R),U_mpc) +
                                #             mtimes(mtimes((x_g[:,j] - x_t).T,Qf),(x_g[:,j] - x_t)) + obstacle_cost)

                                if j == 0:

                                    X_cur = X_next
                                    X_temp = X_next

                                else:
                                    X_cur = np.append(X_cur,X_next,axis=1)  #stack state horizontally
                                    X_temp = np.append(X_temp,X_next,axis=0) #stack state vertically

                            X = np.append(X,X_temp,axis=1)  #stack state horizontally
                            #enablePrint()
                            print "State:", X[:,-1]
                            blockPrint()
                            U_prev = U_opti[:,0]
                        #inter agent collision cost

                        inter_agent_cost = inter_agent_cost_func(X[:,t])
                        Cost = Cost + inter_agent_cost



                    for j in range(N):
                        Cost = Cost + mtimes(mtimes((x_g[:,j] - X[4*j:4*(j+1),K-1]).T,Qf),
                            (x_g[:,j] - X[4*j:4*(j+1),K-1]))

                    enablePrint()
                    print "Iteration:",iter
                    #print "U:", U
                    blockPrint()

                    end = time.time()

                    Time_iter[iter] = end-start + end_opti - start_opti
                    Cost_iter[iter] = Cost



                    j=0
                    for j in range(N):
                        pylab.plot(X[4*j,:],X[4*j+1,:],'--',linewidth=2,label="MPC-SH " + "$\epsilon =$"+ str(epsi/100.0))

                    nav = 1

                    pylab.plot(x_i[0,1:N], x_i[1,1:N], 'co')
                    pylab.plot(x_g[0,1:N], x_g[1,1:N], 'g*')

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
                        #pylab.savefig('/home/naveed/Documents/MPC_fast_h7'+'run'+str(epsi)+'_'+str(iter))
                        #pylab.close()
                        pass

                    else:
                        pass
                        #pylab.show()

                    iter += 1

                except:
                    enablePrint()
                    print "Unexpected error:", sys.exc_info()
                    blockPrint()
            #print "Time taken:", Time_iter[iter], "Cost:", Cost
            enablePrint()
            print "epsilon:",epsilon , " Hc:",Hc_ini, "   Average Cost:",np.mean(Cost_iter), " Cost variance:", np.var(Cost_iter), " Avg time:", np.mean(Time_iter)
            file.write(str(epsilon) + ',' + str(np.mean(Cost_iter)) + ',' + str(np.var(Cost_iter)) + ',' + str(np.mean(Time_iter)) + ',' + str(Hc_ini) + '\n')
            blockPrint()


    pylab.plot(x_i[0,0], x_i[1,0], 'co',label='Start',markersize=10)
    pylab.plot(x_g[0,0], x_g[1,0], 'g*',label='Goal',markersize=10)

    legend = pylab.legend()

    frame = legend.get_frame()
    #frame.set_facecolor('0.9')
    #frame.set_edgecolor('0.9')

    pylab.savefig('/home/naveed/Dropbox/Research/Manuscripts/Media/Single_agent_plots/'+'1_agent_MPCfast_test_cases.pdf', format='pdf',bbox_inches='tight',pad_inches = 0.02)
    pylab.close()
    #pylab.show()

    file.close()
