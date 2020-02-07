# Trajectory optimised Linear Quadratic Regulator with replans Sims
# Motion planning for a robot in a fully observable environment under uncertain process noise.
# Author: Mohamed Naveed Gul Mohamed
# email:mohdnaveed96@gmail.com
# Date: Feb 3rd 2020
import casadi as c
import math as m
# import matplotlib
# matplotlib.use('Agg')

from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse
import numpy as np
from numpy import linalg
import pylab
import sys,os
import time

import Algorithm_classes as algo
from robot_models import youBot_model
from robot_models import car_model
from robot_models import quad
import simulation_params as params

no_iters = 10

PLOT = False #True #
PLOT_ROBOT_BOUNDARY = False #True #
FILE_WRITE = True #False #
HPRC = False
Model = 'quad'#'car_w_trailers' #'car_model' #'youBot_model'
OL_time = 0 #Open loop time taken.

Xg = np.array([2,2,2,0,0,0,0,0,0,0,0,0])

if FILE_WRITE:
    filename = "/home/naveed/Dropbox/Research/Data/WAFR20/Quad/TLQR2_1_wo_obs_modified1.csv"
    file = open(filename,"a")
    file.write('epsilon' + ',' + 'Average Cost' + ',' + 'Cost variance' + ',' + 'Average Time' + ',' + 'Average Replans' + ','
        + 'Replan Variance' + ',' + 'replan_bound' + '\n' )


if Model == 'quad':
    robot = quad(dt=0.1)
    #open-loop optimisation gain
    R = c.DM([[5,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]])
    Q = c.DM.eye(robot.nx)
    Q[0,0] = 10; Q[1,1] = 10;Q[2,2] = 10;
    Qf = 1000*c.DM.eye(robot.nx)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def calc_cost(U, X, T):

    Cost_OL_vec = np.zeros(T+1)
    Cost_OL = 0

    for i in range(T):

        Cost_OL = Cost_OL + c.mtimes(c.mtimes(c.reshape(X[:,i],(1,robot.nx)),Q),c.reshape(X[:,i],(robot.nx,1))) + \
                            c.mtimes(c.mtimes(c.reshape(U[:,i],(1,robot.nu)),R),c.reshape(U[:,i],(robot.nu,1)))

        if params.OBSTACLES:
            Obstacle_cost = control.obstacle_cost_func(X_path_ss[0:2,i])
            Cost_OL = Cost_OL + Obstacle_cost

        Cost_OL_vec[i] = Cost_OL

    Cost_OL = Cost_OL + c.mtimes(c.mtimes(c.reshape(Xg - X[:,T],(1,robot.nx)),Qf),c.reshape(Xg - X[:,T],(robot.nx,1)))
    Cost_OL_vec[T] = Cost_OL

    return Cost_OL_vec

def SSP(T, X0, Q, R, Qf, U_guess):
    #State space planning

    control = algo.SSP(T, X0, Xg, R, Q, Qf, params.Xmin, params.Xmax)

    #U_guess = np.zeros((robot.nu,T))

    U_guess = np.reshape(U_guess.T,(robot.nu*T,1))

    print("Solving state space problem...")
    U_opti_ss = control.solve_SSP(robot,X0, np.zeros(robot.nu), U_guess)
    print("Done.")
    #print("Uss:",U_opti_ss)
    X_path_ss = np.zeros((robot.nx,T+1))
    X_path_ss[:,0] = X0
    Cost_OL_vec = np.zeros(T+1)
    Cost_OL = 0

    for i in range(T):

        X_path_ss[:,i+1] = np.reshape(robot.kinematics(X_path_ss[:,i],U_opti_ss[:,i]),(robot.nx,)) #to make it compatible in dimension reshaping from (3,1) to 3

        Cost_OL = Cost_OL + c.mtimes(c.mtimes(c.reshape(X_path_ss[:,i],(1,robot.nx)),Q),c.reshape(X_path_ss[:,i],(robot.nx,1))) + \
                            c.mtimes(c.mtimes(c.reshape(U_opti_ss[:,i],(1,robot.nu)),R),c.reshape(U_opti_ss[:,i],(robot.nu,1)))

        if params.OBSTACLES:
            Obstacle_cost = control.obstacle_cost_func(X_path_ss[0:2,i])
            Cost_OL = Cost_OL + Obstacle_cost

        Cost_OL_vec[i] = Cost_OL

    Cost_OL = Cost_OL + c.mtimes(c.mtimes(c.reshape(Xg - X_path_ss[:,T],(1,robot.nx)),Qf),c.reshape(Xg - X_path_ss[:,T],(robot.nx,1)))
    Cost_OL_vec[T] = Cost_OL

    print "U nom:", U_opti_ss

    return U_opti_ss, X_path_ss, Cost_OL_vec

def TLQR2_run_varying_epsilon():

    global OL_time
    T = 30
    X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

    #LQR gain
    Wx = c.DM.eye(robot.nx) #10
    Wx[0,0] = 10; Wx[1,1] = 10; Wx[2,2] = 10;
    Wxf = 100*c.DM.eye(robot.nx)  #100
    Wu = c.DM.eye(robot.nu)

    U_guess = np.zeros((robot.nu,T))

    OL_start = time.time()
    U_opti, X_nom, Cost_nom = SSP(T,X0,Q,R,Qf,U_guess)

    control = algo.LQR(Wx,Wu,Wxf,T)

    print("Solving for LQR gain...")
    K_lqr = control.solve_K(robot, X_nom, U_opti)

    OL_end = time.time()

    OL_time = OL_end - OL_start

    replan_bound_vec = np.array([.05])
    print "replan bound range:", replan_bound_vec

    epsi_range = np.linspace(0.09,.12,4)
    print "Epsilon range:", epsi_range

    blockPrint()
    for replan_bound in replan_bound_vec:
        for epsilon in epsi_range:
            TLQR2(control, T, X0, U_opti, X_nom, K_lqr, Cost_nom, replan_bound, epsilon)

def TLQR2(control, T, X0, U_opti_ini, X_nom_ini, K_lqr_ini, Cost_nom_ini, replan_bound, epsilon):


    print("Executing Plan...")
    #Execution of plan

    start_exec = time.time()

    Cost_CL_iter = np.zeros(no_iters)
    Replan_iter = np.zeros(no_iters) #vector to store # of replans in every iteration.

    iter = 0
    #np.random.seed(4)
    # enablePrint()
    while iter < no_iters:


        U_opti = U_opti_ini
        X_nom = X_nom_ini
        K_lqr = K_lqr_ini
        Cost_nom = Cost_nom_ini

        #try:

        X_act_CL = np.zeros((robot.nx,T+1))

        X_act_CL[:,0] = X0 #initial condition

        U_CL = np.zeros((robot.nu,T))
        Cost_CL = 0
        replan_count = 0
        replan = False
        replan_index = []

        #closed loop implementation
        for i in range(T):
            #Applying LQR control

            #print "X nom:", X_nom[:,i], " X act:", X_act_CL[:,i]
            if replan:
                replan_count += 1

                replan_index.append(i)

                U_guess = U_opti[:,i:]
                print "Replanning at step:", i
                U_opti[:,i:T], X_nom[:,i:T+1],_ = SSP(T-i,X_act_CL[:,i],Q,R,Qf, U_guess) #calculating new nominal

                #print "Cost_nom:", calc_cost(U_opti,X_nom,T)
                Cost_nom = calc_cost(U_opti,X_nom,T)
                control.T = T - i #setting new horizon.
                K_lqr[:,i:T+1] = control.solve_K(robot, X_nom[:,i:T+1], U_opti[:,i:T]) #calculating new LQR gains

                replan = False

            U_temp = U_opti[:,i] - c.mtimes(c.reshape(K_lqr[:,i],robot.nu,robot.nx),(X_act_CL[:,i] - X_nom[:,i]))

            U_CL[:,i] = np.reshape(control.U_bounds(robot,U_temp),(robot.nu,))

            X_act_CL[:,i+1] = np.reshape(robot.kinematics(X_act_CL[:,i],U_CL[:,i],epsilon),(robot.nx,))

            Cost_CL = Cost_CL + c.mtimes(c.mtimes(c.reshape(X_act_CL[:,i],(1,robot.nx)),Q),c.reshape(X_act_CL[:,i],(robot.nx,1))) + \
                                c.mtimes(c.mtimes(c.reshape(U_CL[:,i],(1,robot.nu)),R),c.reshape(U_CL[:,i],(robot.nu,1)))

            if params.OBSTACLES:
                Obstacle_cost = control.obstacle_cost_func(X_act_CL[0:2,i])
                Cost_CL = Cost_CL + Obstacle_cost

            #enablePrint()
            print "Cost_CL:", Cost_CL, " Cost_nom:", Cost_nom[i], " deviation:", abs(Cost_CL - Cost_nom[i])/Cost_nom[i]
            blockPrint()

            if abs(Cost_CL - Cost_nom[i])/Cost_nom[i] > replan_bound:
                replan = True

        Cost_CL = Cost_CL + c.mtimes(c.mtimes(c.reshape(Xg - X_act_CL[:,T],(1,robot.nx)),Qf),c.reshape(Xg - X_act_CL[:,T],(robot.nx,1)))
        #enablePrint()
        print "Cost_CL:", Cost_CL, " Cost_nom:", Cost_nom[T], " deviation:", abs(Cost_CL - Cost_nom[T])/Cost_nom[T]
        blockPrint()

        Cost_CL_iter[iter] = Cost_CL
        Replan_iter[iter] = replan_count
        iter = iter + 1

        # except:
        #         print "Unexpected error:", sys.exc_info()

    end_exec = time.time()
    time_taken = OL_time + (end_exec - start_exec)/no_iters

    enablePrint()
    print "epsilon:",epsilon , "   Average Cost:",np.mean(Cost_CL_iter), "  Cost var:", np.var(Cost_CL_iter), " Average replans:", np.mean(Replan_iter), " Replan Var:", np.var(Replan_iter), "replan_bound",replan_bound," Time taken:",time_taken
    blockPrint()

    if FILE_WRITE:
        file.write(str(epsilon) + ',' + str(np.mean(Cost_CL_iter)) + ',' +
            str(np.var(Cost_CL_iter)) + ',' + str(time_taken) + ',' + str(np.mean(Replan_iter)) + ','
            + str(np.var(Replan_iter))+ ',' + str(replan_bound) + '\n')

    if PLOT:
        pylab.figure(1)
        pylab.plot(X_nom[0,:],X_nom[1,:],'y',linewidth=2,label="Nominal")
        pylab.plot(X_act_CL[0,:],X_act_CL[1,:],'--b',linewidth=2,label="TLQR")
        pylab.plot(X0[0],X0[1], 'co',label='Start',markersize=10)
        pylab.plot(Xg[0], Xg[1], 'g*',label='Goal',markersize=10)
        pylab.xlabel('X (meters)')
        pylab.ylabel('Y (meters)')

        pylab.figure(2)
        pylab.plot(X_nom[0,:],X_nom[2,:],'y',linewidth=2,label="Nominal")
        pylab.plot(X_act_CL[0,:],X_act_CL[2,:],'--b',linewidth=2,label="TLQR")
        pylab.plot(X0[0],X0[2], 'co',label='Start',markersize=10)
        pylab.plot(Xg[0], Xg[2], 'g*',label='Goal',markersize=10)
        pylab.xlabel('X (meters)')
        pylab.ylabel('Z (meters)')

        if PLOT_ROBOT_BOUNDARY:
            #plotting robot boundary
            diag = m.sqrt(robot.length**2 + robot.breadth**2)/2
            alp = m.atan2(robot.breadth,robot.length)
            ax = pylab.gca()

            for i in np.arange(0,T,10):
                ax.add_patch(Rectangle((X_act_CL[0,i] - diag*m.cos(X_act_CL[2,i] + alp),
                    X_act_CL[1,i] - diag*m.sin(X_act_CL[2,i] + alp)), robot.length, robot.breadth,angle=m.degrees(X_act_CL[2,i]),fill=None,linewidth=.1,color='b'))

            ax.add_patch(Rectangle((X_act_CL[0,T] - diag*m.cos(X_act_CL[2,T] + alp),
                X_act_CL[1,T] - diag*m.sin(X_act_CL[2,T] + alp)), robot.length, robot.breadth,angle=m.degrees(X_act_CL[2,T]),fill=None,linewidth=.1,color='b'))



        legend = pylab.legend(loc=0)
        frame = legend.get_frame()

        pylab.show()


def TLQR2_hprc(epsilon,replan_bound):

    T = 30
    X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

    #LQR gain
    Wx = c.DM.eye(robot.nx) #10
    Wx[0,0] = 10; Wx[1,1] = 10; Wx[2,2] = 10;
    Wxf = 100*c.DM.eye(robot.nx)  #100
    Wu = c.DM.eye(robot.nu)


    U_guess = np.zeros((robot.nu,T))

    OL_start = time.time()
    U_opti, X_nom, Cost_nom = SSP(T,X0,Q,R,Qf,U_guess)

    control = algo.LQR(Wx,Wu,Wxf,T)

    print("Solving for LQR gain...")
    K_lqr = control.solve_K(robot, X_nom, U_opti)

    OL_end = time.time()

    OL_time = OL_end - OL_start

    print("Executing Plan...")
    #Execution of plan

    start_exec = time.time()
    COMPLETE = False
    while not COMPLETE:

        try:

            X_act_CL = np.zeros((robot.nx,T+1))

            X_act_CL[:,0] = X0 #initial condition

            U_CL = np.zeros((robot.nu,T))
            Cost_CL = 0
            replan_count = 0
            replan = False
            replan_index = []

            #closed loop implementation
            for i in range(T):
                #Applying LQR control

                #print "X nom:", X_nom[:,i], " X act:", X_act_CL[:,i]
                if replan:
                    replan_count += 1

                    replan_index.append(i)

                    U_guess = U_opti[:,i:]
                    print "Replanning at step:", i
                    U_opti[:,i:T], X_nom[:,i:T+1],_ = SSP(T-i,X_act_CL[:,i],Q,R,Qf, U_guess) #calculating new nominal

                    #print "Cost_nom:", calc_cost(U_opti,X_nom,T)

                    control.T = T - i #setting new horizon.
                    K_lqr[:,i:T+1] = control.solve_K(robot, X_nom[:,i:T+1], U_opti[:,i:T]) #calculating new LQR gains

                    replan = False

                U_temp = U_opti[:,i] - c.mtimes(c.reshape(K_lqr[:,i],robot.nu,robot.nx),(X_act_CL[:,i] - X_nom[:,i]))

                U_CL[:,i] = np.reshape(control.U_bounds(robot,U_temp),(robot.nu,))

                X_act_CL[:,i+1] = np.reshape(robot.kinematics(X_act_CL[:,i],U_CL[:,i],epsilon),(robot.nx,))

                Cost_CL = Cost_CL + c.mtimes(c.mtimes(c.reshape(X_act_CL[:,i],(1,robot.nx)),Q),c.reshape(X_act_CL[:,i],(robot.nx,1))) + \
                                    c.mtimes(c.mtimes(c.reshape(U_CL[:,i],(1,robot.nu)),R),c.reshape(U_CL[:,i],(robot.nu,1)))

                if params.OBSTACLES:
                    Obstacle_cost = control.obstacle_cost_func(X_act_CL[0:2,i])
                    Cost_CL = Cost_CL + Obstacle_cost

                #enablePrint()
                print "Cost_CL:", Cost_CL, " Cost_nom:", Cost_nom[i], " deviation:", abs(Cost_CL - Cost_nom[i])/Cost_nom[i]
                blockPrint()

                if abs(Cost_CL - Cost_nom[i])/Cost_nom[i] > replan_bound:
                    replan = True

            Cost_CL = Cost_CL + c.mtimes(c.mtimes(c.reshape(Xg - X_act_CL[:,T],(1,robot.nx)),Qf),c.reshape(Xg - X_act_CL[:,T],(robot.nx,1)))
            #enablePrint()
            print "Cost_CL:", Cost_CL, " Cost_nom:", Cost_nom[T], " deviation:", abs(Cost_CL - Cost_nom[T])/Cost_nom[T]
            blockPrint()

            COMPLETE = True

        except:
            print "Unexpected error:", sys.exc_info()

    end_exec = time.time()
    time_taken = OL_time + (end_exec - start_exec)

    return(Cost_CL,time_taken, replan_count)



if __name__=='__main__':

    # if Model == 'car_w_trailers':
    #     #open-loop optimisation gain
    #     R = c.DM([[20,0],[0,10]])
    #     Q = 5*c.DM.eye(robot.nx)
    #     Qf = 1000*c.DM.eye(robot.nx)
    #
    #
    # U_opti, X_nom = SSP(Q,R,Qf)
    #
    # TLQR(U_opti, X_nom, epsilon=0.3)

    TLQR2_run_varying_epsilon()

    # if FILE_WRITE:
    #     file.close()
