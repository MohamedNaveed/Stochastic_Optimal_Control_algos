# Trajectory optimised Linear Quadratic Regulator Sims
# Motion planning for a robot in a fully observable environment under uncertain process noise.
# Author: Mohamed Naveed Gul Mohamed
# email:mohdnaveed96@gmail.com
# Date: Feb 3rd 2020
import casadi as c
import math as m

from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse
import numpy as np
from numpy import linalg
import pylab
import time

import Algorithm_classes as algo
from robot_models import youBot_model
from robot_models import car_model
from robot_models import quad
import simulation_params as params

T = 30


no_iters = 100

PLOT = False #True #
PLOT_ROBOT_BOUNDARY = False #True #
FILE_WRITE = True #False #
HPRC = False
Model = 'quad'#'car_w_trailers' #'car_model' #'youBot_model'
OL_time = 0 #Open loop time taken.

if 'quad':
    X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    Xg = np.array([2,2,2,0,0,0,0,0,0,0,0,0])

if FILE_WRITE:
    filename = "/home/naveed/Dropbox/Research/Data/WAFR20/Quad/TLQR_1_wo_obs_modified1.csv"
    file = open(filename,"a")
    file.write('epsilon' + ',' + 'Average Cost' + ',' + 'Cost variance' + ',' + 'Average Time' + '\n' )

np.random.seed(4)

if Model == 'quad':
    robot = quad(dt=0.1)
    #open-loop optimisation gain
    R = c.DM([[5,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]])
    Q = c.DM.eye(robot.nx)
    Q[0,0] = 10; Q[1,1] = 10;Q[2,2] = 10;
    Qf = 1000*c.DM.eye(robot.nx)



def SSP(Q,R,Qf):
    #State space planning

    control = algo.SSP(T, X0, Xg, R, Q, Qf, params.Xmin, params.Xmax)

    U_guess = np.zeros((robot.nu,T))

    U_guess = np.reshape(U_guess,(robot.nu*T,1))

    print("Solving state space problem...")
    U_opti_ss = control.solve_SSP(robot,X0, np.zeros(robot.nu), U_guess)
    print("Done.")
    #print("Uss:",U_opti_ss)
    X_path_ss = np.zeros((robot.nx,T+1))
    X_path_ss[:,0] = X0

    for i in range(T):

        X_path_ss[:,i+1] = np.reshape(robot.kinematics(X_path_ss[:,i],U_opti_ss[:,i]),(robot.nx,)) #to make it compatible in dimension reshaping from (3,1) to 3

    #print "U nom:", U_opti_ss
    return U_opti_ss, X_path_ss

def TLQR_run_varying_epsilon():

    global OL_time
    #LQR gain
    Wx = c.DM.eye(robot.nx) #10
    Wx[0,0] = 10; Wx[1,1] = 10; Wx[2,2] = 10;
    Wxf = 100*c.DM.eye(robot.nx)  #100
    Wu = c.DM.eye(robot.nu)

    OL_start = time.time()
    U_opti, X_nom = SSP(Q,R,Qf)

    control = algo.LQR(Wx,Wu,Wxf,T)

    print "U_nom:", U_opti
    #print "X_nom:", X_nom

    print("Solving for LQR gain...")
    K_lqr = control.solve_K(robot, X_nom, U_opti)

    OL_end = time.time()

    OL_time = OL_end - OL_start

    epsi_range = np.linspace(0.0,.2,21)
    #epsi_range = np.append(epsi_range, np.linspace(0.45,1.00,12), axis=0)
    print "Epsilon range:", epsi_range

    for epsilon in epsi_range:
        TLQR(control, U_opti, X_nom, K_lqr, epsilon)

def TLQR(control, U_opti, X_nom, K_lqr, epsilon):


    print("Executing Plan...")
    #Execution of plan

    start_exec = time.time()

    Cost_CL_iter = np.zeros(no_iters)


    for iter in range(no_iters):

        X_act_CL = np.zeros((robot.nx,T+1))

        X_act_CL[:,0] = X0 #initial condition

        U_CL = np.zeros((robot.nu,T))
        Cost_CL = 0

        #closed loop implementation
        for i in range(T):
            #Applying LQR control

            #print "X nom:", X_nom[:,i], " X act:", X_act_CL[:,i]

            U_temp = U_opti[:,i] - c.mtimes(c.reshape(K_lqr[:,i],robot.nu,robot.nx),(X_act_CL[:,i] - X_nom[:,i]))

            U_CL[:,i] = np.reshape(control.U_bounds(robot,U_temp),(robot.nu,))

            X_act_CL[:,i+1] = np.reshape(robot.kinematics(X_act_CL[:,i],U_CL[:,i],epsilon),(robot.nx,))

            Cost_CL = Cost_CL + c.mtimes(c.mtimes(c.reshape(X_act_CL[:,i],(1,robot.nx)),Q),c.reshape(X_act_CL[:,i],(robot.nx,1))) + \
                                c.mtimes(c.mtimes(c.reshape(U_CL[:,i],(1,robot.nu)),R),c.reshape(U_CL[:,i],(robot.nu,1)))

            if params.OBSTACLES:
                Obstacle_cost = control.obstacle_cost_func(X_act_CL[0:2,i])
                Cost_CL = Cost_CL + Obstacle_cost


        Cost_CL = Cost_CL + c.mtimes(c.mtimes(c.reshape(Xg - X_act_CL[:,T],(1,robot.nx)),Qf),c.reshape(Xg - X_act_CL[:,T],(robot.nx,1)))

        Cost_CL_iter[iter] = Cost_CL

    end_exec = time.time()
    time_taken = OL_time + (end_exec - start_exec)/no_iters

    print "epsilon:",epsilon , "   Average Cost:",np.mean(Cost_CL_iter), "  Cost var:", np.var(Cost_CL_iter), " Time taken:",time_taken

    if FILE_WRITE:
        file.write(str(epsilon) + ',' + str(np.mean(Cost_CL_iter)) + ',' + str(np.var(Cost_CL_iter)) + ',' + str(time_taken) + '\n')

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

    TLQR_run_varying_epsilon()

    if FILE_WRITE:
        file.close()
