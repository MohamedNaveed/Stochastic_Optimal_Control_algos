#!/usr/bin/env python

# Belief-space planning.
# Motion planning for a robot in an uncertain environment.
# Author: Mohamed Naveed Gul Mohamed
# email:mohdnaveed96@gmail.com
# Date: Nov 14th 2019

import multiprocessing as mp
import numpy as np

import TLQR_replan_sims as sims_tlqr
import MPC_sims as sims_mpc

MPC_BOOL = False
MPC_FAST = False
TLQR_REPLAN = False
TLQR_SH_BOOL = True

if TLQR_REPLAN:
        filename = "TLQR2_1_wo_obs_modified.csv"
        file = open(filename,"a")
        file.write('epsilon' + ',' + 'Average Cost' + ',' + 'Cost variance' + ',' + 'Average Time' + ',' + 'Average Replans' + ','
                + 'Replan Variance' + ',' + 'replan_bound' + '\n' )

if MPC_BOOL:
        filename = "MPC_1_wo_obs_modified.csv"
        file = open(filename,"a")
        file.write('epsilon' + ',' + 'Average Cost' + ',' + 'Cost variance' + ',' + 'Average Time' + '\n' )

no_iters = 100

if __name__=='__main__':

    epsi_range = np.linspace(0.0,1.00,21)
    replan_bound = 0.05
    print "Epsilon range:", epsi_range

    sims_tlqr.blockPrint()

    for epsilon in epsi_range:

        pool = mp.Pool(100)

        Cost_CL_iter = np.zeros(no_iters)
        Time_iter = np.zeros(no_iters) #store time for each iteration
        Replan_iter = np.zeros(no_iters) #store replan count for each iteration

        if TLQR_REPLAN:
                result_objects = [pool.apply_async(sims_tlqr.TLQR2_hprc, args=(epsilon,replan_bound)) for iter in range(no_iters)]
                Replan_iter = [r.get()[2] for r in result_objects]

        if MPC_BOOL:
                result_objects = [pool.apply_async(sims_mpc.MPC_hprc, args=(epsilon)) for iter in range(no_iters)]

        Cost_iter = [r.get()[0] for r in result_objects]
        Time_iter = [r.get()[1] for r in result_objects]



        pool.close()
        pool.join()
        sims_tlqr.enablePrint()
        if TLQR_REPLAN:
                print("epsilon:",epsilon , "   Average Cost:",np.mean(Cost_iter),
                        " Cost Variance:", np.var(Cost_iter), " Average replans:", np.mean(Replan_iter),
                        " Avg time:", np.mean(Time_iter), " Replan Var:", np.var(Replan_iter),
                        " Replan bound:", replan_bound)

                file.write(str(epsilon) + ',' + str(np.mean(Cost_iter)) + ',' + str(np.var(Cost_iter))
                                + ',' + str(np.mean(Time_iter)) + ',' + str(np.mean(Replan_iter))
                                + ',' + str(np.var(Replan_iter)) + ',' + str(replan_bound) + '\n')

        if MPC_BOOL:
                print("epsilon:",epsilon , "   Average Cost:",np.mean(Cost_iter),
                        " Cost Variance:", np.var(Cost_iter)," Avg time:", np.mean(Time_iter))
                file.write(str(epsilon) + ',' + str(np.mean(Cost_iter)) + ',' + str(np.var(Cost_iter))
                                + ',' + str(np.mean(Time_iter)) + '\n')

        sims_tlqr.blockPrint()



    sims_tlqr.enablePrint()
    print("Completed")
    file.close()
