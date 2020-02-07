# Author: Mohamed Naveed Gul Mohamed
# email:mohdnaveed96@gmail.com
# Date: Oct 14th 2019
#
# Parameters for simulations
import casadi as c
import math as m
import numpy as np
from robot_models import youBot_model

def Rot_z(angle):

    R =  c.DM([[c.cos(angle), c.sin(angle)],[-c.sin(angle),c.cos(angle)]])
    return R

#obstacles centers
c_obs_1 = c.DM([2.5,.5]);#c.DM([1,1.5]); #
c_obs_2 = c.DM([2,5]);
c_obs_3 = c.DM([-0.7,3]);
c_obs_4 = c.DM([3.5,2]); #c.DM([2.5,3]); #

robot = youBot_model()

#obstacle matrix
E_obs_1 = c.mtimes(c.mtimes(Rot_z(m.radians(30)).T,c.DM([[1/((.6+robot.length)**2),0],[0,1/((.3+robot.length)**2)]])),Rot_z(m.radians(30))) #give a>b always
E_obs_2 = c.mtimes(c.mtimes(Rot_z(m.radians(30)).T,c.DM([[1/((.5+robot.length)**2),0],[0,1/((.5+robot.length)**2)]])),Rot_z(m.radians(30)))
E_obs_3 = c.mtimes(c.mtimes(Rot_z(m.radians(90)).T,c.DM([[1/((.6+robot.length)**2),0],[0,1/((.4+robot.length)**2)]])),Rot_z(m.radians(90)))
E_obs_4 = c.mtimes(c.mtimes(Rot_z(m.radians(60)).T,c.DM([[1/((.4+robot.length)**2),0],[0,1/((.3+robot.length)**2)]])),Rot_z(m.radians(60)))


OBSTACLES = False #True #
M = 100 #obstacle penalty scaling factor

SF = 50 #interagent penalty scaling factor
r_th = 0.8 #radius threshold

Xmin = np.array([-1,-1])
Xmax = np.array([6,6])
