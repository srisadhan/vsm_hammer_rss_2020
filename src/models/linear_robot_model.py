from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections
import pandas as pd 

import pyomo.environ as pyo
import pyomo.dae as pyodae
from pyomo.opt.parallel import SolverManagerFactory

def robot_linear_dynamics(tf, config):
    """PYOMO model for the robot's movement in 1-Dof. Trajectory optimization based on maximum end-effector velocity when it hits the nail.

    Arguments:
        tf {float} -- final time mentioned by the user
        config {yaml} -- configurations imported from the config.yaml file
    """

    def traj_constr1(m, t):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail
        
        Arguments:
            m {pyomo model} -- model created in pyomo
            t {pyomo variable} -- independent time variable in pyomo
        """
        if t < config['alpha'] * m.tf:
            return m.baseD[t] <= config['nail_pos']
        elif t > config['alpha'] * m.tf:
            return m.baseD[t] >= config['nail_pos']
        else:
            return pyo.Constraint.Skip

    def traj_constr2(m, t):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail
        
        Arguments:
            m {pyomo model} -- model created in pyomo
            t {pyomo variable} -- independent time variable in pyomo
        """
        if t > config['alpha']:
            return m.baseD[t] <= config['nail_pos'] + config['pos_threshold'] - 0.015
        else:
            return pyo.Constraint.Skip

    # Robot parameters
    freq            = config['freq']

    # Model paramters
    M               = config['M']

    # Selection of parameters
    start_pos       = config['start_pos']
    nail_pos        = config['nail_pos']
    end_pos         = config['end_pos']
    alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']

    # initialize the PYOMO model
    m               = pyo.ConcreteModel()
    m.tf            = pyo.Param(initialize=tf)
    
    # time variables
    if tf < 1.0 :
        N   = freq
    else:
        N   = int(round(tf * (freq)))

    tvec            = np.around(np.linspace(0, tf, N+1), decimals=4).tolist()
    m.t             = pyodae.ContinuousSet(bounds=(0,tf), initialize=tvec)

    # initialize variables
    m.baseD         = pyo.Var(m.t, bounds=(config['min_robot_disp'], nail_pos + pos_threshold), initialize=0.1)
   

    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t, bounds=(-config['max_robot_vel'], config['max_robot_vel']))
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t, bounds=(-config['max_robot_acc'], config['max_robot_acc']))

    # States at initial Time
    m.constr        = pyo.ConstraintList()
    m.constr.add(m.baseD[0] == start_pos)
    m.constr.add(m.baseV[0] == 0.0)
    
    # Intermediate time constraints
    # constraints to make sure the hammer hits the nail in the optimization problem
    m.NailHit1    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr1(m, t))
    m.NailHit2    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr2(m, t))

    # Final time constraints
    m.constr.add(m.baseD[tf] == end_pos)
    m.constr.add(m.baseV[tf] == 0.0)

    # Cost function
    m.obj       = pyo.Objective(expr= (m.baseV[round(alpha*tf, 4)]), sense = pyo.maximize)

    return m


def robot_linear_dynamics_optimize_tf(weights, config):
    """PYOMO model for the robot's movement in 1-Dof. 
    Trajectory and Time optimization for maximum end-effector velocity when it hits the nail (@Alpha).

    Arguments:
        weights {list} -- list of weights for multi-objective optimizing: weights[0] to tf and weights[1] to hammer velocity
        config {yaml}  -- configurations imported from the config.yaml file
    """
    def traj_constr1(m, t, config):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail
        
        Arguments:
            m {pyomo model} -- model created in pyomo
            t {pyomo variable} -- independent time variable in pyomo
        """
        if t < config['alpha']:
            return m.baseD[t] <= config['nail_pos']
        elif t > config['alpha']:
            return m.baseD[t] >= config['nail_pos']
        else:
            return pyo.Constraint.Skip

    def traj_constr2(m, t, config):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail
        
        Arguments:
            m {pyomo model} -- model created in pyomo
            t {pyomo variable} -- independent time variable in pyomo
        """
        if t > config['alpha']:
            return m.baseD[t] <= config['nail_pos'] + config['pos_threshold'] - 0.01
        else:
            return pyo.Constraint.Skip

    def obj_function(m, weights, config):
        """ Obj function for multiobjective optimization - w1 * ((f1 -f1l)/(f1u-f1l))^2 + w2 * ((f2-f2l)/(f2u-f2l))^2."""

        return - weights[0] * ((m.tf - config['tf_range'][0]) / (config['tf_range'][1] - config['tf_range'][0]))**2 
        + weights[1] * (m.baseV[alpha] + 0.5)**2


    # Robot parameters
    freq            = config['freq']

    # Model paramters
    M               = config['M']

    # Selection of parameters
    start_pos       = config['start_pos']
    nail_pos        = config['nail_pos']
    end_pos         = config['end_pos']
    alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']

    # initialize the PYOMO model
    m               = pyo.ConcreteModel()

    # time variables
    m.tf            = pyo.Var(within=pyo.NonNegativeReals, bounds=(config['tf_range'][0], config['tf_range'][1]), initialize=1.5)
    tvec            = np.around(np.linspace(0, 1, freq + 1), decimals=4).tolist()
    m.t             = pyodae.ContinuousSet(bounds=(0,1), initialize=tvec)

    # initialize variables
    m.baseD         = pyo.Var(m.t, bounds=(config['min_robot_disp'], nail_pos + pos_threshold), initialize=0.1)
   

    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t) 
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t) 

    # Bounds on the states
    m.velConstr1    = pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] >= -config['max_robot_vel'] * m.tf)
    m.velConstr2    = pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] <=  config['max_robot_vel'] * m.tf) 
    m.accConstr1    = pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] >= -config['max_robot_acc'] * m.tf**2)
    m.accConstr2    = pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] <=  config['max_robot_acc'] * m.tf**2) 

    # States at initial Time
    m.constr        = pyo.ConstraintList()
    m.constr.add(m.baseD[0] == start_pos)
    m.constr.add(m.baseV[0] == 0.0)  
    
    # Intermediate time constraints
    m.NailHit1    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr1(m, t, config))
    m.NailHit2    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr2(m, t, config))

    # Final time constraints
    m.constr.add(m.baseD[1.0] == end_pos)
    m.constr.add(m.baseV[1.0] == 0.0)   

    # Cost function
    m.obj       = pyo.Objective(expr= obj_function(m, weights, config) , sense = pyo.maximize)

    return m


def robot_linear_dynamics_optimize_tf_and_alpha(config):
    """PYOMO model for the robot's movement in 1-Dof. 
    Trajectory, Time and Alpha optimization for maximum end-effector velocity @Alpha.
    Arguments:
        tf {float} -- final time mentioned by the user
        config {yaml} -- configurations imported from the config.yaml file
    """

    # Robot parameters
    freq            = config['freq']

    # Model paramters
    M               = config['M']

    # Selection of parameters
    start_pos       = config['start_pos']
    nail_pos        = config['nail_pos']
    end_pos         = config['end_pos']
    # alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']

    # initialize the PYOMO model
    m               = pyo.ConcreteModel()

    # time variables
    m.tf            = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 10.0), initialize=1.5)
    m.alpha         = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 1.0), initialize=0.8)

    tvec            = np.around(np.linspace(0, 1, freq + 1), decimals=4).tolist()
    m.t             = pyodae.ContinuousSet(bounds=(0,1), initialize=tvec)

    # initialize variables
    m.baseD         = pyo.Var(m.t, bounds=(config['min_robot_disp'], nail_pos + pos_threshold), initialize=0.1)
   

    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t) #, bounds=(-config['max_robot_vel'] * m.tf, config['max_robot_vel'] * m.tf))
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t) #, bounds=(-config['max_robot_acc'] * pyo.value(m.tf)**2, config['max_robot_acc'] * pyo.value(m.tf)**2))

    # Bounds on the states
    m.velConstr1    = pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] >= -config['max_robot_vel'] * m.tf)
    m.velConstr2    = pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] <=  config['max_robot_vel'] * m.tf) 
    m.accConstr1    = pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] >= -config['max_robot_acc'] * m.tf**2)
    m.accConstr2    = pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] <=  config['max_robot_acc'] * m.tf**2) 

    # States at initial Time
    m.constr        = pyo.ConstraintList()
    m.constr.add(m.baseD[0] == start_pos)
    m.constr.add(m.baseV[0] == 0.0)
    m.constr.add(m.baseA[0] == 0.0)
    
    # Intermediate time constraints
    # constraints to make sure the hammer hits the nail in the optimization problem
    m.constr.add(m.baseD[pyo.value(m.alpha)] >= nail_pos + 0.01)
    m.constr.add(m.baseD[pyo.value(m.alpha)] <= nail_pos + pos_threshold)

    # Final time constraints
    m.constr.add(m.baseD[1.0] == end_pos)
    m.constr.add(m.baseV[1.0] == 0.0)

   

    # Cost function
    m.obj       = pyo.Objective(expr= (- m.tf + m.baseV[pyo.value(m.alpha)] + m.alpha), sense = pyo.maximize)

    pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=500, scheme='BACKWARD')
    pyo.SolverFactory('ipopt').solve(m).write()

    print("the optimal time is: %f and value is: %f" %(pyo.value(m.tf), pyo.value(m.alpha)))

    # Extract the Results
    tf         = pyo.value(m.tf) 
    t          = np.array([t * tf for t in m.t])
    baseD      = np.array([m.baseD[t]() for t in m.t])
    baseV      = np.array([m.baseV[t]() / tf for t in m.t])
    baseA      = np.array([m.baseA[t]() / tf**2 for t in m.t])
    hammerD    = np.array([0 for t in m.t])
    hammerV    = np.array([0 for t in m.t])
    magnetD1   = np.array([-config['w'] / 2 for t in m.t])
    magnetD2   = np.array([ config['w'] / 2 for t in m.t])

    # save the end-effector and magnet displacement
    df = pd.DataFrame({'time' : t,
                        'bd'  : baseD,
                        'bv'  : baseV,
                        'hd'  : hammerD,
                        'hv'  : hammerV,
                        'md1' : magnetD1,
                        'md2' : magnetD2,
                        })
    
    return df, m, tf