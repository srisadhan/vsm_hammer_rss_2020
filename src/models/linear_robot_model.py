from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections
import pandas as pd 

import pyomo.environ as pyo
import pyomo.dae as pyodae

def robot_linear_dynamics(tf, config):
    """PYOMO model for the robot's movement in 1-Dof. Trajectory optimization based on maximum end-effector velocity when it hits the nail.

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
    alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']

    # initialize the PYOMO model
    m               = pyo.ConcreteModel()

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
    m.constr.add(m.baseA[0] == 0.0)
    
    # Intermediate time constraints
    # constraints to make sure the hammer hits the nail in the optimization problem
    m.constr.add(m.baseD[round(alpha*tf, 4)] >= nail_pos + 0.01)
    m.constr.add(m.baseD[round(alpha*tf, 4)] <= nail_pos + pos_threshold)

    # Final time constraints
    m.constr.add(m.baseD[tf] == end_pos)
    m.constr.add(m.baseV[tf] == 0.0)

   

    # Cost function
    m.obj       = pyo.Objective(expr= (m.baseV[round(alpha*tf, 4)]), sense = pyo.maximize)

    pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=N, scheme='BACKWARD')
    pyo.SolverFactory('ipopt').solve(m).write()

    # Extract the Results
    t          = np.array([t for t in m.t])
    baseD      = np.array([m.baseD[t]() for t in m.t])
    baseV      = np.array([m.baseV[t]() for t in m.t])
    baseA      = np.array([m.baseA[t]() for t in m.t])
    hammerD    = np.array([0 for t in m.t])
    hammerV    = np.array([0 for t in m.t])
    magnetD1   = np.array([-config['w'] for t in m.t])
    magnetD2   = np.array([ config['w'] for t in m.t])

    # save the end-effector and magnet displacement
    df = pd.DataFrame({'time' : t,
                        'bd'  : baseD,
                        'bv'  : baseV,
                        'hd'  : hammerD,
                        'hv'  : hammerV,
                        'md1' : magnetD1,
                        'md2' : magnetD2,
                        })

    return df, m


def robot_linear_dynamics_optimize_tf(alpha, weights, config):
    """PYOMO model for the robot's movement in 1-Dof. 
    Trajectory and Time optimization for maximum end-effector velocity when it hits the nail (@Alpha).

    Arguments:
        alpha {float}  -- fraction of tf at which the hammer hits the nail. lies in the range [0,1]
        weights {list} -- list of weights for multi-objective optimizing: weights[0] to tf and weights[1] to hammer velocity
        config {yaml}  -- configurations imported from the config.yaml file
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
    m.tf            = pyo.Var(within=pyo.NonNegativeReals, bounds=(config['tf_range'][0], config['tf_range'][1]), initialize=1.5)
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
    m.constr.add(m.baseD[alpha] >= nail_pos + 0.01)
    m.constr.add(m.baseD[alpha] <= nail_pos + pos_threshold)

    # Final time constraints
    m.constr.add(m.baseD[1.0] == end_pos)
    m.constr.add(m.baseV[1.0] == 0.0)   

    # Cost function
    m.obj       = pyo.Objective(expr= -weights[0] * m.tf + weights[1] * (m.baseV[alpha]), sense = pyo.maximize)

    pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=500, scheme='BACKWARD')
    pyo.SolverFactory('ipopt').solve(m).write()

    # Extract the Results
    tf         = pyo.value(m.tf) 
    t          = np.array([t * tf for t in m.t])
    baseD      = np.array([m.baseD[t]() for t in m.t])
    baseV      = np.array([m.baseV[t]() / tf for t in m.t])
    baseA      = np.array([m.baseA[t]() / tf**2 for t in m.t])
    hammerD    = np.array([0 for t in m.t])
    hammerV    = np.array([0 for t in m.t])
    magnetD1   = np.array([-config['w'] for t in m.t])
    magnetD2   = np.array([ config['w'] for t in m.t])

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