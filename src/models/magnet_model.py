from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections 
import pandas as pd

import pyomo.environ as pyo
import pyomo.dae as pyodae
from pyomo.opt.parallel import SolverManagerFactory

# Not sure if this is a good cost function
def model_cost(m, config):
    """Calculate the cost funtion of the using the hammer velocity
    
    Arguments:
        m {pyomo model} --  a model of pyomo created using pyomo.ConcreteModel() or pyomo.AbstractModel()
        config {yaml} -- configuration mentioned in the config.yaml file
    
    Returns:
        pyomo cost -- cost function return for pyomo
    """
    return sum(abs(m.hammerV[t]) for t in m.t if pyo.value(m.hammerD[t] + m.baseD[t] >= config['nail_pos']))


def two_magnets_moving(tf, setting, config):
    """PYOMO model for the VSM with two moving magnets controlled symmetrically in opposite direction by a single actuator
    The problem formulation is according to: 
    In upright position of the VSM: left --> Magnet1 and right --> Magnet2

    Arguments:
        tf {float} -- final time mentioned by the user
        setting {string} -- Stiffness setting from one of the three options: 'high_stiffness','low_stiffness','variable_stiffness'
        config {yaml} -- configurations imported from the config.yaml file
    """
    # Robot parameters
    freq            = config['freq']

    # Model paramters
    C1              = config['C1']
    C2              = config['C2']
    M               = config['M']
    d               = config['d']
    w               = config['w']
    udot_max        = config['udot_max']    
    u_min           = config['no_mag_fixed'][0]
    u_max           = config['no_mag_fixed'][1]

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
    m.hammerD       = pyo.Var(m.t, initialize=0.0)
    m.Sep1          = pyo.Var(m.t)
    m.Sep2          = pyo.Var(m.t)

    # active magnets
    m.magnetD1      = pyo.Var(m.t, bounds=(u_min, -w))
    m.magnetD2      = pyo.Var(m.t, bounds=(    w,  u_max))
   

    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t, bounds=(-config['max_robot_vel'], config['max_robot_vel']))
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t, bounds=(-config['max_robot_acc'], config['max_robot_acc']))
    m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t)
    m.hammerA       = pyodae.DerivativeVar(m.hammerV, wrt=m.t)
    m.magnetV1      = pyodae.DerivativeVar(m.magnetD1, wrt=m.t, bounds=(-udot_max, udot_max))
    m.magnetV2      = pyodae.DerivativeVar(m.magnetD2, wrt=m.t, bounds=(-udot_max, udot_max))

    # Constraints on hammer movement
    m.hammerDisp1  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] <= (m.magnetD2[t] - w))
    m.hammerDisp2  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] >= (m.magnetD1[t] + w))

    # Constraints on the separation between magnets
    m.sep_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] - m.magnetD1[t] - w)
    m.sep_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD2[t] - m.hammerD[t] - w)
    
    # System dynamics
    m.hammerAcc    = pyo.Constraint(m.t, rule=lambda m, t: m.hammerA[t] == - (M * m.baseA[t] + d * m.hammerV[t] + C1 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M)

    if setting == 'high_stiffness':
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == -w)
        m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD2[t] ==  w)
    elif setting == 'low_stiffness' :
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == u_min)
        m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD2[t] == u_max)
    

    # States at initial Time
    m.constr    = pyo.ConstraintList()
    m.constr.add(m.baseD[0] == start_pos)
    m.constr.add(m.baseV[0] == 0.0)
    m.constr.add(m.baseA[0] == 0.0)
    m.constr.add(m.hammerD[0] == 0.0)
    m.constr.add(m.hammerV[0] == 0.0)
    m.constr.add(m.hammerA[0] == 0.0)
    m.constr.add(m.magnetV1[0] == 0.0)
    m.constr.add(m.magnetV2[0] == 0.0)

    if setting == 'high_stiffness' :
        m.constr.add(m.magnetD1[0] == -w)
        m.constr.add(m.magnetD2[0] ==  w)
    elif setting == 'low_stiffness' :
        m.constr.add(m.magnetD1[0] == u_min)
        m.constr.add(m.magnetD2[0] == u_max)


    # Intermediate time constraints
    # constraints to make sure the hammer hits the nail in the optimization problem
    m.constr.add(m.baseD[round(alpha*tf, 4)]+m.hammerD[round(alpha*tf, 4)] >= nail_pos + 0.01)
    m.constr.add(m.baseD[round(alpha*tf, 4)]+m.hammerD[round(alpha*tf, 4)] <= nail_pos + pos_threshold)

    # Final time constraints
    m.constr.add(m.baseD[tf] == end_pos)
    m.constr.add(m.baseV[tf] == 0.0)
    
    # Objective
    m.obj       = pyo.Objective(expr= (m.hammerV[round(alpha*tf, 4)]) , sense = pyo.maximize)

    pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=N, scheme='BACKWARD')
    pyo.SolverFactory('ipopt').solve(m).write()

    # Extract the Results
    t          = np.array([t for t in m.t])
    baseD      = np.array([m.baseD[t]() for t in m.t])
    baseV      = np.array([m.baseV[t]() for t in m.t])
    baseA      = np.array([m.baseA[t]() for t in m.t])

    hammerD    = np.array([m.hammerD[t]() for t in m.t])
    hammerV    = np.array([m.hammerV[t]() for t in m.t])

    magnetD1   = np.array([m.magnetD1[t]() for t in m.t])
    magnetD2   = np.array([m.magnetD2[t]() for t in m.t])
    magnetV1   = np.array([m.magnetV1[t]() for t in m.t])
    magnetV2   = np.array([m.magnetV2[t]() for t in m.t])
    
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

# FIXME: this implementation is not working for some reason in variable_stiffness setting--(the magnet positions are off)
# Implementation using by defining two magnet inputs
# def two_magnets_moving_optimize_tf(alpha, setting, weights, config):
#     """PYOMO model for the VSM with two moving magnets controlled symmetrically in opposite direction by a single actuator
#     The problem formulation is according to: 
#     In upright position of the VSM: left --> Magnet1 and right --> Magnet2

#     Arguments:
#         alpha {float}    -- Value between (0,1) which decides the time at which the hammer hits the nail.
#         setting {string} -- Stiffness setting from one of the three options: 'high_stiffness','low_stiffness','variable_stiffness'
#         weights {list}   -- list of weights for multi-objective optimizing: weights[0] to tf and weights[1] to hammer velocity
#         config {yaml}    -- configurations imported from the config.yaml file
#     """
#     # Robot parameters
#     freq            = config['freq']

#     # Model paramters
#     C1              = config['C1']
#     C2              = config['C2']
#     M               = config['M']
#     d               = config['d']
#     w               = config['w']
#     udot_max        = config['udot_max']    
#     u_min           = config['no_mag_fixed'][0]
#     u_max           = config['no_mag_fixed'][1]

#     # Selection of parameters
#     start_pos       = config['start_pos']
#     nail_pos        = config['nail_pos']
#     end_pos         = config['end_pos']
#     alpha           = config['alpha']
#     pos_threshold   = config['pos_threshold']

#     # initialize the PYOMO model
#     m               = pyo.ConcreteModel()

#     # Scaled time t in the range [0, 1]
#     m.tf            = pyo.Var(within=pyo.NonNegativeReals, bounds=(config['tf_range'][0], config['tf_range'][1]), initialize=1.5)
#     tvec            = np.around(np.linspace(0, 1.0, freq + 1), decimals=4).tolist()
#     m.t             = pyodae.ContinuousSet(bounds=(0, 1.0), initialize=tvec)

#     # initialize variables
#     m.baseD         = pyo.Var(m.t, bounds=(config['min_robot_disp'], nail_pos + pos_threshold), initialize=0.1)
#     m.hammerD       = pyo.Var(m.t, initialize=0.0)
#     m.Sep1          = pyo.Var(m.t)
#     m.Sep2          = pyo.Var(m.t)

#     # active magnets
#     m.magnetD1      = pyo.Var(m.t, bounds=(u_min, -w))
#     m.magnetD2      = pyo.Var(m.t, bounds=(    w,  u_max))
   

#     # Derivatives of the variables
#     m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t)
#     m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t)
#     m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t)
#     m.hammerA       = pyodae.DerivativeVar(m.hammerV, wrt=m.t)
#     m.magnetV1      = pyodae.DerivativeVar(m.magnetD1, wrt=m.t) #, bounds=(-udot_max , udot_max))
#     m.magnetV2      = pyodae.DerivativeVar(m.magnetD2, wrt=m.t) #, bounds=(-udot_max, udot_max))

#     # Constraints on hammer displacement
#     m.hammerDisp1  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] <= (m.magnetD2[t] - w))
#     m.hammerDisp2  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] >= (m.magnetD1[t] + w))

#     # Constraints on robot velocity and acceleration
#     m.baseVelConst1= pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] <= m.tf *  config['max_robot_vel'])
#     m.baseVelConst2= pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] >= m.tf * -config['max_robot_vel'])
#     m.baseAccConst1= pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] <= m.tf**2 *  config['max_robot_acc'])
#     m.baseAccConst2= pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] >= m.tf**2 * -config['max_robot_acc'])

#     # Constraints on the separation between magnets
#     # m.mag_D1_D2    = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == - m.magnetD2[t]) 
#     m.sep_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] - m.magnetD1[t] - w)
#     m.sep_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD2[t] - m.hammerD[t] - w)
#     m.vel_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV1[t] <= m.tf * udot_max )
#     m.vel_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV1[t] >= m.tf * -udot_max)
#     m.vel_ham_mag3 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV2[t] <= m.tf * udot_max )
#     m.vel_ham_mag4 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV2[t] >= m.tf * -udot_max)
    
#     # System dynamics
#     m.hammerAcc    = pyo.Constraint(m.t, rule=lambda m, t: m.hammerA[t] == - (M * m.baseA[t] + d * m.tf * m.hammerV[t] + C1 * m.tf**2 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M)

#     if setting == 'high_stiffness':
#         m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == -w)
#         m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD2[t] ==  w)
#     elif setting == 'low_stiffness' :
#         m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == u_min)
#         m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD2[t] == u_max)
#     else:
#         m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == - m.magnetD2[t])

#     # States at initial Time
#     m.constr    = pyo.ConstraintList()
#     m.constr.add(m.baseD[0] == start_pos)
#     m.constr.add(m.baseV[0] == 0.0)
#     m.constr.add(m.baseA[0] == 0.0)
#     m.constr.add(m.hammerD[0] == 0.0)
#     m.constr.add(m.hammerV[0] == 0.0)
#     m.constr.add(m.hammerA[0] == 0.0)
#     m.constr.add(m.magnetV1[0] == 0.0)
#     m.constr.add(m.magnetV2[0] == 0.0)

#     if setting == 'high_stiffness' :
#         m.constr.add(m.magnetD1[0] == -w)
#         m.constr.add(m.magnetD2[0] ==  w)
#     elif setting == 'low_stiffness' :
#         m.constr.add(m.magnetD1[0] == u_min)
#         m.constr.add(m.magnetD2[0] == u_max)


#     # Intermediate time constraints
#     # constraints to make sure the hammer hits the nail in the optimization problem
#     m.constr.add(m.baseD[alpha]+m.hammerD[alpha] >= nail_pos + 0.01)
#     m.constr.add(m.baseD[alpha]+m.hammerD[alpha] <= nail_pos + pos_threshold)

#     # Final time constraints
#     m.constr.add(m.baseD[1.0] == end_pos)
#     m.constr.add(m.baseV[1.0] == 0.0)

#     # Objective
#     m.obj       = pyo.Objective(expr=  - weights[0] * m.tf + weights[1] * (m.hammerV[alpha]) , sense = pyo.maximize)

#     pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=500, scheme='BACKWARD')
#     pyo.SolverFactory('ipopt').solve(m).write()

#     # Extract the Results
#     tf         = pyo.value(m.tf)
#     t          = np.array([t * tf for t in m.t])
#     baseD      = np.array([m.baseD[t]() for t in m.t])
#     baseV      = np.array([m.baseV[t]() / tf for t in m.t])
#     baseA      = np.array([m.baseA[t]() / tf**2 for t in m.t])

#     hammerD    = np.array([m.hammerD[t]() for t in m.t])
#     hammerV    = np.array([m.hammerV[t]() / tf for t in m.t])

#     magnetD1   = np.array([m.magnetD1[t]() for t in m.t])
#     magnetD2   = np.array([m.magnetD2[t]() for t in m.t])
#     magnetV1   = np.array([m.magnetV1[t]() / tf for t in m.t])
#     magnetV2   = np.array([m.magnetV2[t]() / tf for t in m.t])

#    # save the end-effector and magnet displacement
#     df = pd.DataFrame({'time' : t,
#                         'bd'  : baseD,
#                         'bv'  : baseV,
#                         'hd'  : hammerD,
#                         'hv'  : hammerV,
#                         'md1' : magnetD1,
#                         'md2' : magnetD2,
#                         })
    
#     return df, m, tf

def two_magnets_moving_optimize_tf(alpha, setting, weights, config):
    """PYOMO model for the VSM with two moving magnets controlled symmetrically in opposite direction by a single actuator
    The problem formulation is according to: 
    In upright position of the VSM: left --> Magnet1 and right --> Magnet2

    Arguments:
        alpha {float}    -- Value between (0,1) which decides the time at which the hammer hits the nail.
        setting {string} -- Stiffness setting from one of the three options: 'high_stiffness','low_stiffness','variable_stiffness'
        weights {list}   -- list of weights for multi-objective optimizing: weights[0] to tf and weights[1] to hammer velocity
        config {yaml}    -- configurations imported from the config.yaml file
    """
    # Robot parameters
    freq            = config['freq']

    # Model paramters
    C1              = config['C1']
    C2              = config['C2']
    M               = config['M']
    d               = config['d']
    w               = config['w']
    udot_max        = config['udot_max']    
    u_min           = config['no_mag_fixed'][0]
    u_max           = config['no_mag_fixed'][1]
    
    if setting == 'variable_stiffness':
        hammer_vel_range = config['hammer_vel_range_VS']
    elif setting == 'low_stiffness':
        hammer_vel_range = config['hammer_vel_range_LS']
    else:
        hammer_vel_range = [0.0, 0.0]

    # Selection of parameters
    start_pos       = config['start_pos']
    nail_pos        = config['nail_pos']
    end_pos         = config['end_pos']
    alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']

    # initialize the PYOMO model
    m               = pyo.ConcreteModel()

    # Scaled time t in the range [0, 1]
    m.tf            = pyo.Var(within=pyo.NonNegativeReals, bounds=(config['tf_range'][0], config['tf_range'][1]), initialize=1.5)
    tvec            = np.around(np.linspace(0, 1.0, freq + 1), decimals=4).tolist()
    m.t             = pyodae.ContinuousSet(bounds=(0, 1.0), initialize=tvec)

    # initialize variables
    m.baseD         = pyo.Var(m.t, bounds=(config['min_robot_disp'], nail_pos + pos_threshold), initialize=0.1)
    m.hammerD       = pyo.Var(m.t, initialize=0.0)
    m.Sep1          = pyo.Var(m.t)
    m.Sep2          = pyo.Var(m.t)

    # active magnets
    m.magnetD       = pyo.Var(m.t, bounds=(w,  u_max))   

    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t)
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t)
    m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t)
    m.hammerA       = pyodae.DerivativeVar(m.hammerV, wrt=m.t)
    m.magnetV       = pyodae.DerivativeVar(m.magnetD, wrt=m.t)
   

    # Constraints on hammer displacement
    m.hammerDisp1  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] <= (m.magnetD[t] - w))
    m.hammerDisp2  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] >= (-m.magnetD[t] + w))

    # Constraints on robot velocity and acceleration
    m.baseVelConst1= pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] <= m.tf *  config['max_robot_vel'])
    m.baseVelConst2= pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] >= m.tf * -config['max_robot_vel'])
    m.baseAccConst1= pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] <= m.tf**2 *  config['max_robot_acc'])
    m.baseAccConst2= pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] >= m.tf**2 * -config['max_robot_acc'])

    # Constraints on the separation between magnets
    # m.mag_D1_D2    = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == - m.magnetD2[t]) 
    m.sep_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] + m.magnetD[t] - w)
    m.sep_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD[t] - m.hammerD[t] - w)
    m.vel_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV[t] <= m.tf * udot_max )
    m.vel_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV[t] >= m.tf * -udot_max)
    
    # System dynamics
    m.hammerAcc    = pyo.Constraint(m.t, rule=lambda m, t: m.hammerA[t] == - (M * m.baseA[t] + d * m.tf * m.hammerV[t] + C1 * m.tf**2 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M)

    if setting == 'high_stiffness':
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] ==  w)
    elif setting == 'low_stiffness' :
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] == u_max)

    # States at initial Time
    m.constr    = pyo.ConstraintList()
    m.constr.add(m.baseD[0] == start_pos)
    m.constr.add(m.baseV[0] == 0.0)
    m.constr.add(m.baseA[0] == 0.0)
    m.constr.add(m.hammerD[0] == 0.0)
    m.constr.add(m.hammerV[0] == 0.0)
    m.constr.add(m.hammerA[0] == 0.0)
    m.constr.add(m.magnetV[0] == 0.0)

    if (setting == 'high_stiffness') :
        m.constr.add(m.magnetD[0] ==  w)
    elif (setting == 'low_stiffness') :
        m.constr.add(m.magnetD[0] == u_max)
    elif (setting == 'variable_stiffness'):
        m.constr.add(m.magnetD[0] ==  w)


    # Intermediate time constraints
    # constraints to make sure the hammer hits the nail in the optimization problem
    m.constr.add(m.baseD[alpha]+m.hammerD[alpha] >= nail_pos + 0.01)
    m.constr.add(m.baseD[alpha]+m.hammerD[alpha] <= nail_pos + pos_threshold)

    # Final time constraints
    m.constr.add(m.baseD[1.0] == end_pos)
    m.constr.add(m.baseV[1.0] == 0.0)

    # Objective
    m.obj       = pyo.Objective(expr=  - weights[0] * ((m.tf - config['tf_range'][0]) / (config['tf_range'][1] - config['tf_range'][0]))**2 + weights[1] * ((m.hammerV[alpha] - hammer_vel_range[0]) / (hammer_vel_range[1] - hammer_vel_range[0]))**2 , sense = pyo.maximize)

    pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=500, scheme='BACKWARD')
    # pyo.SolverFactory('ipopt').solve(m).write()

    # opt = pyo.SolverFactory('l-bfgs-b')
    SolverManagerFactory('neos').solve(m, opt='snopt').write()

    # Extract the Results
    tf         = pyo.value(m.tf)
    t          = np.array([t * tf for t in m.t])
    baseD      = np.array([m.baseD[t]() for t in m.t])
    baseV      = np.array([m.baseV[t]() / tf for t in m.t])
    baseA      = np.array([m.baseA[t]() / tf**2 for t in m.t])

    hammerD    = np.array([m.hammerD[t]() for t in m.t])
    hammerV    = np.array([m.hammerV[t]() / tf for t in m.t])

    magnetD1   = np.array([-m.magnetD[t]() for t in m.t])
    magnetD2   = np.array([m.magnetD[t]() for t in m.t])

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


# Change of constraints 
def hammer_nail_constr(m, t):
    if  t > m.alpha - 0.2:
        return pyo.Constraint.Skip
    return m.hammerD[t] + m.baseD[t] <= m.nail_pos

def two_magnets_moving_optimize_tf_New(alpha, setting, weights, config):
    """PYOMO model for the VSM with two moving magnets controlled symmetrically in opposite direction by a single actuator
    The problem formulation is according to: 
    In upright position of the VSM: left --> Magnet1 and right --> Magnet2

    Arguments:
        alpha {float}    -- Value between (0,1) which decides the time at which the hammer hits the nail.
        setting {string} -- Stiffness setting from one of the three options: 'high_stiffness','low_stiffness','variable_stiffness'
        weights {list}   -- list of weights for multi-objective optimizing: weights[0] to tf and weights[1] to hammer velocity
        config {yaml}    -- configurations imported from the config.yaml file
    """
    # Robot parameters
    freq            = config['freq']

    # Model paramters
    C1              = config['C1']
    C2              = config['C2']
    M               = config['M']
    d               = config['d']
    w               = config['w']
    udot_max        = config['udot_max']    
    u_min           = config['no_mag_fixed'][0]
    u_max           = config['no_mag_fixed'][1]

    # Selection of parameters
    start_pos       = config['start_pos']
    # nail_pos        = config['nail_pos']
    end_pos         = config['end_pos']
    # alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']

    # initialize the PYOMO model
    m               = pyo.ConcreteModel()
    m.alpha         = pyo.Param(initialize=config['alpha'])
    m.nail_pos      = pyo.Param(initialize=config['nail_pos'])

    # Scaled time t in the range [0, 1]
    m.tf            = pyo.Var(within=pyo.NonNegativeReals, bounds=(config['tf_range'][0], config['tf_range'][1]), initialize=1.5)
    tvec            = np.around(np.linspace(0, 1.0, freq + 1), decimals=4).tolist()
    m.t             = pyodae.ContinuousSet(bounds=(0, 1.0), initialize=tvec)

    # initialize variables
    m.baseD         = pyo.Var(m.t, bounds=(config['min_robot_disp'], m.nail_pos + pos_threshold), initialize=0.1)
    m.hammerD       = pyo.Var(m.t, initialize=0.0)
    m.Sep1          = pyo.Var(m.t)
    m.Sep2          = pyo.Var(m.t)

    # active magnets
    m.magnetD       = pyo.Var(m.t, bounds=(w,  u_max))   

    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t)
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t)
    m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t)
    m.hammerA       = pyodae.DerivativeVar(m.hammerV, wrt=m.t)
    m.magnetV       = pyodae.DerivativeVar(m.magnetD, wrt=m.t)
   

    # Constraints on hammer displacement
    m.hammerDisp1  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] <= (m.magnetD[t] - w))
    m.hammerDisp2  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] >= (-m.magnetD[t] + w))

    # Constraints on robot velocity and acceleration
    m.baseVelConst1= pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] <= m.tf *  config['max_robot_vel'])
    m.baseVelConst2= pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] >= m.tf * -config['max_robot_vel'])
    m.baseAccConst1= pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] <= m.tf**2 *  config['max_robot_acc'])
    m.baseAccConst2= pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] >= m.tf**2 * -config['max_robot_acc'])

    # Constraints on the separation between magnets
    # m.mag_D1_D2    = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == - m.magnetD2[t]) 
    m.sep_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] + m.magnetD[t] - w)
    m.sep_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD[t] - m.hammerD[t] - w)
    m.vel_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV[t] <= m.tf * udot_max )
    m.vel_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV[t] >= m.tf * -udot_max)
    
    # Constraint on the hammer + base displacement before 0.8 * tf
    m.hammerBase1  = pyo.Constraint(m.t, rule=hammer_nail_constr)

    # The following constraint will restrict the hammer to return back to the position
    # m.hammerDisp2  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] + baseD[t] >= nail_pos + 0.01 for t >= alpha) 

    # System dynamics
    m.hammerAcc    = pyo.Constraint(m.t, rule=lambda m, t: m.hammerA[t] == - (M * m.baseA[t] + d * m.tf * m.hammerV[t] + C1 * m.tf**2 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M)

    if setting == 'high_stiffness':
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] ==  w)
    elif setting == 'low_stiffness' :
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] == u_max)

    # States at initial Time
    m.constr    = pyo.ConstraintList()
    m.constr.add(m.baseD[0] == start_pos)
    m.constr.add(m.baseV[0] == 0.0)
    m.constr.add(m.baseA[0] == 0.0)
    m.constr.add(m.hammerD[0] == 0.0)
    m.constr.add(m.hammerV[0] == 0.0)
    m.constr.add(m.hammerA[0] == 0.0)
    m.constr.add(m.magnetV[0] == 0.0)

    if (setting == 'high_stiffness') :
        m.constr.add(m.magnetD[0] ==  w)
    elif (setting == 'low_stiffness') :
        m.constr.add(m.magnetD[0] == u_max)
    elif (setting == 'variable_stiffness'):
        m.constr.add(m.magnetD[0] ==  w)


    # Intermediate time constraints
    # constraints to make sure the hammer hits the nail in the optimization problem
    m.constr.add(m.baseD[m.alpha]+m.hammerD[m.alpha] >= m.nail_pos)
    m.constr.add(m.baseD[m.alpha]+m.hammerD[m.alpha] <= m.nail_pos + pos_threshold)

    # Final time constraints
    m.constr.add(m.baseD[1.0] == end_pos)
    m.constr.add(m.baseV[1.0] == 0.0)

    # Objective
    m.obj       = pyo.Objective(expr=  - weights[0] * m.tf + weights[1] * (m.hammerV[m.alpha]) , sense = pyo.maximize)

    pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=500, scheme='BACKWARD')
    pyo.SolverFactory('ipopt').solve(m).write()

    # Extract the Results
    tf         = pyo.value(m.tf)
    t          = np.array([t * tf for t in m.t])
    baseD      = np.array([m.baseD[t]() for t in m.t])
    baseV      = np.array([m.baseV[t]() / tf for t in m.t])
    baseA      = np.array([m.baseA[t]() / tf**2 for t in m.t])

    hammerD    = np.array([m.hammerD[t]() for t in m.t])
    hammerV    = np.array([m.hammerV[t]() / tf for t in m.t])

    magnetD1   = np.array([-m.magnetD[t]() for t in m.t])
    magnetD2   = np.array([m.magnetD[t]() for t in m.t])

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


def user_provided_magnet_separation(tf, sep, config):
    """PYOMO model for the VSM with constant magnet separation provided by the user for a given tf
    The problem formulation is according to: 
    In upright position of the VSM: left --> Magnet1 and right --> Magnet2

    Arguments:
        tf {float} -- final time mentioned by the user
        sep {float} -- Separation of the magnet provided by the user in between 0.03 and 0.06
        config {yaml} -- configurations imported from the config.yaml file
    """
    # Robot parameters
    freq            = config['freq']

    # Model paramters
    C1              = config['C1']
    C2              = config['C2']
    M               = config['M']
    d               = config['d']
    w               = config['w']
    udot_max        = config['udot_max']    
    u_min           = config['no_mag_fixed'][0]
    u_max           = config['no_mag_fixed'][1]

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
    m.hammerD       = pyo.Var(m.t, initialize=0.0)
    m.Sep1          = pyo.Var(m.t)
    m.Sep2          = pyo.Var(m.t)

    # active magnets
    m.magnetD1      = pyo.Var(m.t, bounds=(u_min, -w))
    m.magnetD2      = pyo.Var(m.t, bounds=(    w,  u_max))
   

    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t, bounds=(-config['max_robot_vel'], config['max_robot_vel']))
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t, bounds=(-config['max_robot_acc'], config['max_robot_acc']))
    m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t)
    m.hammerA       = pyodae.DerivativeVar(m.hammerV, wrt=m.t)
    m.magnetV1      = pyodae.DerivativeVar(m.magnetD1, wrt=m.t, bounds=(-udot_max, udot_max))
    m.magnetV2      = pyodae.DerivativeVar(m.magnetD2, wrt=m.t, bounds=(-udot_max, udot_max))

    # Constraints on hammer movement
    m.hammerDisp1  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] <= (m.magnetD2[t] - w))
    m.hammerDisp2  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] >= (m.magnetD1[t] + w))

    # Constraints on the separation between magnets
    m.sep_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] - m.magnetD1[t] - w)
    m.sep_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD2[t] - m.hammerD[t] - w)
    
    # System dynamics
    m.hammerAcc    = pyo.Constraint(m.t, rule=lambda m, t: m.hammerA[t] == - (M * m.baseA[t] + d * m.hammerV[t] + C1 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M)

    m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == -sep)
    m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD2[t] ==  sep)
    

    # States at initial Time
    m.constr    = pyo.ConstraintList()
    m.constr.add(m.baseD[0] == start_pos)
    m.constr.add(m.baseV[0] == 0.0)
    m.constr.add(m.baseA[0] == 0.0)
    m.constr.add(m.hammerD[0] == 0.0)
    m.constr.add(m.hammerV[0] == 0.0)
    m.constr.add(m.hammerA[0] == 0.0)
    m.constr.add(m.magnetV1[0] == 0.0)
    m.constr.add(m.magnetV2[0] == 0.0)

    m.constr.add(m.magnetD1[0] == -sep)
    m.constr.add(m.magnetD2[0] ==  sep)


    # Intermediate time constraints
    # constraints to make sure the hammer hits the nail in the optimization problem
    m.constr.add(m.baseD[round(alpha*tf, 4)]+m.hammerD[round(alpha*tf, 4)] >= nail_pos + 0.01)
    m.constr.add(m.baseD[round(alpha*tf, 4)]+m.hammerD[round(alpha*tf, 4)] <= nail_pos + pos_threshold)

    # Final time constraints
    m.constr.add(m.baseD[tf] == end_pos)
    m.constr.add(m.baseV[tf] == 0.0)
    
    # Objective
    m.obj       = pyo.Objective(expr= (m.hammerV[round(alpha*tf, 4)]) , sense = pyo.maximize)

    pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=N, scheme='BACKWARD')
    pyo.SolverFactory('ipopt').solve(m).write()

    # Extract the Results
    t          = np.array([t for t in m.t])
    baseD      = np.array([m.baseD[t]() for t in m.t])
    baseV      = np.array([m.baseV[t]() for t in m.t])
    baseA      = np.array([m.baseA[t]() for t in m.t])

    hammerD    = np.array([m.hammerD[t]() for t in m.t])
    hammerV    = np.array([m.hammerV[t]() for t in m.t])

    magnetD1   = np.array([m.magnetD1[t]() for t in m.t])
    magnetD2   = np.array([m.magnetD2[t]() for t in m.t])
    magnetV1   = np.array([m.magnetV1[t]() for t in m.t])
    magnetV2   = np.array([m.magnetV2[t]() for t in m.t])
    
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

