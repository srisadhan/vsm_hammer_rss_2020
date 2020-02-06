from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import pyomo.environ as pyo
import pyomo.dae as pyodae


def one_fixed_magnet(tf, setting, magnet_cond, config):
    """PYOMO model for the VSM with one magnet fixed
    The problem formulation is according to: 
    In upright position of the VSM: left --> Magnet1 and right --> Magnet2

    Arguments:
        tf {float} -- final time mentioned by the user
        setting {string} -- Stiffness setting from one of the three options: 'high_stiffness','low_stiffness','variable_stiffness'
        magnet_cond {string} -- information about the fixed magnet, e.g. left_mag_fixed or right_mag_fixed 
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
    
    if (magnet_cond == 'left_mag_fixed') :
        u_min       = config['left_mag_fixed'][0]
        u_max       = config['left_mag_fixed'][1]
    elif (magnet_cond == 'right_mag_fixed'):
        u_min       = config['right_mag_fixed'][0]
        u_max       = config['right_mag_fixed'][1]

    # Selection of parameters
    start_pos       = config['start_pos']
    nail_pos        = config['nail_pos']
    end_pos         = config['end_pos']
    alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']

    # initialize the PYOMO model
    m               = pyo.ConcreteModel()

    # time variables
    tvec            = np.around(np.linspace(0, tf, freq + 1), decimals=4).tolist()
    m.t             = pyodae.ContinuousSet(bounds=(0,tf), initialize=tvec)

    # initialize variables
    m.baseD         = pyo.Var(m.t, bounds=(config['min_robot_disp'], nail_pos + pos_threshold))
    m.hammerD       = pyo.Var(m.t)
    m.Sep1          = pyo.Var(m.t)
    m.Sep2          = pyo.Var(m.t)

    # active magnets
    if (magnet_cond == 'left_mag_fixed'):
        m.magnetD1      = pyo.Var(m.t, bounds=(-u_max, -u_max))
        m.magnetD2      = pyo.Var(m.t, bounds=( u_min,  u_max))
    elif (magnet_cond == 'right_mag_fixed'):
        m.magnetD1      = pyo.Var(m.t, bounds=(u_min, u_max))
        m.magnetD2      = pyo.Var(m.t, bounds=(-u_min, -u_min))

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

    if (setting == 'high_stiffness') and (magnet_cond == 'left_mag_fixed'):
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD2[t] == u_min)
    elif (setting == 'low_stiffness') and (magnet_cond == 'left_mag_fixed'):
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD2[t] == u_max)
    elif (setting == 'high_stiffness') and (magnet_cond == 'right_mag_fixed'):
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == u_max)
    elif (setting == 'low_stiffness') and (magnet_cond == 'right_mag_fixed'):
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == u_min)

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

    if (setting == 'high_stiffness') and (magnet_cond == 'left_mag_fixed'):
        m.constr.add(m.magnetD2[0] == u_min)
    elif (setting == 'low_stiffness') and (magnet_cond == 'left_mag_fixed'):
        m.constr.add(m.magnetD2[0] == u_max)
    elif (setting == 'high_stiffness') and (magnet_cond == 'right_mag_fixed'):
        m.constr.add(m.magnetD1[0] == u_max)
    elif (setting == 'low_stiffness') and (magnet_cond == 'right_mag_fixed'):
        m.constr.add(m.magnetD1[0] == u_min)

    # Intermediate time constraints
    # constraints to make sure the hammer hits the nail in the optimization problem
    m.constr.add(m.baseD[round(alpha*100*tf)/100]+m.hammerD[round(alpha*100*tf)/100] >= nail_pos + 0.01)
    m.constr.add(m.baseD[round(alpha*100*tf)/100]+m.hammerD[round(alpha*100*tf)/100] <= nail_pos + pos_threshold)

    # Final time constraints
    m.constr.add(m.baseD[tf] == end_pos)
    m.constr.add(m.baseV[tf] == 0.0)

    # Objective
    m.obj       = pyo.Objective(expr= (m.hammerV[round(alpha*100*tf)/100]), sense = pyo.maximize)

    pyo.TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=500, scheme='BACKWARD')
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
    df = pd.DataFrame({'t' : t,
                        'bd'  : baseD,
                        'bv'  : baseV,
                        'hd'  : hammerD,
                        'hv'  : hammerV,
                        'md1' : magnetD1,
                        'md2' : magnetD2,
                        })
    
    return df