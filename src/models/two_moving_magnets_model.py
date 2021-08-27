from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections 
import pandas as pd

import pyomo.environ as pyo
import pyomo.dae as pyodae
from pyomo.opt.parallel import SolverManagerFactory

def trajectory_optimization(tf, setting, config):
    """PYOMO model for the VSM with two moving magnets controlled symmetrically in opposite direction by a single actuator
    The problem formulation is according to: 
    In upright position of the VSM: left --> Magnet1 and right --> Magnet2

    Arguments:
        tf {float} -- final time mentioned by the user
        setting {string} -- Stiffness setting from one of the three options: 'high_stiffness','low_stiffness','variable_stiffness'
        config {yaml} -- configurations imported from the config.yaml file
    """
    def traj_constr1(m, t):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail"""
        if t < config['alpha'] * (m.tf - 0.1):
            return m.hammerD[t] + m.baseD[t] <= config['nail_pos'] - 0.005
        elif t > config['alpha'] * m.tf :
            # constraints can be expressed in the form (lb, expr, ub)
            return ( config['nail_pos'], m.hammerD[t] + m.baseD[t], config['nail_pos'] + config['pos_threshold'])
        else:
            return pyo.Constraint.Skip
    
    # def traj_constr2(m, t):
    #     """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail"""
    #     if t > config['alpha'] * m.tf:
    #         return m.hammerD[t] + m.baseD[t] <= config['nail_pos'] + config['pos_threshold'] #-0.015
    #     else:
    #         return pyo.Constraint.Skip
    
    def obj_function(m):
        """ Obj function for multiobjective optimization - w1 * f1^2 + w2 * f2^2."""
        # return ((m.hammerV[round(config['alpha'] * tf, 4)] - hammer_vel_range[0]) / (hammer_vel_range[1] - hammer_vel_range[0]))**2
        return (m.hammerV[round(config['alpha'] * tf, 4)] + m.baseV[round(config['alpha'] * tf, 4)])**2
    
    # Robot parameters
    freq            = config['freq']

    # Model paramters
    C1              = config['C1']
    C2              = config['C2']
    M               = config['M']
    d               = config['d']
    mu              = config['mu']
    w               = config['w']
    udot_max        = config['udot_max']    
    u_min           = config['no_mag_fixed'][0]
    u_max           = config['no_mag_fixed'][1]
    
    hammer_vel_range = config[setting]['vel']

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
    m.hammerD       = pyo.Var(m.t, initialize=0.0)
    m.Sep1          = pyo.Var(m.t)
    m.Sep2          = pyo.Var(m.t)

    # active magnets
    # m.magnetD1      = pyo.Var(m.t, bounds=(u_min, -w))
    m.magnetD       = pyo.Var(m.t, bounds=(w,  u_max))
   
    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t, bounds=(-config['max_robot_vel'], config['max_robot_vel']))
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t, bounds=(-config['max_robot_acc'], config['max_robot_acc']))
    m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t)
    m.hammerA       = pyodae.DerivativeVar(m.hammerV, wrt=m.t)
    # m.magnetV1      = pyodae.DerivativeVar(m.magnetD1, wrt=m.t, bounds=(-udot_max, udot_max))
    m.magnetV       = pyodae.DerivativeVar(m.magnetD, wrt=m.t, bounds=(-udot_max, udot_max))

    # Constraints on hammer movement
    m.hammerDisp1  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] <= (m.magnetD[t] - w))
    m.hammerDisp2  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] >= (-m.magnetD[t] + w))

    # Constraints on the separation between magnets
    m.sep_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] + m.magnetD[t] - w)
    m.sep_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD[t] - m.hammerD[t] - w)
    
    # System dynamics
    m.hammerAcc    = pyo.Constraint(m.t, rule=lambda m, t: m.hammerA[t] == - (M * m.baseA[t] + d * m.hammerV[t] + C1 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M)

    if setting == 'high_stiffness':
        # m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == -w)
        m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] ==  w)
    elif setting == 'low_stiffness' :
        # m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == u_min)
        m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] == u_max)
    

    # States at initial Time
    # m.constr    = pyo.ConstraintList()
    m.baseD[0].fix(start_pos)
    m.baseV[0].fix(0.0)
    m.baseA[0].fix(0.0)
    m.hammerD[0].fix(0.0)
    m.hammerV[0].fix(0.0)
    m.hammerA[0].fix(0.0)
    # m.magnetV1[0].fix(0.0)
    m.magnetV[0].fix(0.0)
    m.magnetD[0].fix(config[setting]['start'])


    # Intermediate time constraints
    m.NailHit1    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr1(m, t))
    # m.NailHit2    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr2(m, t))

    # Final time constraints
    m.baseD[tf].fix(end_pos)
    m.baseV[tf].fix(0.0)
    
    # Objective
    m.obj       = pyo.Objective(expr= obj_function(m) , sense = pyo.maximize)

    return m


def optimize_tf_and_vel(setting, weights, config):
    """PYOMO model for the VSM with two moving magnets controlled symmetrically in opposite direction by a single actuator
    The problem formulation is according to: 
    In upright position (hammer facing up) of the VSM: left --> Magnet1 and right --> Magnet2

    Arguments:
        setting {string} -- Stiffness setting from one of the three options: 'high_stiffness','low_stiffness','variable_stiffness'
        weights {list}   -- list of weights for multi-objective optimization: weights[0] to tf and weights[1] to hammer velocity
        config  {yaml}   -- configurations imported from the config.yaml file
    
    Returns:
        m {pyomo model}  -- The optimization problem formulated using a Pyomo model
    """

    def traj_constr1(m, t):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail"""
        if t < config['alpha']:
            return m.hammerD[t] + m.baseD[t] <= config['nail_pos'] - 0.005
        elif t > config['alpha']:
            # constraints can be expressed in the form (lb, expr, ub)
            return ( config['nail_pos'], m.hammerD[t] + m.baseD[t], config['nail_pos'] + config['pos_threshold'])
        else:
            return pyo.Constraint.Skip

    def obj_function(m):
        """ Obj function for multiobjective optimization - w1 * f1^2 + w2 * f2^2."""
        print(weights, hammer_vel_range)
        print(pyo.value((m.tf - config['tf_range'][0]) / (config['tf_range'][1] - config['tf_range'][0])), pyo.value((m.hammerV[alpha] - hammer_vel_range[0]) / (hammer_vel_range[1] - hammer_vel_range[0])))
        # return - weights[0] * ((m.tf - config['tf_range'][0]) / (config['tf_range'][1] - config['tf_range'][0]))**2
        # + weights[1] * ((m.hammerV[alpha] + m.baseV[alpha] - hammer_vel_range[0]) / (hammer_vel_range[1] - hammer_vel_range[0]))**2

        return -(m.tf )**2 + (m.hammerV[alpha]+m.baseV[alpha])**2


    # Robot parameters
    freq            = config['freq']

    # Model paramters
    C1              = config['C1']
    C2              = config['C2']
    M               = config['M']
    d               = config['d']
    mu              = config['mu']
    w               = config['w']
    udot_max        = config['udot_max']    
    u_min           = config['no_mag_fixed'][0]
    u_max           = config['no_mag_fixed'][1]
    
    hammer_vel_range = config[setting]['vel']

    # Selection of parameters
    start_pos       = config['start_pos']
    nail_pos        = config['nail_pos']
    end_pos         = config['end_pos']
    alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']

    # initialize the PYOMO model
    m               = pyo.ConcreteModel()

    # Scaled time t in the range [0, 1]
    m.tf            = pyo.Var(within=pyo.NonNegativeReals, bounds=(config['tf_range'][0], config['tf_range'][1]), initialize=1.0)
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
    m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t, initialize=0)
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
    m.sep_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] + m.magnetD[t] - w)
    m.sep_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD[t] - m.hammerD[t] - w)
    m.vel_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV[t] <= m.tf * udot_max )
    m.vel_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV[t] >= m.tf * -udot_max)
    
    # System dynamics
    m.hammerAcc    = pyo.Constraint(m.t, rule=lambda m, t: m.hammerA[t] == - (M * m.baseA[t] + d * m.tf * m.hammerV[t] + C1 * m.tf**2 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M)

    if setting == 'high_stiffness':
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] == w)
    elif setting == 'low_stiffness' :
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] == u_max)

    # States at initial Time
    # m.constr    = pyo.ConstraintList()
    m.baseD[0].fix(start_pos)
    m.baseV[0].fix(0.0)
    m.baseA[0].fix(0.0)
    m.hammerD[0].fix(0.0)
    m.hammerV[0].fix(0.0)
    m.hammerA[0].fix(0.0)
    m.magnetV[0].fix(0.0)
    m.magnetD[0].fix(config[setting]['start'])


    # Intermediate time constraints
    m.NailHit1    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr1(m, t))

    # Final time constraints
    m.baseD[1.0].fix(end_pos)
    m.baseV[1.0].fix(0.0)

    # Objective
    m.obj       = pyo.Objective(expr= obj_function(m) , sense = pyo.maximize)

    return m

#Change of constraints 
# FIXME: Dont use this for now
def optimize_tf_New(alpha, setting, weights, config):
    """PYOMO model for the VSM with two moving magnets controlled symmetrically in opposite direction by a single actuator
    The problem formulation is according to: 
    In upright position of the VSM: left --> Magnet1 and right --> Magnet2

    Arguments:
        alpha {float}    -- Value between (0,1) which decides the time at which the hammer hits the nail.
        setting {string} -- Stiffness setting from one of the three options: 'high_stiffness','low_stiffness','variable_stiffness'
        weights {list}   -- list of weights for multi-objective optimizing: weights[0] to tf and weights[1] to hammer velocity
        config {yaml}    -- configurations imported from the config.yaml file
    """
    def hammer_nail_constr(m, t):
        if  t > m.alpha - 0.2:
            return pyo.Constraint.Skip
        return m.hammerD[t] + m.baseD[t] <= m.nail_pos

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
    m.baseD[0].fix(start_pos)
    m.baseV[0].fix(0.0)
    m.baseA[0].fix(0.0)
    m.hammerD[0].fix(0.0)
    m.hammerV[0].fix(0.0)
    m.hammerA[0].fix(0.0)
    m.magnetV[0].fix(0.0)

    if (setting == 'high_stiffness') :
        m.magnetD[0].fix(w)
    elif (setting == 'low_stiffness') :
        m.magnetD[0].fix(u_max)
    elif (setting == 'variable_stiffness'):
        m.magnetD[0].fix(w)


    # Intermediate time constraints
    # constraints to make sure the hammer hits the nail in the optimization problem
    m.constr.add(m.baseD[m.alpha]+m.hammerD[m.alpha] >= m.nail_pos)
    m.constr.add(m.baseD[m.alpha]+m.hammerD[m.alpha] <= m.nail_pos + pos_threshold)

    # Final time constraints
    m.baseD[1.0].fix(end_pos)
    m.baseV[1.0].fix(0.0)

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
    def traj_constr1(m, t):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail"""
        if t < config['alpha'] * tf:
            return m.hammerD[t] + m.baseD[t] <= config['nail_pos'] - 0.005
        elif t > config['alpha'] * tf:
            return m.hammerD[t] + m.baseD[t] >= config['nail_pos']
        else:
            return pyo.Constraint.Skip

    def traj_constr2(m, t):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail"""
        if t > config['alpha'] * tf:
            return m.hammerD[t] + m.baseD[t] <= config['nail_pos'] + config['pos_threshold']
        else:
            return pyo.Constraint.Skip
    
    def obj_function(m):
        """ Obj function for multiobjective optimization - w1 * f1^2 + w2 * f2^2."""
        # return ((m.hammerV[round(config['alpha'] * tf, 4)] - hammer_vel_range[0]) / (hammer_vel_range[1] - hammer_vel_range[0]))**2
        return (m.hammerV[round(alpha * tf, 4)] + m.baseV[round(alpha * tf, 4)])**2
    
    # Robot parameters
    freq            = config['freq']

    # Model paramters
    C1              = config['C1']
    C2              = config['C2']
    M               = config['M']
    d               = config['d']
    mu              = config['mu']
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
    m.tf            = pyo.Param(initialize=tf)

    # time variables
    if tf < 1.0 :
        N   = freq
    else:
        N   = int(round(tf * (freq)))

    tvec            = np.around(np.linspace(0, tf, N+1), decimals=4).tolist()
    m.t             = pyodae.ContinuousSet(bounds=(0,tf), initialize=tvec)

    # initialize variables
    m.baseD         = pyo.Var(m.t, bounds=(config['min_robot_disp'], nail_pos + pos_threshold), initialize=0.0)
    m.hammerD       = pyo.Var(m.t, initialize=0.0)
    m.Sep1          = pyo.Var(m.t)
    m.Sep2          = pyo.Var(m.t)

    # active magnets
    # m.magnetD1      = pyo.Var(m.t, bounds=(u_min, -w))
    m.magnetD       = pyo.Var(m.t, bounds=(w,  u_max))
   

    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t, bounds=(-config['max_robot_vel'], config['max_robot_vel']))
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t, bounds=(-config['max_robot_acc'], config['max_robot_acc']))
    m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t)
    m.hammerA       = pyodae.DerivativeVar(m.hammerV, wrt=m.t)
    m.magnetV       = pyodae.DerivativeVar(m.magnetD, wrt=m.t, bounds=(-udot_max, udot_max))
    # m.magnetV2      = pyodae.DerivativeVar(m.magnetD2, wrt=m.t, bounds=(-udot_max, udot_max))

    # Constraints on hammer movement
    m.hammerDisp1  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] <= (m.magnetD[t] - w))
    m.hammerDisp2  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] >= (-m.magnetD[t] + w))

    # Constraints on the separation between magnets
    m.sep_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] + m.magnetD[t] - w)
    m.sep_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD[t] - m.hammerD[t] - w)
    
    # System dynamics
    m.hammerAcc    = pyo.Constraint(m.t, rule=lambda m, t: m.hammerA[t] == - (M * m.baseA[t] + d * m.hammerV[t] + C1 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M)

    # m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == -sep)
    m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] ==  sep)
    

    # States at initial Time
    # m.constr    = pyo.ConstraintList()
    m.baseD[0].fix(start_pos)
    m.baseV[0].fix(0.0)
    m.baseA[0].fix(0.0)
    m.hammerD[0].fix(0.0)
    m.hammerV[0].fix(0.0)
    m.hammerA[0].fix(0.0)
    # m.magnetV1[0].fix(0.0)
    m.magnetV[0].fix(0.0)

    # m.magnetD1[0].fix(-sep)
    m.magnetD[0].fix(sep)


    # Intermediate time constraints
    m.NailHit1    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr1(m, t))
    m.NailHit2    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr2(m, t))

    # Final time constraints
    m.baseD[tf].fix(end_pos)
    m.baseV[tf].fix(0.0)
        
    # Objective
    m.obj       = pyo.Objective(expr= obj_function(m), sense = pyo.maximize)
    
    return m

# Function utilizing the impulse in the objective
def maximize_vel_minimize_impulse(tf, setting, config):
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
    
    hammer_vel_range = config[setting]['vel']

    # Selection of parameters
    start_pos       = config['start_pos']
    nail_pos        = config['nail_pos']
    end_pos         = config['end_pos']
    alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']
    
    def traj_constr1(m, t):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail"""
        if t < config['alpha'] * (m.tf - 0.1):
            return m.hammerD[t] + m.baseD[t] <= config['nail_pos'] - 0.005
        elif t > config['alpha'] * m.tf :
            # constraints can be expressed in the form (lb, expr, ub)
            return ( config['nail_pos'], m.hammerD[t] + m.baseD[t], config['nail_pos'] + config['pos_threshold'])
        else:
            return pyo.Constraint.Skip
    
    def impulse_constr(m, t):
        """ Imposes impulse constraint on the dynamics of the system """
        if (t > config['alpha'] * m.tf) and (t < config['alpha'] * (m.tf + 0.01)):
            return m.hammerA[t] == - (M * m.baseA[t] + d * m.hammerV[t] + C1 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M
        else:
            return m.hammerA[t] == - (M * m.baseA[t] + d * m.hammerV[t] + C1 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M + 10

        
    def obj_function(m):
        """ Obj function for multiobjective optimization - w1 * f1^2 + w2 * f2^2."""
        return (m.hammerV[round(config['alpha'] * tf, 4)] + m.baseV[round(config['alpha'] * tf, 4)])**2 
    
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
    m.hammerD       = pyo.Var(m.t, initialize=0.0)
    m.Sep1          = pyo.Var(m.t)
    m.Sep2          = pyo.Var(m.t)

    # active magnets
    m.magnetD       = pyo.Var(m.t, bounds=(w,  u_max))
   
    # Derivatives of the variables
    m.baseV         = pyodae.DerivativeVar(m.baseD, wrt=m.t, bounds=(-config['max_robot_vel'], config['max_robot_vel']))
    m.baseA         = pyodae.DerivativeVar(m.baseV, wrt=m.t, bounds=(-config['max_robot_acc'], config['max_robot_acc']))
    m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t)
    m.hammerA       = pyodae.DerivativeVar(m.hammerV, wrt=m.t)
    m.magnetV       = pyodae.DerivativeVar(m.magnetD, wrt=m.t, bounds=(-udot_max, udot_max))

    # Constraints on hammer movement
    m.hammerDisp1  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] <= (m.magnetD[t] - w))
    m.hammerDisp2  = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] >= (-m.magnetD[t] + w))

    # Constraints on the separation between magnets
    m.sep_ham_mag1 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] + m.magnetD[t] - w)
    m.sep_ham_mag2 = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD[t] - m.hammerD[t] - w)
    
    # System dynamics
    m.hammerAcc    = pyo.Constraint(m.t, rule=lambda m, t: impulse_constr(m, t))

    if setting == 'high_stiffness':
        # m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == -w)
        m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] ==  w)
    elif setting == 'low_stiffness' :
        # m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD1[t] == u_min)
        m.magdisp4  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] == u_max)
    

    # States at initial Time
    m.constr    = pyo.ConstraintList()
    m.baseD[0].fix(start_pos)
    m.baseV[0].fix(0.0)
    m.baseA[0].fix(0.0)
    m.hammerD[0].fix(0.0)
    m.hammerV[0].fix(0.0)
    m.hammerA[0].fix(0.0)
    # m.magnetV1[0].fix(0.0)
    m.magnetV[0].fix(0.0)
    m.magnetD[0].fix(config[setting]['start'])


    # Intermediate time constraints
    m.NailHit1    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr1(m, t))
    
    # Final time constraints
    m.baseD[tf].fix(end_pos)
    m.baseV[tf].fix(0.0)
    
    # Objective
    m.obj       = pyo.Objective(expr= obj_function(m), sense = pyo.maximize)

    return m

def maximize_vel_minimize_time_impulse(setting, weights, config):
    """PYOMO model for the VSM with two moving magnets controlled symmetrically in opposite direction by a single actuator
    
    The problem formulation is for maximizing the impact on nail and minimizing the impact on the robot joints:
    In upright position (hammer facing up) of the VSM: left --> Magnet1 and right --> Magnet2

    Arguments:
        setting {string} -- Stiffness setting from one of the three options: 'high_stiffness','low_stiffness','variable_stiffness'
        weights {list}   -- list of weights for multi-objective optimization: weights[0] to tf and weights[1] to hammer velocity
        config  {yaml}   -- configurations imported from the config.yaml file
    
    Returns:
        m {pyomo model}  -- The optimization problem formulated using a Pyomo model
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
    
    hammer_vel_range = config[setting]['vel']

    # Selection of parameters
    start_pos       = config['start_pos']
    nail_pos        = config['nail_pos']
    end_pos         = config['end_pos']
    alpha           = config['alpha']
    pos_threshold   = config['pos_threshold']


    def traj_constr1(m, t):
        """Constraints on the absolute displacement of the hammer to make sure that the hammer hits the nail"""
        if t < config['alpha']:
            return m.hammerD[t] + m.baseD[t] <= config['nail_pos'] - 0.005
        elif t > config['alpha']:
            # constraints can be expressed in the form (lb, expr, ub)
            return ( config['nail_pos'], m.hammerD[t] + m.baseD[t], config['nail_pos'] + config['pos_threshold'])
        else:
            return pyo.Constraint.Skip

    def obj_function(m):
        """ Obj function for multiobjective optimization - w1 * f1^2 + w2 * f2^2."""
        print(weights, hammer_vel_range)
        print(pyo.value((m.tf - config['tf_range'][0]) / (config['tf_range'][1] - config['tf_range'][0])), pyo.value((m.hammerV[alpha] - hammer_vel_range[0]) / (hammer_vel_range[1] - hammer_vel_range[0])))
        # return - weights[0] * ((m.tf - config['tf_range'][0]) / (config['tf_range'][1] - config['tf_range'][0]))**2
        # + weights[1] * ((m.hammerV[alpha] + m.baseV[alpha] - hammer_vel_range[0]) / (hammer_vel_range[1] - hammer_vel_range[0]))**2
        return -(m.tf )**2 + (M * (m.hammerV[alpha] + m.baseV[alpha]))**2 #- (m.impulseR)**2

    # initialize the PYOMO model
    m               = pyo.ConcreteModel()

    # Scaled time t in the range [0, 1]
    m.tf            = pyo.Var(within=pyo.NonNegativeReals, bounds=(config['tf_range'][0], config['tf_range'][1]), initialize=1.0)
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
    m.hammerV       = pyodae.DerivativeVar(m.hammerD, wrt=m.t, initialize=0)
    m.hammerA       = pyodae.DerivativeVar(m.hammerV, wrt=m.t)
    m.magnetV       = pyodae.DerivativeVar(m.magnetD, wrt=m.t)   

    # Integral of force acting on the robot
    m.impulseR      = pyodae.Integral(m.t, wrt=m.t, rule=lambda m, t: C1 * m.tf**2 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t])))
    
    # Constraints on hammer displacement
    m.hammerDisp1   = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] <= (m.magnetD[t] - w))
    m.hammerDisp2   = pyo.Constraint(m.t, rule=lambda m, t: m.hammerD[t] >= (-m.magnetD[t] + w))

    # Constraints on robot velocity and acceleration
    m.baseVelConst1 = pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] <= m.tf *  config['max_robot_vel'])
    m.baseVelConst2 = pyo.Constraint(m.t, rule=lambda m, t: m.baseV[t] >= m.tf * -config['max_robot_vel'])
    m.baseAccConst1 = pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] <= m.tf**2 *  config['max_robot_acc'])
    m.baseAccConst2 = pyo.Constraint(m.t, rule=lambda m, t: m.baseA[t] >= m.tf**2 * -config['max_robot_acc'])

    # Constraints on the separation between magnets
    m.sep_ham_mag1  = pyo.Constraint(m.t, rule=lambda m, t: m.Sep1[t] == m.hammerD[t] + m.magnetD[t] - w)
    m.sep_ham_mag2  = pyo.Constraint(m.t, rule=lambda m, t: m.Sep2[t] == m.magnetD[t] - m.hammerD[t] - w)
    m.vel_ham_mag1  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV[t] <= m.tf * udot_max )
    m.vel_ham_mag2  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetV[t] >= m.tf * -udot_max)
    
    # System dynamics
    m.hammerAcc     = pyo.Constraint(m.t, rule=lambda m, t: m.hammerA[t] == - (M * m.baseA[t] + d * m.tf * m.hammerV[t] + C1 * m.tf**2 * (pyo.exp(-C2 * m.Sep2[t]) - pyo.exp(-C2 * m.Sep1[t]))) / M)

    if setting == 'high_stiffness':
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] == w)
    elif setting == 'low_stiffness' :
        m.magdisp3  = pyo.Constraint(m.t, rule=lambda m, t: m.magnetD[t] == u_max)

    # States at initial Time
    # m.constr    = pyo.ConstraintList()
    m.baseD[0].fix(start_pos)
    m.baseV[0].fix(0.0)
    m.baseA[0].fix(0.0)
    m.hammerD[0].fix(0.0)
    m.hammerV[0].fix(0.0)
    m.hammerA[0].fix(0.0)
    m.magnetV[0].fix(0.0)
    m.magnetD[0].fix(config[setting]['start'])


    # Intermediate time constraints
    m.NailHit1    = pyo.Constraint(m.t, rule=lambda m, t: traj_constr1(m, t))

    # Final time constraints
    m.baseD[1.0].fix(end_pos)
    m.baseV[1.0].fix(0.0)

    # Objective
    m.obj       = pyo.Objective(expr= obj_function(m) , sense = pyo.maximize)

    return m
