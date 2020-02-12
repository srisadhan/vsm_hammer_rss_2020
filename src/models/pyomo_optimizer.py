import pyomo.environ as pyo
import pyomo.dae as pyodae
from pyomo.opt.parallel import SolverManagerFactory

def pyomo_classical_solver(model, solver, config):
    """Solve the continuous NLP by discretizing it using backward finite difference and pyomo solvers
    
    Arguments:
        model {pyomo model} -- a concrete model build using Pyomo 
        solver {string}     -- Solver used for the optimization problem. For e.g. ipopt
        config{yaml}        -- configurations of the setting
    """
    # time variables
    if model.tf < 1.0 :
        N   = config['freq']
    else:
        N   = int(round(model.tf * config['freq']))

    pyo.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=N, scheme='BACKWARD')
    pyo.SolverFactory(solver).solve(model).write()

    return model

def pyomo_neos_solver(model, solver, config):
    """Solve the continuous NLP by discretizing it using backward finite difference and pyomo solvers
    
    Arguments:
        model {pyomo model} -- a concrete model build using Pyomo
        solver {string}     -- Solver used for the optimization problem. For e.g. ipopt
        config{yaml}        -- configurations of the setting
    """

    # time variables
    if model.tf < 1.0 :
        N   = config['freq']
    else:
        N   = int(round(model.tf * config['freq']))

    pyo.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=N, scheme='BACKWARD')
    SolverManagerFactory('neos').solve(model, opt=solver).write()

    return model