import pyomo.environ as pyo
import pyomo.dae as pyodae
from pyomo.opt.parallel import SolverManagerFactory

# def pyomo_classical_solver(model, solver, config):
#     """Solve the continuous NLP by discretizing it using backward finite difference and pyomo solvers
    
#     Arguments:
#         model {pyomo model} -- a concrete model build using Pyomo 
#         solver {string}     -- Solver used for the optimization problem. For e.g. ipopt
#         config{yaml}        -- configurations of the setting
#     """
#     # time variables
#     if model.tf < 1.0 :
#         N   = config['freq']
#     else:
#         N   = int(round(model.tf * config['freq']))

#     pyo.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=N, scheme='BACKWARD')
#     pyo.SolverFactory(solver).solve(model).write()

#     return model

def pyomo_solver(model, solver, config, neos=False):
    """Solve the continuous NLP by discretizing it using backward finite difference and pyomo solvers
    
    Arguments:
        model {pyomo model} -- a concrete model build using Pyomo
        solver {string}     -- Solver used for the optimization problem. For e.g. ipopt
        config{yaml}        -- configurations of the setting
        neos  {bool}        -- Use the neos solver if True
    """

    # time variables
    if model.tf < 1.0 :
        N   = config['freq']
    else:
        N   = int(round(model.tf * config['freq']))

    # finite difference discretization
    # pyo.TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=N, scheme='BACKWARD')
    
    # direct collocation 
    pyo.TransformationFactory('dae.collocation').apply_to(model, wrt=model.t, nfe=N, ncp=6)
    # if a particular variable has to be piecewise constant then uncomment the following and replace the var parameter in 'reduce_collocation_points' 
    # pyo.TransformationFactory('dae.collocation').apply_to(model, wrt=model.t, nf=10, ncp=6).reduce_collocation_points(model, var=model.u, ncp=1, contset=model.t)

    if neos:
        print("Using NEOS with Pyomo")
        SolverManagerFactory('neos').solve(model, opt=solver).write()
    else:
        pyo.SolverFactory(solver).solve(model).write()

    return model