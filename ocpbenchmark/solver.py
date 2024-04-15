import mim_solvers
from utils import load_config_file
import numpy as np
def make_solver(problem, solver_config):
    """
    input:
        problem: crocoddyl.ShootingProblem
            The shooting problem instance.
        solver_config: str
            The name of the solver configuration file.
    output:
        solver: crocoddyl.SolverAbstract
            The solver instance.
        max_iter: int
            The maximum number of iterations for the solver.
    """
    config = load_config_file(solver_config)
    if config['solver'] == 'csqp':
        solver = mim_solvers.SolverCSQP(problem)
        solver.th_stop = config['termination_tolerance'] # SQP termination tolerance
        
        # QP termination tolerance
        solver.eps_abs = config['eps_abs']
        solver.eps_rel = config['eps_rel']
        
        solver.use_filter_line_search = config['use_filter_line_search'] 
        solver.setCallbacks()
        
    else:
        raise ValueError(f"Solver '{config['solver']}' not implemented.")
    
    T = problem.T
    if config['initial_guess'] == 'quasi-static':
        xs_init = np.ones((T+1, problem.ndx)) * problem.x0
        us_init = problem.quasiStatic(xs_init)

    elif config['initial_guess'] == 'random':
        xs_init = np.random.uniform(config['state_lb'], config['state_ub'], (T+1, problem.ndx))
        us_init = np.random.uniform(config['control_lb'], config['control_ub'], (T, problem.nu))

    elif config['initial_guess'] == 'warm-start':
        
        xs_init = np.ones((T+1, problem.ndx)) * problem.x0
        us_init = problem.quasiStatic(xs_init)
        
        solver = mim_solvers.SolverCSQP(problem)
        solver.solve(xs_init, us_init, 10000)
        xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
        xs_init[0] = problem.x0
        us_init = list(solver.us[1:]) + [solver.us[-1]] 
    else:
        raise ValueError(f"Initial guess '{config['type']}' not implemented.")

    
    return solver, config['max_iter'], list(xs_init), list(us_init)