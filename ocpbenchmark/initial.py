import problem
import crocoddyl
import numpy as np
import mim_solvers

#set seed for reproducibility
np.random.seed(0)

def make_initial(problem, initial_configs):
    """
    input:
        problem: crocoddyl.ShootingProblem
            The shooting problem instance.
        initial_configs: dict
            The initial configuration dictionary.
    output:
        xs_init: initial guess of the state trajectory, xs_init[0] = x0
        us_init: initial guess of the control trajectory.
    """
    T = problem.T

    if initial_configs['type'] == 'quasi-static':
        xs_init = np.ones((T+1, problem.ndx)) * problem.x0
        us_init = problem.quasiStatic(xs_init)
    
    elif initial_configs['type'] == 'random':
        xs_init = np.random.uniform(initial_configs['state_lb'], initial_configs['state_ub'], (T+1, problem.ndx))
        us_init = np.random.uniform(initial_configs['control_lb'], initial_configs['control_ub'], (T, problem.nu))

    elif initial_configs['type'] == 'warm-start':
        xs_init = np.ones((T+1, problem.ndx)) * problem.x0
        us_init = problem.quasiStatic(xs_init)
        
        solver = mim_solvers.SolverCSQP(problem)
        solver.solve(xs_init, us_init, 10000)
        xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
        xs_init[0] = problem.x0
        us_init = list(solver.us[1:]) + [solver.us[-1]] 

    return list(xs_init), list(us_init)