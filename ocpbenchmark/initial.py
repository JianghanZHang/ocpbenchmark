import problem
import crocoddyl
import numpy as np

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
        xs_init = np.ones((T+1, problem.ndx))
        us_init = problem.quasiStatic(xs_init)
    
    elif initial_configs['type'] == 'random':

    return xs_init, us_init