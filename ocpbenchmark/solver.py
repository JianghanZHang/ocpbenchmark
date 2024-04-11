import mim_solvers
from utils import load_config_file
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
        solver.th_stop = config['termination_tolerance']
        solver.eps_abs = config['eps_abs']
        solver.eps_rel = config['eps_rel']
        solver.use_filter_line_search = config['use_filter_line_search'] 
        solver.setCallbacks()
        
    else:
        raise ValueError(f"Solver '{config['solver']}' not implemented.")
    
    return solver, config['max_iter']