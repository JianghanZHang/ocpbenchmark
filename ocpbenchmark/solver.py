import mim_solvers
from utils import load_config_file
def make_solver(problem, solver_config):
    config = load_config_file(solver_config)
    if config['solver'] == 'csqp':
        solver = mim_solvers.SolverCSQP(problem)
        solver.th_stop = config['termination_tolerance']
        solver.eps_abs = config['eps_abs']
        solver.eps_rel = config['eps_rel']
        solver.max_iter = config['max_iter']
        solver.use_filter_line_search = config['use_filter_line_search'] 
        solver.setCallbacks()
        
    else:
        raise ValueError(f"Solver '{config['solver']}' not implemented.")