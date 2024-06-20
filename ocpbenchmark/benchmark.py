import crocoddyl
import problem
import plotter
import utils
import solver
from callback import BenchmarkLogger
import mim_solvers
from examples.quadruped_hard_constraints import plotSolution


def make(problem_config, solver_config, data_config):
    benchmark = Benchmark()
    benchmark.set_problem(problem_config)
    benchmark.set_solver(solver_config)
    benchmark.set_callback(data_config)
    return benchmark

class Benchmark:
    def __init__(self):
        self.problem = None
        self.solver = None
        self.data = None
        self.xs_init = None
        self.us_init = None  
        self.max_iter = None

        
    def set_problem(self, problem_config):
        self.problem = problem.make_problem(problem_config) 

    def set_solver(self, solver_config):
        if self.problem is None:
            raise ValueError("Problem is not set. Please set the problem first.")
        self.solver, self.max_iter, self.xs_init, self.us_init = solver.make_solver(self.problem, solver_config) 
    
    def set_callback(self, data_config):
        if self.solver is None:
            raise ValueError("Solver is not set. Please set the solver first.")
        benchmark_callback = BenchmarkLogger(data_config)
        self.data = benchmark_callback.data
        self.solver.setCallbacks(
            [benchmark_callback])

    def run(self):
        if self.solver is None:
            raise ValueError("Solver is not set. Please set the solver first.")
        if self.problem is None:
            raise ValueError("Problem is not set. Please set the problem first.")
        self.solver.solve(self.xs_init, self.us_init, self.max_iter)
        
        return self.data
