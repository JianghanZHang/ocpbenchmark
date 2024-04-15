import crocoddyl
import problem
import plotter
import utils
import solver
from callback import benchmark_callback

class benchmark:
    def __init__(self):
        self.problem = None
        self.solver = None
        self.plotter = None
        self.data = None
        self.initial = None

    @staticmethod
    def make(problem_config, plotter_config, solver_config, data_config):
        benchmark = benchmark()
        benchmark.set_problem(problem_config)
        benchmark.set_solver(solver_config)
        benchmark.set_plotter(plotter_config)
        benchmark.set_callback(data_config)
        return benchmark
        
    def set_plotter(self, plotter_config):
        self.plotter = plotter(plotter_config)

    def set_problem(self, problem_config):
        self.problem = problem.make_problem(problem_config) 

    def set_solver(self, solver_config):
        if self.problem is None:
            raise ValueError("Problem is not set. Please set the problem first.")
        self.solver = solver.make_solver(self.problem, solver_config) 

    def set_callback(self, data_config):
        if self.solver is None:
            raise ValueError("Solver is not set. Please set the solver first.")
        benchmark_callback = benchmark_callback(data_config)
        self.data = benchmark_callback.data
        self.solver.setCallbacks(benchmark_callback(data_config))

    def run(self):
        if self.solver is None:
            raise ValueError("Solver is not set. Please set the solver first.")
        self.solver.solve()
        self.plotter.plot(self.data)

    