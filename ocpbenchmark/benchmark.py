import crocoddyl
import problem
import plot
import utils
import solver
class benchmark:
    def __init__(self):
        self.problem = None
        self.solver = None
        self.plotter = None

    @staticmethod
    def make(problem_config, plotter_config, solver_config):
        benchmark = benchmark()
        benchmark.set_problem(problem_config)
        benchmark.set_solver(solver_config)
        benchmark.set_plotter(plotter_config)
        return benchmark
        

    def set_plotter(self, plotter_config):
        self.plotter = plot(plotter_config)

    def set_problem(self, problem_config):
        self.problem = problem.make_problem(problem_config) 

    def set_solver(self, solver_config):
        if self.problem is None:
            raise ValueError("Problem is not set. Please set the problem first.")
        self.solver = solver.make_solver(self.problem, solver_config) 

    