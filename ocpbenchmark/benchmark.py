import crocoddyl
import problem
import plot
import utils
class benchmark:
    def __init__(self):
        self.problem = None
        self.solver = None
        self.plotter = None

    @staticmethod
    def make(solver, problem_config, plotter_config):
        benchmark = benchmark()
        benchmark.set_problem(problem_config)
        benchmark.set_solver(solver)
        benchmark.set_plotter(plotter_config)
        return benchmark
        

    def set_plotter(self, plotter_config):
        self.plotter = plot(plotter_config)

    def set_problem(self, problem_config):
        self.problem = problem.make_problem(problem_config) 

    def set_solver(self, solver):
        self.solver = solver

    