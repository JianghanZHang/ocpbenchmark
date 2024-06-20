import benchmark
import numpy as np
from plotter import plot_data

problem_config, solver_config, data_config =  "problem_configs/" + "quadrupedal_walking_hard", \
                                                              "solver_configs/" + "csqp_merit", \
                                                              "data_configs/" + "csqp_data"
benchmark_ = benchmark.make(problem_config, solver_config, data_config)
data = benchmark_.run()

plot = plot_data(data)
