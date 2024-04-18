import benchmark

problem_config, solver_config, data_config =  "problem_configs/" + "quadrupedal_walking_fwd", \
                                                              "solver_configs/" + "csqp_filter", \
                                                              "data_configs/" + "csqp_data"
benchmark_ = benchmark.make(problem_config, solver_config, data_config)
data = benchmark_.run()