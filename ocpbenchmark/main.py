import benchmark
import numpy as np
# print('SQP:')
# problem_config, solver_config, data_config =  "problem_configs/" + "quadrupedal_walking_hard", \
#                                                               "solver_configs/" + "sqp", \
#                                                               "data_configs/" + "ddp_data"
# benchmark_ = benchmark.make(problem_config, solver_config, data_config)
# data = benchmark_.run()

print('CSQP:')
problem_config, solver_config, data_config =  "problem_configs/" + "quadrupedal_walking_hard", \
                                                              "solver_configs/" + "csqp_merit", \
                                                              "data_configs/" + "csqp_data"
benchmark_ = benchmark.make(problem_config, solver_config, data_config)
data = benchmark_.run()

# optimal_controls_csqp = list(data['us'][-1])

print('DDP:')
problem_config, solver_config, data_config =  "problem_configs/" + "quadrupedal_walking_soft", \
                                                              "solver_configs/" + "ddp", \
                                                              "data_configs/" + "ddp_data"
benchmark_ = benchmark.make(problem_config, solver_config, data_config)
data = benchmark_.run()


# optimal_controls_ddp = list(data['us'][-1])
# diff = 0.
# for i in range(len(optimal_controls_csqp)):
#     optimal_control_csqp = optimal_controls_csqp[i]
#     optimal_control_ddp = optimal_controls_ddp[i]
#     diff += np.linalg.norm(np.asarray(optimal_control_csqp, dtype="object")- np.asarray(optimal_control_ddp, dtype="object"), 1)

# print(f'Diff between CSQP (with hard constriants) and DDP (with soft constraints): {diff / len(optimal_controls_csqp)}')
