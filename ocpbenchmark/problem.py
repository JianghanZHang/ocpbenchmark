"""
Return a predefined shooting problem instance.
"""
import crocoddyl

def make_problem(problem_config):
    initial_state = problem_config['initial_state']
    horizon = problem_config['horizon']
    weights = problem_config['weights']
    robot_type = problem_config['robot_type']
    problem_type = problem_config['problem_type']
    if robot_type == 'bipedal':
        

