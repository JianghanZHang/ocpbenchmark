"""
Return a predefined shooting problem instance.
"""
import crocoddyl
import callback
from examples.biped import SimpleBipedGaitProblem
import example_robot_data
import pinocchio
import numpy as np

def make_problem(problem_config):
    initial_state = problem_config['initial_state']
    initial_guess = problem_config['initial_guess']
    horizon = problem_config['horizon']
    weights = problem_config['weights']
    robot_type = problem_config['robot_type']
    problem_type = problem_config['problem_type']
    dt = problem_config['dt']

    if robot_type == 'bipedal':
        talos_legs = example_robot_data.load("talos_legs")
        # Defining the initial state of the robot
        q0 = talos_legs.model.referenceConfigurations[initial_state].copy()
        v0 = pinocchio.utils.zero(talos_legs.model.nv)
        x0 = np.concatenate([q0, v0])

        # Setting up the 3d walking problem
        rightFoot = "right_sole_link"
        leftFoot = "left_sole_link"
        gait = SimpleBipedGaitProblem(talos_legs.model, rightFoot, leftFoot)

        GAITPHASES = [
            {
                "walking": {
                    "stepLength": 0.6,
                    "stepHeight": 0.1,
                    "timeStep": 0.03,
                    "stepKnots": 35,
                    "supportKnots": 10,
                }
            },
            {
                "jumping": {
                    "jumpHeight": 0.15,
                    "jumpLength": [0.0, 0.3, 0.0],
                    "timeStep": 0.03,
                    "groundKnots": 10,
                    "flyingKnots": 20,
                }
            }  
        ]
        
        if problem_type == "walking":
            # Creating a walking problem
                return gait.createWalkingProblem(
                    x0,
                    GAITPHASES[problem_type]["stepLength"],
                    GAITPHASES[problem_type]["stepHeight"],
                    GAITPHASES[problem_type]["timeStep"],
                    GAITPHASES[problem_type]["stepKnots"],
                    GAITPHASES[problem_type]["supportKnots"],
                )
        elif problem_type == "jumping":
             return gait.createJumpingProblem(
                    x0,
                    GAITPHASES[problem_type]["stepLength"],
                    GAITPHASES[problem_type]["stepHeight"],
                    GAITPHASES[problem_type]["timeStep"],
                    GAITPHASES[problem_type]["stepKnots"],
                    GAITPHASES[problem_type]["supportKnots"],
                )
        else:
            raise ValueError(f"Problem type '{problem_type}' not implemented.")
        
    else:
        raise ValueError(f"Robot type '{robot_type}' not implemented.")



                

