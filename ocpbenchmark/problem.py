"""
Return a predefined shooting problem instance.
"""
import crocoddyl
import callback
from examples.biped import SimpleBipedGaitProblem
from examples.quadruped_hard_constraints import SimpleQuadrupedGaitProblem_hard_constraints
from examples.quadruped_soft_constraints import SimpleQuadrupedGaitProblem_soft_constraints
import example_robot_data
import pinocchio
import numpy as np
from utils import load_config_file

def make_problem(problem_config):
    """
    input:
        problem_config: str
            The name of the problem configuration file.
            
    output:
        problem: crocoddyl.ShootingProblem
            The shooting problem instance.
    """
    config = load_config_file(problem_config)
    robot_type = config['robot_type']
    problem_type = config['problem_type']
    dynamics_type = config['dynamics_type']
    constraints_type = config['constraints_type']

    if robot_type == 'bipedal':
        talos_legs = example_robot_data.load("talos_legs")
        # Defining the initial state of the robot
        q0 = talos_legs.model.referenceConfigurations["half_sitting"].copy()
        v0 = pinocchio.utils.zero(talos_legs.model.nv)
        x0 = np.concatenate([q0, v0])

        # Setting up the 3d walking problem
        rightFoot = "right_sole_link"
        leftFoot = "left_sole_link"

        if dynamics_type == "inv":
            raise ValueError(f"Dynamics type '{dynamics_type}' is not supported since the solver cannot deal with equality constraints.")
            gait = SimpleBipedGaitProblem(talos_legs.model, rightFoot, leftFoot, fwddyn=False)

        elif dynamics_type == "fwd":
            gait = SimpleBipedGaitProblem(talos_legs.model, rightFoot, leftFoot, fwddyn=True)

        else:
            raise ValueError(f"Dynamics type '{dynamics_type}' not implemented.")


        GAITPHASES = {
                "walking": {
                    "stepLength": 0.6,
                    "stepHeight": 0.1,
                    "timeStep": 0.03,
                    "stepKnots": 35,
                    "supportKnots": 10,
                }
           ,
                "jumping": {
                    "jumpHeight": 0.15,
                    "jumpLength": [0.0, 0.3, 0.0],
                    "timeStep": 0.03,
                    "groundKnots": 10,
                    "flyingKnots": 20,
                }
            }  
        
        
        if problem_type == "walking":
            # Creating a walking problem
                problem = gait.createWalkingProblem(
                    x0,
                    GAITPHASES[problem_type]["stepLength"],
                    GAITPHASES[problem_type]["stepHeight"],
                    GAITPHASES[problem_type]["timeStep"],
                    GAITPHASES[problem_type]["stepKnots"],
                    GAITPHASES[problem_type]["supportKnots"],
                )
        elif problem_type == "jumping":
             problem = gait.createJumpingProblem(
                    x0,
                    GAITPHASES[problem_type]["stepLength"],
                    GAITPHASES[problem_type]["stepHeight"],
                    GAITPHASES[problem_type]["timeStep"],
                    GAITPHASES[problem_type]["stepKnots"],
                    GAITPHASES[problem_type]["supportKnots"],
                )
        else:
            raise ValueError(f"Problem type '{problem_type}' not implemented.")
    
    elif robot_type == 'quadrupedal':
                
        # Loading the anymal model
        anymal = example_robot_data.load("anymal")

        # Defining the initial state of the robot
        q0 = anymal.model.referenceConfigurations["standing"].copy()
        v0 = pinocchio.utils.zero(anymal.model.nv)
        x0 = np.concatenate([q0, v0])

        # Setting up the 3d walking problem
        lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
        if dynamics_type == "inv":
            if constraints_type == "hard":
                gait = SimpleQuadrupedGaitProblem_hard_constraints(
                    anymal.model, lfFoot, rfFoot, lhFoot, rhFoot, fwddyn=False
                )
            elif constraints_type == "soft":
                gait = SimpleQuadrupedGaitProblem_soft_constraints(
                    anymal.model, lfFoot, rfFoot, lhFoot, rhFoot, fwddyn=False
                )
        elif dynamics_type == "fwd":
            if constraints_type == "hard":
                gait = SimpleQuadrupedGaitProblem_hard_constraints(
                    anymal.model, lfFoot, rfFoot, lhFoot, rhFoot, fwddyn=True
                )
            elif constraints_type == "soft":
                gait = SimpleQuadrupedGaitProblem_soft_constraints(
                    anymal.model, lfFoot, rfFoot, lhFoot, rhFoot, fwddyn=True
                )
        else:
            raise ValueError(f"Dynamics type '{dynamics_type}' not implemented.")

        # Setting up all tasks
        GAITPHASES ={
                "walking": {
                    "stepLength": 0.25,
                    "stepHeight": 0.15,
                    "timeStep": 1e-2,
                    "stepKnots": 25,
                    "supportKnots": 2,
                }
            ,
                "trotting": {
                    "stepLength": 0.15,
                    "stepHeight": 0.1,
                    "timeStep": 1e-2,
                    "stepKnots": 25,
                    "supportKnots": 2,
                }
            ,
                "pacing": {
                    "stepLength": 0.15,
                    "stepHeight": 0.1,
                    "timeStep": 1e-2,
                    "stepKnots": 25,
                    "supportKnots": 5,
                }
            ,
                "bounding": {
                    "stepLength": 0.15,
                    "stepHeight": 0.1,
                    "timeStep": 1e-2,
                    "stepKnots": 25,
                    "supportKnots": 5,
                }
            ,
                "jumping": {
                    "jumpHeight": 0.15,
                    "jumpLength": [0.0, 0.3, 0.0],
                    "timeStep": 1e-2,
                    "groundKnots": 10,
                    "flyingKnots": 20,
                }
            }
        if problem_type == "walking":
            problem =  gait.createWalkingProblem(
                                x0,
                                GAITPHASES[problem_type]["stepLength"],
                                GAITPHASES[problem_type]["stepHeight"],
                                GAITPHASES[problem_type]["timeStep"],
                                GAITPHASES[problem_type]["stepKnots"],
                                GAITPHASES[problem_type]["supportKnots"],
                            )
        elif problem_type == "trotting":
             problem =  gait.createTrottingProblem(
                                x0,
                                GAITPHASES[problem_type]["stepLength"],
                                GAITPHASES[problem_type]["stepHeight"],
                                GAITPHASES[problem_type]["timeStep"],
                                GAITPHASES[problem_type]["stepKnots"],
                                GAITPHASES[problem_type]["supportKnots"],
                            )
        elif problem_type == "pacing":
            problem = gait.createBoundingProblem(
                                x0,
                                GAITPHASES["stepLength"],
                                GAITPHASES["stepHeight"],
                                GAITPHASES["timeStep"],
                                GAITPHASES["stepKnots"],
                                GAITPHASES["supportKnots"],
                            )
        elif problem_type == "jumping":
            problem = gait.createJumpingProblem(
                                x0,
                                GAITPHASES["jumpHeight"],
                                GAITPHASES["jumpLength"],
                                GAITPHASES["timeStep"],
                                GAITPHASES["groundKnots"],
                                GAITPHASES["flyingKnots"],
                            )
        elif problem_type == "bounding":
            problem = gait.createBoundingProblem(
                                x0,
                                GAITPHASES["stepLength"],
                                GAITPHASES["stepHeight"],
                                GAITPHASES["timeStep"],
                                GAITPHASES["stepKnots"],
                                GAITPHASES["supportKnots"],
                            )
        else:
            raise ValueError(f"Problem type '{problem_type}' not implemented.")

    else:
        raise ValueError(f"Robot type '{robot_type}' not implemented.")
    
    return problem



                

