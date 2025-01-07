import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# In this example test, we will solve the reaching-goal task with the Kinova arm.
# For that, we use the inverse dynamics (with its analytical derivatives) developed
# inside crocoddyl; it is described inside DifferentialActionModelFreeInvDynamics class.
# Finally, we use an Euler sympletic integration scheme.



def one_step(current_q, current_goal):
    # First, let's load create the state and actuation models
    kinova = example_robot_data.load("panda")
    print("kinova = ", kinova)
    robot_model = kinova.model
    print("robot_model = ", robot_model)
    state = crocoddyl.StateMultibody(robot_model)
    print("state = ", state)
    actuation = crocoddyl.ActuationModelFull(state)
    print("actuation = ", actuation)
    print(actuation)
    # q0 = kinova.model.referenceConfigurations["arm_up"]
    q0 = current_q
    # x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])
    x0 = q0
    print("first x0")
    print(x0)

    # Create a cost model per the running and terminal action model.
    nu = state.nv
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)

    # Note that we need to include a cost model (i.e. set of cost functions) in
    # order to fully define the action model for our optimal control problem.
    # For this particular example, we formulate three running-cost functions:
    # goal-tracking cost, state and control regularization; and one terminal-cost:
    # goal cost. First, let's create the common cost functions.
    framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        robot_model.getFrameId("panda_link6"),
        pinocchio.SE3(np.eye(3), current_goal),
        nu,
    )
    uResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)
    xResidual = crocoddyl.ResidualModelState(state, x0, nu)
    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
    # xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    # uRegCost = crocoddyl.CostModelResidual(state, uResidual)

    # Then let's added the running and terminal cost functions
    runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
    # regularization cost is removed for simplicity
    # runningCostModel.addCost("xReg", xRegCost, 1e-1)
    # runningCostModel.addCost("uReg", uRegCost, 1e-1)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)

    # Next, we need to create an action model for running and terminal knots. The
    # forward dynamics (computed using RNEA) are implemented
    # inside DifferentialActionModelFreeInvDynamics.
    dt = 1e-2
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeInvDynamics(
            state, actuation, runningCostModel
        ),
        dt,
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeInvDynamics(
            state, actuation, terminalCostModel
        ),
        0.0,
    )

    # For this optimal control problem, we define 100 knots (or running action
    # models) plus a terminal knot
    T = 5
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    solver = crocoddyl.SolverIntro(problem)

    # I am not using logger for now
    #### # defining a logger
    #### if WITHPLOT:
    ####     solver.setCallbacks(
    ####         [
    ####             crocoddyl.CallbackVerbose(),
    ####             crocoddyl.CallbackLogger(),
    ####         ]
    ####     )
    #### else:
    ####     solver.setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving it with the solver algorithm
    print("python called solve\n\n\n", flush=True)
    solver.solve()
    print("python done solve\n\n\n", flush=True)

    return solver.xs


kinova = example_robot_data.load("panda")
robot_model = kinova.model
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFull(state)
# q0 = kinova.model.referenceConfigurations["arm_up"]
q0 = [-1.7935987837186267, -1.59100425037732, -0.03295082534947273, -2.206859679378054, 0.053584024206261, 0.600290502125782
  , 0.61800000e+00,  1.00000000e+00,
  0.00000000e+00, 0, 0, 0]
print("q0 = ", q0)
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])
print("x0 = ", x0)
x0 = x0[:-3]
x0 = x0[:-3]
print("x00 = ", x0)

goal = np.array([0.5, 0.3, 0.4])
guess_q0 = np.array([1.0, 0.0, 0.5])
N = 5
diff = goal - guess_q0
diff = diff / N

curr_goal = guess_q0 + diff
print("x0 = ", x0, flush=True)
xs = one_step(x0, curr_goal)

all_xs = []
all_xs = all_xs + [x for x in xs]
print("xs = ", xs)

xs = xs[-1]

for i in range(N-1):
    curr_goal = curr_goal + diff
    xs = one_step(xs, curr_goal)
    all_xs = all_xs + [x for x in xs]
    xs = xs[-1]
    # print("xs = ", xs)
    # print("goal = ", goal)
    # print("q0 = ", guess_q0)

print("all_xs = ", all_xs)



"""
all_qs = []
# make it a loop so that we call it multiple times
for i in range(1000):
    # we need to change the start position
    current_q = solver.problem.terminalData.differential.multibody.pinocchio.oMf[
        robot_model.getFrameId("j2s6s200_end_effector")
    ].translation.T,
    q0 = kinova.model.referenceConfigurations["arm_up"]
    print("q0 = ", q0)
    print("current_q = ", [d.differential.multibody.joint.tau for d in solver.problem.runningDatas][-1])
    q0 = solver.xs[-1]
    all_qs.append(q0)
    # append [2.618, 1.0, 0.0] to q0
    # q0 = np.append(q0, [2.618, 1.0, 0.0])
    # x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])
    x0 = q0

    xResidual = crocoddyl.ResidualModelState(state, x0, nu)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    newRunningCostModel = crocoddyl.CostModelSum(state, nu)
    newRunningCostModel.addCost("gripperPose", goalTrackingCost, 1)
    newRunningCostModel.addCost("xReg", xRegCost, 1e-1)
    newRunningCostModel.addCost("uReg", uRegCost, 1e-1)

    newRunningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeInvDynamics(
            state, actuation, newRunningCostModel
        ),
        dt,
    )

    T = 40
    problem = crocoddyl.ShootingProblem(x0, [newRunningModel] * T, terminalModel)

    solver = crocoddyl.SolverIntro(problem)
    if WITHPLOT:
        solver.setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    solver.solve()

print("all_qs = ")
for q in all_qs:
    print(q)
"""

"""
print(
    "Finally reached = ",
    solver.problem.terminalData.differential.multibody.pinocchio.oMf[
        robot_model.getFrameId("j2s6s200_end_effector")
    ].translation.T,
)
"""

"""
# Plotting the solution and the solver convergence
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(
        solver.xs,
        [d.differential.multibody.joint.tau for d in solver.problem.runningDatas],
        figIndex=1,
        show=False,
    )
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
    )
"""

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    try:
        import gepetto

        cameraTF = [2.0, 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(kinova, 4, 4, cameraTF, floor=False)
    except Exception:
        display = crocoddyl.MeshcatDisplay(kinova)

    display.rate = -1
    display.freq = 1
    while True:
        display.display(all_xs)
        # print("solver.xs = ", solver.xs)
        from pprint import pprint
        # pprint(solver.xs[0])
        # display.displayFromSolver(solver)
        time.sleep(1.0)
        

