# we don't have the image-based planner in this code! and the MPC takes a set of desired waypoints as input for planning
from evotorch import Problem
from evotorch.algorithms import SNES, CEM, CMAES
from evotorch.logging import StdOutLogger, PandasLogger
import torch
import numpy as np
from mpc_planner.system_identification import MHE_MPC

class rc_car_model:
    def __init__(self):
        # initial values of the parameters
        self.C1= torch.tensor(0.5)
        self.C2 = torch.tensor(10/6)
        self.Cm1 = torch.tensor(12)
        self.Cm2 = torch.tensor(2.5)
        self.Cr2 = torch.tensor(0.15)
        self.Cr0 = torch.tensor(0.7)
        self.mu_m = torch.tensor(4.0)
        self.g_ = torch.tensor(9.81)
        self.dt = torch.tensor(0.2)

        self.states = torch.tensor(np.array([0, 0, 0, 0, 0]),dtype=torch.float32)
    # where we use MHE to update the parameters each time we get a new measurements
    def parameters_update(self, updated_parameters):
        self.C1,self.Cm1, self.Cm2, self.Cr2, self.Cr0, self.mu_m = torch.tensor(updated_parameters, dtype=torch.float32).cuda()

    def step(self, X, Y, Sai, V, Pitch, sigma, forward_throttle):
        sigma = torch.tanh(sigma)*(-0.6)
        #forward_throttle = torch.tanh(forward_throttle)
        X = (V * torch.cos(Sai + self.C1 * sigma))*self.dt + X
        Y = (V * torch.sin(Sai + self.C1 * sigma))*self.dt + Y
        Sai = (V * sigma * self.C2)*self.dt + Sai
        V = ((self.Cm1 - self.Cm2 * V ) * forward_throttle - ((self.Cr2 * V ** 2 + self.Cr0) + \
                    (V * sigma)**2 * (self.C2 * self.C1 ** 2)) - self.mu_m*self.g_*torch.sin(Pitch))*self.dt + V
        Pitch = Pitch # static
        return X, Y, Sai, V, Pitch

class mpc_planner:
    def __init__(self, receding_horizon, num_of_actions):
        self.system_model = rc_car_model()
        self.receding_horizon = receding_horizon
        self.num_of_actions = num_of_actions
        self.num_of_states = 5
        #self.Q =  np.eye(self.num_of_states)# Weighting matrix for state trajectories
        #self.R =  0.1*np.eye(self.num_of_actions)# Weighting matrix for control actions
        #self.delta_R = np.eye(self.num_of_actions)
        #self.stack_variables = np.ones(4)
        #self.lambda_list = np.ones(4)
        # The estimator
        self.estimation_algorithm = MHE_MPC()
        # the optimization solver
        self.CEM_initialization()

        self.desired_pose = torch.zeros(2) # X , Y
        self.desired_heading = torch.zeros(0)
    def CEM_initialization(self):
        bounds = (-1,1)
        higher_bounds = bounds[1]*torch.ones(self.num_of_actions)
        lower_bounds = bounds[0]*torch.ones(self.num_of_actions)
        problem = Problem(
            "min",
            self.MPC_cost,
            initial_bounds=(-1, 1),
            bounds=(-1,1),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            solution_length=self.num_of_actions,
            # Higher-than-default precision
            dtype=torch.float32,
        )

        # Create a SearchAlgorithm instance to optimise the Problem instance
        self.searcher = CMAES(problem, popsize=50, stdev_init=1)

    def MPC_cost(self, set_of_actions):
        set_of_steering_angles = set_of_actions[:self.receding_horizon]
        ##set_of_throttles = set_of_actions[self.receding_horizon:]
        # initialization of the mpc algorithm with the current states of the model
        states = self.system_model.states
        loss = torch.zeros(self.receding_horizon).cuda()#torch.tensor([0],dtype=torch.float32).cuda()
        set_of_throttles = torch.tensor(0.2).cuda() # constant velocity
        for i in range(self.receding_horizon):
            states = self.system_model.step(*states, set_of_steering_angles[i], set_of_throttles)
            #coef_vel = torch.tensor([1],dtype=torch.float32).cuda()
            loss[i] = (self.desired_heading - states[2])**2#(states[0] - self.desired_pose[0])**2 + 2*(states[1] - self.desired_pose[1])**2
        return loss.sum()
    def plan(self, desired_pose, unreal_mode=False, given_observations=None):
        if unreal_mode:
            observations = np.array(given_observations)
        else:
            observations = self.estimation_algorithm.measurement_update()
        # update the states
        print('obs: ',observations)
        self.system_model.states = torch.tensor(self.estimation_algorithm.mhe.make_step(observations),dtype=torch.float32).cuda()
        # update the parameters
        # Note that we need to set local targets, while the states of the vehicle are more like global
        if desired_pose[0]==-100 and desired_pose[1] == -100:
            return [torch.tensor(0.0).cuda(),0.0]
        else:
            self.desired_pose[0] = self.system_model.states[0] + desired_pose[0]
            self.desired_pose[1] = self.system_model.states[1] + desired_pose[1]
            self.desired_heading = self.system_model.states[2] + \
                                   (torch.atan2(torch.tensor(desired_pose[0]),-torch.tensor(desired_pose[1])) - 0.5*torch.tensor(np.math.pi))
            self.searcher.run(5)
            print('states of the vehicle: ',self.system_model.states[2])
            print('Desired: ', self.desired_heading)
            print('Target heading: ', torch.atan2(torch.tensor(desired_pose[0]),-torch.tensor(desired_pose[1])))

            #self.debug_(self.searcher.status['pop_best'].values)
            return torch.tanh(self.searcher.status['pop_best'].values)*(-0.6) #torch.tanh(sigma)*(-0.6)

    def debug_(self, set_of_actions):
        set_of_steering_angles = set_of_actions[:self.receding_horizon]
        #set_of_throttles = set_of_actions[self.receding_horizon:]
        states = self.system_model.states
        set_of_throttles = torch.tensor(0.2).cuda()  # constant velocity
        for i in range(self.receding_horizon):
            states = self.system_model.step(*states, set_of_steering_angles[i], set_of_throttles)
            #print('Action: ',[set_of_steering_angles[i].detach().cpu(),set_of_throttles.detach().cpu()],' *states: ', states)

if __name__ == '__main__':
    import time
    planning_horizon = 10
    num_of_actions = planning_horizon   # one for the velocity
    main_planner = mpc_planner(planning_horizon, num_of_actions)
    dX = 0
    dY = 0
    for i in range(1):
        dX = 1.5 + dX
        dY = -0.5 + dY
        begin_time = time.time()
        main_planner.plan([dX, dY])
        print(time.time() - begin_time)
