import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.num_moves = 0 # Number of Moves
        self.num_penalty = 0 # Number of Penalty
        self.cumulative_rewards = 0 # Cumulative Rewards
        self.failure = 0 # Number of Failure
        self.penalty_ratio = 0 # Penalty Ratio = Number of Penalty / Number of Moves * 100
        self.num_trial = 0 # Number of Trials
        self.success_rate = 0 # Success Rate = 1 - (Number of Failure / Number of Trials)
        self.possible_action = [None, 'forward', 'left', 'right']
        self.Q = {}
        for i in ['green', 'red']: #light
            for j in ['forward', 'left', 'right']: #next_waypoint
                for k in [None, 'forward', 'left', 'right']: #oncoming
                    for l in [None, 'forward', 'left', 'right']: #left
                        #initialize Q with traffic policies
                        #self.Q[(i,j,k,l)] = [random.randint(0,5)] * 4
                        if i == 'green':
                            if k == 'forward':
                                self.Q[(i,j,k,l)] = [0, 0, -2, 0]
                            else: 
                                self.Q[(i,j,k,l)] = [-2, 0, 0, 0]
                        else:
                            if k == 'left' or l == 'forward':
                                self.Q[(i,j,k,l)] = [0, -2, -2, -2]
                            else:
                                self.Q[(i,j,k,l)] = [0, -2, -2, 0] 
                        if j == 'forward':
                            self.Q[(i,j,k,l)][1] += 1
                        elif j == 'left':
                            self.Q[(i,j,k,l)][2] += 1
                        else:
                            self.Q[(i,j,k,l)][3] += 1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.num_trial += 1
        print "Penalty Ratio: {:.3f}, Success Rate: {:.3f}, Cumulative Rewards: {}, Number of Moves: {}".format(self.penalty_ratio, self.success_rate, self.cumulative_rewards, self.num_moves)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        alpha = 1.0 / (1.2 + float(self.num_moves) / 60.0)
        #alpha = 1.0
        gamma = 0.3
        self.state = (inputs['light'],self.next_waypoint,inputs['oncoming'],inputs['left'])
        
        # TODO: Select action according to your policy
        Q_max = self.Q[self.state].index(max(self.Q[self.state]))
        # implement epsilon-greedy approach
        epsilon = 1.0 / (1.0 + self.num_moves / 10.0)
        if random.random() < epsilon:
            action = random.choice(self.possible_action)
        else:
            action = self.possible_action[Q_max]

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            self.num_penalty += 1
        self.cumulative_rewards += reward
        if deadline == 0:
            self.failure += 1
        self.success_rate = 1.0 - float(self.failure) / float(self.num_trial)
        self.num_moves += 1
        self.penalty_ratio = float(self.num_penalty) / float(self.number_moves) * 100

        # TODO: Learn policy based on state, action, reward
        future_inputs = self.env.sense(self)
        future_waypoint = self.planner.next_waypoint()
        future_state = (future_inputs['light'],future_waypoint,future_inputs['oncoming'],future_inputs['left'],)
        self.Q[self.state][self.possible_action.index(action)] = (1 - alpha) * self.Q[self.state][self.possible_action.index(action)] + (alpha * (reward + gamma * max(self.Q[future_state])))
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print "Penalty Ratio: {:.3f}, Success Rate: {:.3f}, Cumulative Rewards: {}, Number of Moves: {}".format(self.penalty_ratio, self.success_rate, self.cumulative_rewards, self.num_moves)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()
