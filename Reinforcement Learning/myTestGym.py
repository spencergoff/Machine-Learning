import gym

env = gym.make('CartPole-v0') #installs any necessary dependencies; none for CartPole

env.reset() #reset environment to default starting location (e.g. the cart at 0 for CartPole)

for t in range(1000): #render (view) env over several time steps
    env.render()
    env.step(env.action_space.sample()) #take a random action from action_space (e.g. 0 or 1)
    # returns 4 things: observation (angles, velocities, etc.); reward (amount achieved by the previous action);
    #                   done (boolean indicating if env needs to be reset); info (diagnostics, debugging info)
