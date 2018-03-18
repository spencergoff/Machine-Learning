import gym

env = gym.make('CartPole-v0') #make the environment, set up any necessary dependencies
observation = env.reset()

for _ in range(1000):
    env.render() # displays visualization
    cart_position, cart_velocity, pole_angle, angle_velociy = observation

    #Here's the policy
    if pole_angle > 0: #pole is leaning to the right
        action = 1
    else:
        action = 0

    observation, reward, done, info = env.step(action)
