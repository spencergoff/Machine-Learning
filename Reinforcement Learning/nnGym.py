import tensorflow as tf
import gym
import numpy as np

num_inputs = 4
num_hidden = 4
num_outputs = 1 #probability that the cart should go left

initializer = tf.contrib.layers.variance_scaling_initializer()

x = tf.placeholder(tf.float32,shape=[None,num_inputs])

hidden_layer_one = tf.layers.dense(x,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)

hidden_layer_two = tf.layers.dense(hidden_layer_one,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)

output_layer = tf.layers.dense(hidden_layer_two,num_outputs,activation=tf.nn.sigmoid,kernel_initializer=initializer)

probabilities = tf.concat(axis=1,values=[output_layer,1-output_layer])
action = tf.multinomial(probabilities,num_samples=1) #result will be 0 or 1

init = tf.global_variables_initializer()

epi = 250 #how many times to run the game (episodes)
step_limit = 500 #time steps limit within each game/episode (if the game isn't done by then, set done = True)
env = gym.make('CartPole-v0')
avg_steps = []

with tf.Session() as sess:
    init.run()
    for i_episode in range(epi):
        obs = env.reset()
        for step in range(step_limit):
            action_val = action.eval(feed_dict={x:obs.reshape(1,num_inputs)})
            obs, reward, done, info = env.step(action_val[0][0]) #returns 0 or 1
            if done:
                avg_steps.append(step)
                print('Done after {} steps'.format(step))
                break
        env.render() # displays visualization

print('After {} Episodes, average steps per game was {}'.format(epi,np.mean(avg_steps)))
env.close()
