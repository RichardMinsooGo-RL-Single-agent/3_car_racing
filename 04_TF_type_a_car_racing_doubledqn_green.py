import tensorflow as tf
import random
import numpy as np
import time, datetime
from collections import deque
import pickle

from game import Game
# from model import DQN_agent

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()

import sys

# target_update_cycle = 1000

# 4 프레임마다 한 번씩 학습합니다.
TRAIN_INTERVAL = 4

screen_width = 6
screen_height = 10

game_name = 'car_racing_double_dqn'    # the name of the game being played for log files
# action: 0: left, 1: keep, 2: right
action_size = 3

model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

class DQN_agent:
    STATE_LEN = 4

    def __init__(self, sess, screen_width, screen_height, action_size):
        # Get parameters
        self.progress = " "
        
        self.sess = sess
        self.action_size = action_size
        
        # train time define
        self.training_time = 5*60
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.0001
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        # final value of epsilon
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.001
        self.epsilon = self.epsilon_max
        
        self.step = 0
        self.score = 0
        self.episode = 0
        
        self.ep_trial_step = 5000
        
        # parameters for skipping and stacking
        # Parameter for Experience Replay
        self.size_replay_memory = 50000
        self.batch_size = 64
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 1000
        
        # Parameter for Target Network        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.state = None

        self.input_X = tf.placeholder(tf.float32, [None, screen_width, screen_height, self.STATE_LEN])
        self.input_A = tf.placeholder(tf.int64, [None])
        self.input_Y = tf.placeholder(tf.float32, [None])

        self.Q = self.build_model('main')
        self.target_Q = self.build_model('target')
        self.train_step, self.loss = self.loss_and_train()

    def reset_env(self, state):
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.stack(state, axis=2)

    def build_model(self, network_name):
        with tf.variable_scope(network_name):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            # model = tf.layers.dense(model, 1024, activation=tf.nn.relu)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)
            output = tf.layers.dense(model, self.action_size, activation=None)

        return output

    def loss_and_train(self):
        # Loss function and Train
        action_tgt = tf.one_hot(self.input_A, self.action_size, 1.0, 0.0)
        y_prediction = tf.reduce_sum(tf.multiply(self.Q, action_tgt), axis=1)
        Loss = tf.reduce_mean(tf.square(y_prediction - self.input_Y))
        train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(Loss)

        return train_step, Loss

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        # sample a minibatch to train on
        minibatch = random.sample(self.memory, self.batch_size)

        # Save the each batch data
        states      = [batch[0] for batch in minibatch]
        actions     = [batch[1] for batch in minibatch]
        rewards     = [batch[2] for batch in minibatch]
        next_states = [batch[3] for batch in minibatch]
        dones       = [batch[4] for batch in minibatch]

        # Get target values
        y_array = []
        # Selecting actions
        q_value_next = self.sess.run(self.Q, feed_dict={self.input_X: next_states})
        tgt_q_value_next = self.sess.run(self.target_Q, feed_dict={self.input_X: next_states})
        
        for i in range(self.batch_size):
            if dones[i]:
                y_array.append(rewards[i])
            else:
                a = np.argmax(tgt_q_value_next[i])
                y_array.append(rewards[i] + self.discount_factor * q_value_next[i][a] )

        feed_dict={self.input_X: states, self.input_A: actions, self.input_Y: y_array}
        self.sess.run(self.train_step, feed_dict=feed_dict)
        
        # Decrease epsilon while training
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else :
            self.epsilon = self.epsilon_min

    # get action from model using epsilon-greedy policy
    def get_action(self):
        if np.random.rand() < self.epsilon:
            action = random.randrange(action_size)
        else:
            Q_value = self.sess.run(self.Q, feed_dict={self.input_X: [self.state]})
            action = np.argmax(Q_value[0])

        return action

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, done):
        next_state = np.reshape(state, (self.screen_width, self.screen_height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)
        #in every action put in the memory
        self.memory.append((self.state, action, reward, next_state, done))

        self.state = next_state
        
        while len(self.memory) > self.size_replay_memory:
            self.memory.popleft()
            
    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        op_holder = []
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        self.sess.run(op_holder)

    def save_model(self):
        # Save the variables to disk.
        save_path = self.saver.save(self.sess, model_path + "/model.ckpt")

        with open(model_path + '/append_sample.pickle', 'wb') as f:
            pickle.dump(self.memory, f)

        save_object = (self.epsilon, self.episode, self.step)
        with open(model_path + '/epsilon_episode.pickle', 'wb') as ggg:
            pickle.dump(save_object, ggg)

        print("\n Model saved in file: %s" % save_path)

def main():
    
    sess = tf.Session()
    # with tf.Session() as sess:
    game = Game(screen_width, screen_height, show_game=False)
    agent = DQN_agent(sess, screen_width, screen_height, action_size)

    rewards = tf.placeholder(tf.float32, [None])
    # Initialize variables
    # Load the file if the saved file exists
    init = tf.global_variables_initializer()
    agent.saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        agent.saver.restore(agent.sess, ckpt.model_checkpoint_path)
        if os.path.isfile(model_path + '/append_sample.pickle'):  
            with open(model_path + '/append_sample.pickle', 'rb') as f:
                agent.memory = pickle.load(f)

            with open(model_path + '/epsilon_episode.pickle', 'rb') as ggg:
                agent.epsilon, agent.episode, agent.step = pickle.load(ggg)
            
        print('\n\n Variables are restored!')

    else:
        agent.sess.run(init)
        print('\n\n Variables are initialized!')
        agent.epsilon = agent.epsilon_max
    
    # open up a game state to communicate with emulator
    avg_score = 0
    scores = []

    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    print("\n\n",game_name, "-game start at :",display_time,"\n")
    
    start_time = time.time()
    
    # Initialize target network.
    agent.Copy_Weights()
    
    while time.time() - start_time < 15*60 and avg_score < 4900:
    # for episode in range(MAX_EPISODE):
        done = False
        agent.score = 0
        ep_step = 0
        
        state = game.reset()
        agent.reset_env(state)

        while not done and ep_step < agent.ep_trial_step:
            
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"            
            else:
                agent.progress = "Training"

            ep_step += 1
            agent.step += 1

            # Select action
            action = agent.get_action()
            state, reward, done = game.step(action)
            agent.append_sample(state, action, reward, done)
            
            if agent.progress == "Training" and agent.step % TRAIN_INTERVAL == 0:
                # Training!
                agent.train_model()

                if agent.step % agent.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agent.Copy_Weights()

            agent.score += reward
            
            # If game is over (done)
            if done or ep_step == agent.ep_trial_step:
                if agent.progress == "Training":
                    agent.episode += 1
                print('episode :{:>7,d}'.format(agent.episode),'/ ep step :{:>6,d}'.format(ep_step), \
                      '/ time step :{:>10,d}'.format(agent.step),'/ progress :',agent.progress, \
                      '/ epsilon :{:>1.5f}'.format(agent.epsilon),'/ score :{:> 5.1f}'.format(agent.score) )
                break
    # Save model
    agent.save_model()

    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()

if __name__ == "__main__":
    main()