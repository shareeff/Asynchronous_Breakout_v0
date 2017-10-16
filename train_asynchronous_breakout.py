#!/usr/bin/env python
import threading
import tensorflow as tf
import cv2
import sys
import random
import numpy as np
import time
import gym

#Shared global parameters
#env = gym.make('Breakout-v0').unwrapped
TMAX = 500000000
T = 0
It = 10000
Iasync = 32
THREADS = 12
WISHED_SCORE = 100

GAME = 'Breakout-v0' # The name of the game being played for log files
KEYMAP = {'Breakout-v0': [1, 4, 5]}
ACTIONS = len(KEYMAP[GAME])# Number of valid actions
GAMMA = 0.99 # Decay rate of past observations
OBSERVE = 5. # Timesteps to observe before training
EXPLORE = 400000. # Frames over which to anneal epsilon
FINAL_EPSILONS = [0.01, 0.01, 0.05] # Final values of epsilon
INITIAL_EPSILONS = [0.4, 0.3, 0.3] # Starting values of epsilon
EPSILONS = 3

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def createNetwork(s):
    # input layer

    # network weights
    with tf.name_scope('conv_layer_1'):
        with tf.name_scope('W_conv1'):
            W_conv1 = weight_variable([8, 8, 4, 32])
        with tf.name_scope('b_conv1'):
            b_conv1 = bias_variable([32])
        with tf.name_scope('h_conv1'):
            h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
        with tf.name_scope('h_pool1'):
            h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv_layer_2'):
        with tf.name_scope('W_conv2'):
            W_conv2 = weight_variable([4, 4, 32, 64])
        with tf.name_scope('b_conv2'):
            b_conv2 = bias_variable([64])
        with tf.name_scope('h_conv2'):
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    with tf.name_scope('conv_layer_3'):
        with tf.name_scope('W_conv3'):
            W_conv3 = weight_variable([3, 3, 64, 64])
        with tf.name_scope('b_conv3'):
            b_conv3 = bias_variable([64])
        with tf.name_scope('h_conv3'):
            h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
        with tf.name_scope('h_pool3_flat'):
            h_pool3_flat = tf.reshape(h_conv3, [-1, 1600])

    with tf.name_scope("fc_layer_1"):
        with tf.name_scope('W_fc1'):
            W_fc1 = weight_variable([1600, 512])
        with tf.name_scope('b_fc1'):
            b_fc1 = bias_variable([512])
        with tf.name_scope('h_fc1'):
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
            variable_summaries(h_fc1)

    with tf.name_scope('fc_layer_2'):
        with tf.name_scope('W_fc2'):
            W_fc2 = weight_variable([512, ACTIONS])
        with tf.name_scope('b_fc2'):
            b_fc2 = bias_variable([ACTIONS])
        with tf.name_scope('readout'):
            # readout layer
            readout = tf.matmul(h_fc1, W_fc2) + b_fc2
            variable_summaries(readout)

    return readout, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

def copyTargetNetwork(sess):
    sess.run(copy_Otarget)

def actorLearner(num, sess, copy_lock, save_lock, env):
    # We use global shared O parameter vector
    # We use global shared Otarget parameter vector
    # We use global shared counter T, and TMAX constant
    global TMAX, T

    # Open up a game state to communicate with emulator
    #lock.acquire()
    game_state = env.reset()
    #lock.release()

    # Initialize network gradients
    s_j_batch = []
    a_batch = []
    y_batch = []

    # Get the first state by doing nothing and preprocess the image to 80x80x4
    x_t = cv2.cvtColor(cv2.resize(game_state, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    aux_s = s_t

    time.sleep(3*num)

    # Initialize target network weights
    copy_lock.acquire()
    copyTargetNetwork(sess)
    copy_lock.release()

    epsilon_index = random.randrange(EPSILONS)
    INITIAL_EPSILON = INITIAL_EPSILONS[epsilon_index]
    FINAL_EPSILON =  FINAL_EPSILONS[epsilon_index]
    epsilon = INITIAL_EPSILON

    print("THREAD ", num, "STARTING...", "EXPLORATION POLICY => INITIAL_EPSILON:", INITIAL_EPSILON, ", FINAL_EPSILON:", FINAL_EPSILON)

    # Initialize thread step counter
    t = 0
    score = 0
    while T < TMAX and score < WISHED_SCORE:

        # Choose an action epsilon greedily
        readout_t = O_readout.eval(session = sess, feed_dict = {s : [s_t]})
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
            a_t_env = action_index
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
            a_t_env = action_index

        # Scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # Run the selected action and observe next state and reward
        x_t1_col, r_t, terminal, info = env.step(KEYMAP[GAME][action_index])
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        aux_s = np.delete(s_t, 0, axis = 2)
        s_t1 = np.append(aux_s, x_t1, axis = 2)

        # Accumulate gradients
        readout_j1 = Ot_readout.eval(session = sess, feed_dict = {st : [s_t1]})
        if terminal:
            y_batch.append(r_t)
        else:
            y_batch.append(r_t + GAMMA * np.max(readout_j1))

        a_batch.append(a_t)
        s_j_batch.append(s_t)

        # Update the old values
        s_t = s_t1
        T += 1
        t += 1
        score += r_t

        # Update the Otarget network
        if T % It == 0:
            copy_lock.acquire()
            copyTargetNetwork(sess)
            copy_lock.release()

        # Update the O network
        if t % Iasync == 0 or terminal:
            if s_j_batch:
                # Perform asynchronous update of O network
                train_O.run(session = sess, feed_dict = {
        	           y : y_batch,
        	           a : a_batch,
        	           s : s_j_batch})

            #Clear gradients
            s_j_batch = []
            a_batch = []
            y_batch = []

        if t % 100 == 0:
            if s_j_batch:
                summary = sess.run(merged, feed_dict={
                    y: y_batch,
                    a: a_batch,
                    s: s_j_batch,
                    st: s_j_batch})
                train_writer.add_summary(summary, t)

        # Save progress every 5000 iterations
        if t % 5000 == 0:
            save_lock.acquire()
            saver.save(sess, 'save_networks_asyn/' + GAME + '-dqn', global_step = t)
            save_lock.release()

        # Print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if terminal:
            print("THREAD:", num, "/ TIME", T, "/ TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", KEYMAP[GAME][action_index], "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t), "/ SCORE", score)
            score = 0
            env.reset()


# We create the shared global networks
# O network
with tf.name_scope("input_layer_0_net"):
    s = tf.placeholder("float", [None, 80, 80, 4])
O_readout, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2 = createNetwork(s)

# Training node
with tf.name_scope('action'):
    a = tf.placeholder("float", [None, ACTIONS])
with tf.name_scope('label'):
    y = tf.placeholder("float", [None])
with tf.name_scope('cross_entropy'):
    O_readout_action = tf.reduce_sum(tf.multiply(O_readout, a), reduction_indices=1)
    with tf.name_scope('total_cost'):
        cost_O = tf.reduce_mean(tf.square(y - O_readout_action))

tf.summary.scalar('cross_entropy', cost_O)
with tf.name_scope('train'):
    #train_O = tf.train.RMSPropOptimizer(0.00025, 0.95, 0.95, 0.01).minimize(cost_O)
    train_O = tf.train.AdamOptimizer(0.00001).minimize(cost_O)       #.00001

# Otarget network
with tf.name_scope("input_layer_target_net"):
    st = tf.placeholder("float", [None, 80, 80, 4])

Ot_readout, W_conv1t, b_conv1t, W_conv2t, b_conv2t, W_conv3t, b_conv3t, W_fc1t, b_fc1t, W_fc2t, b_fc2t = createNetwork(st)
copy_Otarget = [W_conv1t.assign(W_conv1), b_conv1t.assign(b_conv1), W_conv2t.assign(W_conv2), b_conv2t.assign(b_conv2), W_conv3t.assign(W_conv3), b_conv3t.assign(b_conv3), W_fc1t.assign(W_fc1), b_fc1t.assign(b_fc1), W_fc2t.assign(W_fc2), b_fc2t.assign(b_fc2)]

# Initialize session and variables
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
sess = tf.InteractiveSession(config=config)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('summary_log' + '/train',
                                      sess.graph)


saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
checkpoint = tf.train.get_checkpoint_state("./save_networks_asyn")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)

if __name__ == "__main__":
    # Start n concurrent actor threads
    #lock = threading.Lock()
    envs = [gym.make(GAME) for _ in range(THREADS)]
    copy_lock = threading.Lock()
    save_lock = threading.Lock()
    threads = list()
    for i in range(THREADS):
        t = threading.Thread(target=actorLearner, args=(i,sess, copy_lock, save_lock, envs[i]))
        threads.append(t)

    # Start all threads
    for x in threads:
        x.start()

    # Wait for all of them to finish
    for x in threads:
        x.join()

    print("ALL DONE!!")
