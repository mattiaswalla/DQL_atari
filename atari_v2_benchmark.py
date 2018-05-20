# Import the gym module
import gym

import keras
import numpy as np
import random
import tensorflow as tf
import time
from keras import backend as K
from PIL import Image

from threading import Thread

K.set_image_dim_ordering('th')


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img[0:208,:,:]))


def transform_reward(reward):
    return np.sign(reward)


def q_iteration(env, model, state, iteration):

    epsilon =1
    if random.random() < epsilon:
        action = env.action_space.sample()

    else:
        action = choose_best_action(model, state)



    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    #print(action)

    new_frame, reward, is_done, _ = env.step(action)
    new_frame = preprocess(new_frame)  
    
    for i in range(numFrames-1):
        state[0][i-1] = state[0][i+1]
    state[0][numFrames - 1] = new_frame


    #Sample and fit

    
    return (state, is_done);

def choose_best_action(model, state):

    output = model.predict([state, np.ones((1, 4))], batch_size=None, verbose=0, steps=None)
    #print(output)
    return output.argmax()


buffer_size = 500000
#thread = trainThread(1, "Thread-1")
numFrames = 2
NUM_ACTIONS = 4
BATCH_SIZE = 64
PRE_EPOCH = 8
EPOCH = BATCH_SIZE//32
REWARD_BATCH_EPOCH = 16
PRETRAINING_TIMES = 0


# We assume a theano backend here, so the "channels" are first.
ATARI_SHAPE = (numFrames, 104, 80)

#epsilon
LIMIT_1 = 100000 #epsilon = 1 untill this number
LIMIT_2 = 500000 #after this number epsilon = value
VALUE = 0.1




def main():
    config = tf.ConfigProto(intra_op_parallelism_threads=4, \
                            inter_op_parallelism_threads=4, allow_soft_placement=True, \
                            device_count={'CPU': 1, 'GPU': 1})
    session = tf.Session(config=config)
    K.set_session(session)



    model = keras.models.load_model("model_weights_2018_05_16_Guided_just_pretrain.HDF5")
    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4')
    #env = gym.make("Pong-v0")
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    env.render()

    frame_list = []
    #frames = np.zeros(4, 105, 80)
    i = 0
    is_done = False

    action = env.action_space.sample()
    frame, reward, is_done, _ = env.step(action)

    frames = np.zeros((1, numFrames,104,80))
    frame = preprocess(frame)
    frames[0][0] = frame
    actions = np.ones((1,NUM_ACTIONS))
    #print(actions)
    state = frames
    while i<= 1000000:

        

        env.render()

        (state, is_done) = q_iteration(env, model, state, i)
        i += 1

        # Perform a random action, returns the new frame, reward and whether the game is over
        #frame, reward, is_done, _ = env.step(env.action_space.sample())
                     
        time.sleep(0.1)  
        if (is_done):
            env.reset()
            is_done = False


main()
