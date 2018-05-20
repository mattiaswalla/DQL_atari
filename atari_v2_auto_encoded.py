# Import the gym module
import gym

import keras
import numpy as np
import random
import tensorflow as tf
import time
from keras import backend as K
import pickle
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
from threading import Thread

K.set_image_dim_ordering('th')

#class trainThread (Thread):
#    def __init__(self, threadID, name):
#        Thread.__init__(self)
#        self.threadID = threadID
#        self.name = name
#
#
#    def set_args(self, args):
#        self.args = args
#    def run(self):
#        fit_batch(self.args[0], self.args[1], self.args[2], self.args[3], self.args[4], self.args[5], self.args[6], )


class RingBuf:
    def __init__(self, size):
        self.size = size
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    def reward_batch(self):
        batch = []
        for i in range(self.__len__()):
            if(self.data[i][2] > 0):
                i
                for x in range (i-REWARD_FRAME_SIZE,i):
                    batch.insert(0, x%self.__len__() )
            if(len(batch) > BATCH_SIZE * REWARD_FRAME_SIZE):
                break

                
        size = len(batch)
        start_states = np.zeros((size, numFrames, 104, 80))
        actions = np.zeros((size, NUM_ACTIONS))
        rewards = np.zeros((size))
        next_states = np.zeros((size, numFrames, 104, 80))
        is_terminal = []
        index = 0
        
        for x in batch :
            for i in range (numFrames) :
                start_states[index][numFrames-1-i] = self.data[(x-i-1)%self.__len__()][1]
                next_states[index][numFrames-1-i] = self.data[(x-i)%self.__len__()][1]
            actions[index][self.data[x][0]] = 1
            rewards[index] = self.data[x][2]
            is_terminal.append(self.data[x][3])
            index += 1
        return (start_states, actions, rewards, next_states, np.array(is_terminal))
        
    def sample_batch(self, size):
       # if(self.__len__() == self.size):
        #print(self.__len__())
        randIndex = random.sample(range(self.__len__()), size)
        randIndex.sort()

        #sample = np.random.choice(self.data, size, replace=False)
        #else:
            #subdata = self.data[0:self.end]
            #print(subdata)
            #sample = np.random.choice(subdata, size, replace=False)
        start_states = np.zeros((size, numFrames, 104, 80))
        actions = np.zeros((size, NUM_ACTIONS))
        rewards = np.zeros((size))
        next_states = np.zeros((size, numFrames, 104, 80))
        is_terminal = []
        index = 0
        #print(randIndex)
        for x in randIndex:
            #print(x, " ", index, " ", self.data[x][0])
            for i in range (numFrames) :
                start_states[index][numFrames-1-i] = self.data[(x-i-1)%self.__len__()][1]
                next_states[index][numFrames-1-i] = self.data[(x-i)%self.__len__()][1]
            actions[index][self.data[x][0]] = 1
            rewards[index] = self.data[x][2]
            is_terminal.append(self.data[x][3])
            index += 1
        return (start_states, actions, rewards, next_states, np.array(is_terminal))

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img[0:208,:,:]))


def transform_reward(reward):
    return np.sign(reward)


def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal, epochs_n):
    """Do one deep Q learning iteration.

    Params:
    - model: The DQN
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal

    """
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    max_q = np.max(next_Q_values, axis=1)
    #print(rewards)
    Q_values = rewards + gamma * max_q
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    model.fit(
        [start_states, actions], actions * Q_values[:, None],
        epochs=epochs_n, batch_size=len(start_states), verbose=0
    )

    
def pre_train(model, frames, epochs_n):
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.

    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    # print(frames.shape)
    model.fit(
        frames, frames,
        epochs=epochs_n, batch_size=len(frames), verbose=0
    )

def pretraining_model():
    

    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0, output_shape=ATARI_SHAPE)(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.Conv2D(
        16, (8, 8), activation='relu', padding='same'
    )(normalized)

    pooling_1 = keras.layers.MaxPooling2D(pool_size=(4, 4))(conv_1)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.Conv2D(
        32, (4, 4), activation='relu', padding='same'
    )(pooling_1)    



    # Autoencoding
    decoded_1 = keras.layers.Conv2D(
        32, (4, 4), activation='relu', padding='same'
    )(conv_2)

    up_1 = keras.layers.UpSampling2D((4,4))(decoded_1)
    
    decoded_2 = keras.layers.Conv2D(
        numFrames, (8, 8), activation='sigmoid', padding='same'
    )(up_1)
    
    unnormalized = keras.layers.Lambda(lambda x: x*255.0, output_shape = ATARI_SHAPE)(decoded_2)
    model = keras.models.Model(frames_input, unnormalized)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    return model

def atari_model( n_actions):

    
    
    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0, output_shape=ATARI_SHAPE)(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.Conv2D(
        16, (8, 8),  activation='relu'
    )(normalized)    
    
    pooling_1 = keras.layers.MaxPooling2D(pool_size=(4, 4))(conv_1)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.Conv2D(
        32, (4, 4), activation='relu'
    )(pooling_1)
    pooling_2 = keras.layers.MaxPooling2D(pool_size = (2, 2))(conv_2)

    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.Flatten()(pooling_2)

    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.multiply([output, actions_input])

    model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    return model


def get_epsilon_for_iteration(iteration):    
    if iteration < LIMIT_1 :
        return 1
    elif iteration < LIMIT_2:
        return 1 +(VALUE - 1)*(iteration -LIMIT_1)/(LIMIT_2 - LIMIT_1)
    else:
        return VALUE

def q_iteration(env, model, state, iteration, memory, model_pre):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)



    # Choose the action
    if random.random() < epsilon:
        action = env.action_space.sample()

    else:
        action = choose_best_action(model, state)


    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    #print(action)




    new_frame, reward, is_done, _ = env.step(action)
    new_frame = preprocess(new_frame)
    memory.append((action, new_frame, reward, is_done))
    
    for i in range(numFrames-1):
        state[0][i-1] = state[0][i+1]
    state[0][numFrames - 1] = new_frame


    #Sample and fit

    if (iteration > BATCH_SIZE):
        batch = memory.sample_batch(BATCH_SIZE)
        fit_batch(model, 0.99, batch[0], batch[1], batch[2], batch[3], batch[4], EPOCH)

    return (state, is_done);

def choose_best_action(model, state):

    output = model.predict([state, np.ones((1, 4))], batch_size=None, verbose=0, steps=None)
    #print(output)
    return output.argmax()


buffer_size = 1000000
#thread = trainThread(1, "Thread-1")
numFrames = 3
NUM_ACTIONS = 4
BATCH_SIZE = 32
PRE_BATCH_SIZE = 2
PRE_EPOCH = 8 
EPOCH = BATCH_SIZE//32
REWARD_BATCH_EPOCH = 8
PRETRAINING_TIMES = 0
REWARD_FRAME_SIZE = 32

# We assume a theano backend here, so the "channels" are first.
ATARI_SHAPE = (numFrames, 104, 80)

#epsilon
LIMIT_1 = 100000 #epsilon = 1 untill this number
LIMIT_2 = 1000000 #after this number epsilon = value
VALUE = 0.1




def main():
    config = tf.ConfigProto(intra_op_parallelism_threads=4, \
                            inter_op_parallelism_threads=4, allow_soft_placement=True, \
                            device_count={'CPU': 1, 'GPU': 1})
    session = tf.Session(config=config)
    K.set_session(session)

    model = atari_model(NUM_ACTIONS)  
    model_pre = pretraining_model()
    print (model_pre.summary())
    #model = keras.models.load_model("model_weights_2018_05_09_V3.HDF5")
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
    memoryBuffer = pickle.load( open( "save_random.p", "rb" ) )
    is_done = False

    action = env.action_space.sample()
    frame, reward, is_done, _ = env.step(action)

    frames = np.zeros((1, numFrames,104,80))
    frame = preprocess(frame)
    frames[0][0] = frame
    actions = np.ones((1,NUM_ACTIONS))
    #print(actions)
    state = frames
    benchmark = 0
        
    for _ in range(memoryBuffer.__len__()):
        batch = memoryBuffer.sample_batch(PRE_BATCH_SIZE)
        pre_train(model_pre, batch[0], PRE_EPOCH)

    batch = memoryBuffer.sample_batch(BATCH_SIZE)
    print(model_pre.evaluate(batch[0],batch[0], BATCH_SIZE))
    
    weights_1 = model_pre.get_layer("conv2d_3").get_weights()
    weights_2 = model_pre.get_layer("conv2d_4").get_weights()            
    model.get_layer("conv2d_1").set_weights(weights_1)
    model.get_layer("conv2d_2").set_weights(weights_2)
    
    for _ in range(memoryBuffer.__len__()):
        batch = memoryBuffer.sample_batch(BATCH_SIZE)
        fit_batch(model, 0.99, batch[0], batch[1], batch[2], batch[3], batch[4], EPOCH)
        


    while i<= 1000000:

        env.render()

        (state, is_done) = q_iteration(env, model, state, i, memoryBuffer, model_pre)
        i += 1

        # Perform a random action, returns the new frame, reward and whether the game is over
        #frame, reward, is_done, _ = env.step(env.action_space.sample())
          
            
        if i % 5000 ==0 :
            print("saved:")
            print(i)
            print(benchmark)
            model.save("model_weights_2018_05_16_Guided_just_pretrain.HDF5")
            benchmark = 0
            batch = memoryBuffer.reward_batch()
            if( len(batch[0]) >0 and i < LIMIT_2): 
                fit_batch(model, 0.99, batch[0], batch[1], batch[2], batch[3], batch[4], REWARD_FRAME_SIZE )
                
            
        if (is_done):
            env.reset()
            benchmark += 1
            is_done = False


main()
