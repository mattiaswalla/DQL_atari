# Import the gym module
import gym

import keras
import numpy as np
import random

import time



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
    def sample_batch(self, size):
       # if(self.__len__() == self.size):
        #print(self.__len__())
        randIndex = random.sample(range(self.__len__()), size)
        randIndex.sort()
        sample = [self.data[i] for i in randIndex]
        #sample = np.random.choice(self.data, size, replace=False)
        #else:
            #subdata = self.data[0:self.end]
            #print(subdata)
            #sample = np.random.choice(subdata, size, replace=False)
        start_states = np.zeros((size, 4, 105, 80))
        actions = np.zeros((size, 4))
        rewards = np.zeros((size))
        next_states = np.zeros((size, 4, 105, 80))
        is_terminal = []
        index = 0
        #print(randIndex)
        for x in sample:
            #print(sample)
            start_states[index] = x[0]
            actions[index][x[1]] = 1
            next_states[index] = x[2]
            rewards[index] = x[3]
            is_terminal.append(x[4]) 
            index += 1
        return (start_states, actions, rewards, next_states, np.array(is_terminal))

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


def transform_reward(reward):
    return np.sign(reward)


def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
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
        epochs=1, batch_size=len(start_states), verbose=0
    )


def atari_model(n_actions):
    # We assume a theano backend here, so the "channels" are first.
    ATARI_SHAPE = (4, 105, 80)

    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0, output_shape=ATARI_SHAPE)(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.Conv2D(16, (8, 8),
                                 strides=(4, 4), activation='relu'
                                 )(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.Conv2D(32, (4, 4),
                                 strides=(2, 2), activation='relu'
                                 )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.Flatten()(conv_2)

    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.multiply([output, actions_input])

    model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    return model


def get_epsilon_for_iteration(iteration):
    if iteration < 4:
        return 1
    else:
        return 0.1

def q_iteration(env, model, state, iteration, memory):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)



    # Choose the action
    if random.random() < epsilon:
        action = env.action_space.sample()

    else:
        action = choose_best_action(model, state)


    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    #print(action)
    frames = np.zeros((1, 4, 105, 80))
    reward_sum = 0
    reset = False
    for i in range(4):
        new_frame, reward, is_done, _ = env.step(action)
        reward_sum += reward
        reset = is_done or reset
        new_frame = preprocess(new_frame)
        frames[0][i] = new_frame

    actions = np.ones((1, 4))
    # print(actions)
    new_state = frames

    memory.append((state, action, frames, reward_sum, reset))
    #Sample and fit
    if (iteration > buffer_size):
        batch = memory.sample_batch(32)

        #print(iteration)
        fit_batch(model, 0.99, batch[0], batch[1], batch[2], batch[3], batch[4])

    if (iteration % 1000 == 0):
        print("saved:")
        print(iteration)
        model.save("model_weights_2018_05_09.HDF5")
    return (new_state, reset);

def choose_best_action(model, state):

    output = model.predict([state, np.ones((1, 4))], batch_size=None, verbose=0, steps=None)
    #print(output)
    return output.argmax()


buffer_size = 50000

def main():
    #model = atari_model(4)
    model = keras.models.load_model("model_weights_2018_05_09.HDF5")
    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4')
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    env.render()

    frame_list = []
    #frames = np.zeros(4, 105, 80)
    i = 0
    memoryBuffer = RingBuf(buffer_size)
    is_done = False

    action = env.action_space.sample()
    frame, reward, is_done, _ = env.step(action)

    frames = np.zeros((1, 4,105,80))
    frame = preprocess(frame)
    frames[0][0] = frame
    actions = np.ones((1,4))
    #print(actions)
    state = frames

    while True:



        env.render()

        (state, is_done) = q_iteration(env, model, state, i, memoryBuffer)
        i += 1

        # Perform a random action, returns the new frame, reward and whether the game is over
        #frame, reward, is_done, _ = env.step(env.action_space.sample())
        if (is_done):
            env.reset()
            print("reset")
            is_done = False


main()
