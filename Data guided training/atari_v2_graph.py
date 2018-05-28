# Import the gym module
import gym
import numpy as np
import random
import time
import pickle
from PIL import Image

from threading import Thread

import plotly as py
import plotly.graph_objs as go

Q_values = pickle.load(open("saved_q_values.p", "rb"))

y_val = np.zeros(len(Q_values))

for i in range (len(Q_values)) :
    y_val[i] = np.amax(Q_values[i])

x_val = np.array( range(5000, 3500000, 5000))


trace = go.Scatter(
    x=x_val,
    y=y_val
)

data = [trace]

py.offline.plot(data)
