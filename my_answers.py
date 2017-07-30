import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):

    # initialize arrays
    X = []
    y = []
    
    # split "series" array into window_size chunks
    # and create output array
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
          
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    print(X.shape)
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    # build 5-layer LSTM model with input size = window size, ouput to 1 value
    # using hyperbolic tangent activation (so output is between -1 and 1)
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1, activation='tanh'))
    return model
    

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import re
    
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    # replace non-English characters
    text = text.replace('à', 'a')
    text = text.replace('â', 'a')
    text = text.replace('è', 'e')
    text = text.replace('é', 'e')
    
    # use regular expression to keep only a-z characters and punctuation listed above
    text = re.sub('[^a-z!,.:;?]', ' ', text)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # step through text and to create input arrays and output value
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i : i + window_size])
        outputs.append(text[i + window_size])
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    
    # build 200 layer LSTM model using 2D input arrary (window size, number of characters)
    # since it is a classfication problem, output using softmax to get probability
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model
    
