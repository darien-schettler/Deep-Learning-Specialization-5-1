import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

"""

You would like to create a jazz music piece specially for a friend's birthday
However, you don't know any instruments or music composition

Fortunately, you know deep learning and will solve this problem using an LSTM-netwok

You will train a network to generate novel jazz solos in a style representative of a body of performed work

DATASET
    - You will train your algorithm on a corpus of Jazz music
    - Run the cell below to listen to a snippet of the audio from the training set:

We have taken care of the preprocessing of the musical data to render it in terms of musical "values."
You can informally think of each "value" as a note, which comprises a pitch and a duration

For example, if you press down a specific piano key for 0.5 seconds, then you have just played a note
    - In music theory, a "value" is actually more complicated than this
        --> It also captures the information needed to play multiple notes at the same time
        --> i.e. When playing a music piece, you might press down two piano keys at the same time 
            ---- Playng multiple notes at the same time generates what's called a "chord"
            ---- But we don't need to worry about the details of music theory for this assignment
            ---- We will obtain a dataset of values, and will learn an RNN model to generate sequences of values
    - Our music generation system will use 78 unique values


You have just loaded the following:


X: This is an (m,  Tx, 78) dimensional array
    - We have m training examples, each of which is a snippet of  Tx=30  musical values
    - At each time step, the input is one of 78 different possible values, represented as a one-hot vector
    - Thus for example, X[i,t,:] is a one-hot vector representing the value of the i-th example at time t

Y: This is essentially the same as X, but shifted one step to the left (to the past)
    - We're interested in the network using the previous values to predict the next value
    - Our sequence model will try to predict  y⟨t⟩  given  x⟨1⟩,…,x⟨t⟩
    - However, the data in Y is reordered to be dimension  (Ty,m,78), where  Ty=Tx
    - This format makes it more convenient to feed to the LSTM later

n_values: The number of unique values in this dataset. This should b e 78

indices_values: python dictionary mapping from 0-77 to musical values
    
"""

X, Y, n_values, indices_values = load_music_utils()
print('\nShape of X:', X.shape)
print('\nNumber of Training Examples:', X.shape[0])
print('\nTx (Length of Sequence):', X.shape[1])
print('\nTotal # of Unique Values:', n_values)
print('\nShape of Y:', Y.shape)

'''

We will be training the model on random snippets of 30 values taken from a much longer piece of music
Thus, we won't bother to set the first input  x⟨1⟩=0⃗, which we had done previously
    - Since now most of these snippets of audio start somewhere in the middle of a piece of music
    - We are setting each of the snippets to have the same length  Tx=30  to make vectorization easier

BUILDING THE MODEL

In this part you will build and train a model that will learn musical patterns
To do so, you will need to build a model that takes in X of shape  (m,Tx,78)  and Y of shape  (Ty,m,78)
We will use an LSTM with 64 dimensional hidden states. Lets set n_a = 64

'''

n_a = 64

'''

Here's how you can create a Keras model with multiple inputs and outputs:

If you're building an RNN where even at test time entire input sequence  x⟨1⟩,x⟨2⟩,…,x⟨Tx⟩  were given in advance
    - i.e. If the inputs were words & the output was a label, then Keras has built-in functions to build the model
    - However, for sequence generation, at test time we don't know all the values of  x⟨t⟩  in advance;
        --> Instead we generate them one at a time using  x⟨t⟩ = y⟨t − 1⟩
    - Therefore the code will be more complicated...
        ---> We need to implement a for-loop to iterate over the different time steps

The function djmodel() will call the LSTM layer  Tx  times using a for-loop
-- NOTE: it is important that all  Tx  copies have the same weights
    ---> i.e. It should not re-initiaiize the weights every time --- the  Tx  steps should have shared weights
-- The key steps for implementing layers with shareable weights in Keras are:

    1. Define the layer objects (we will use global variables for this)
    2. Call these objects when propagating the input
    3. We have defined the layers objects you need as global variables. Please run the next cell to create them
    
    
Please check the Keras documentation to make sure you understand what these layers are: 

    ---->    Reshape()
        
    ---->    LSTM()
        
    ---->    Dense()

'''

reshapor = Reshape((1, 78))                        # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
densor = Dense(n_values, activation='softmax')     # Used in Step 2.D

'''

Each of reshapor, LSTM_cell and densor are now layer objects, and you can use them to implement djmodel()

In order to propagate a Keras tensor object X through one of these layers, use layer_object(X)
---> Or layer_object([X,Y]) if it requires multiple inputs
---> i.e. reshapor(X) will propagate X through the Reshape((1,78)) layer defined above


TRY IT!
--------------------
 Implement djmodel()
--------------------

You will need to carry out 2 steps:

1. Create an empty list "outputs" to save the outputs of the LSTM Cell at every time step.

2. Loop for  t∈1, …, Tx :
    
    A.  Select the "t"th time-step vector from X
        The shape of this selection should be (78,)
        To do so, create a custom Lambda layer in Keras by using this line of code:
                ---> x = Lambda(lambda x: X[:,t,:])(X)
        Making this function a Keras Layer object to apply to X.
        
        ** Look over the Keras documentation to figure out what this lambda function does...
        ** It is creating a "temporary" or "unnamed" function (that's what Lambda functions are)...
        ** This unnamed function extracts out the appropriate one-hot vector...
    
    B.  Reshape x to be (1,78)
        You may find the reshapor() layer (defined below) helpful
    
    C.  Run x through one step of LSTM_cell
        Remember to initialize the LSTM_cell with the previous step's hidden state  aa  and cell state  cc
        Use the following formatting:
            ---> a, _, c = LSTM_cell(input_x, initial_state=[previous hidden state, previous cell state])
    
    D.  Propagate the LSTM's output activation value through a   dense + softmax   layer using densor
    
    E.  Append the predicted value to the list of "outputs"

'''


def djmodel(Tx, n_a, n_values):
    """
    Implement the model

    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data

    Returns:
    model -- a keras model with the
    """

    # Define the input of your model with a shape
    X = Input(shape=(Tx, n_values))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []

    # Step 2: Loop
    for t in range(Tx):
        # Step 2.A: select the "t"th time step vector from X.
        x = Lambda(lambda x: X[:, t, :])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)

    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)

    return model


model = djmodel(Tx = 30 , n_a = 64, n_values = 78)

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=100)

'''

You should see the model loss going down
Now that you have trained a model, lets go on the the final section to implement an inference algorithm (music!)

GENERATING MUSIC

You now have a trained model which has learned the patterns of the jazz soloist

Lets now use this model to synthesize new music:

At each step of sampling, you will take as input the activation a and cell state c from the previous state of the LSTM
    - Forward propagate by one step
    - Get a new output activation as well as cell state
    - The new activation a can then be used to generate the output, using densor as before

To start off the model, we will initialize x0 as well as the LSTM activation and and cell value a0 and c0 to be zeros


Implement the function below to sample a sequence of musical values


Here are some of the key steps you'll need to implement inside the for-loop that generates the  Ty  output characters:

    Step 2.A: Use LSTM_Cell, which inputs the previous step's c and a to generate the current step's c and a

    Step 2.B: Use densor (defined previously) to compute a softmax on a to get the output for the current step

    Step 2.C: Save the output you have just generated by appending it to outputs

    Step 2.D: Sample x to the be "out"s one-hot version (the prediction) so that you can pass it to the next LSTM's step
              We have already provided this line of code, which uses a Lambda function

        x = Lambda(one_hot)(out)
    
        [  Minor technical Note : Rather than sampling a value at random according to the probabilities in 
           out, this line of code actually chooses the single most likely note at each step using an argmax  ]

'''


# GRADED FUNCTION: music_inference_model

def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate

    Returns:
    inference_model -- Keras model instance
    """

    # Define the input of your model with a shape
    x0 = Input(shape=(1, n_values))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []

    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])

        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)

        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided
        #           the line of code you need to do this.
        x = Lambda(one_hot)(out)

    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model


inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)

x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

'''

Implement predict_and_sample()

This function takes many arguments including the inputs [x_initializer, a_initializer, c_initializer]
    ---> In order to predict the output corresponding to this input, you will need to carry-out 3 steps:

        1. Use your inference model to predict an output given your set of inputs
            - The output pred should be a list of length  Ty
                --- Where each element is a numpy-array of shape (1, n_values)
        
        2. Convert pred into a numpy array of  Ty  indices
            - Each index corresponds is computed by taking the argmax of an element of the pred list
        
        3. Convert the indices into their one-hot vector representations

'''


def predict_and_sample(inference_model, x_initializer=x_initializer, a_initializer=a_initializer,
                       c_initializer=c_initializer):
    """
    Predicts the next value of values using the inference model.

    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel

    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """

    # Step 1: Use the inference model to predict an output sequence given x_initializer, a_initializer and c_initializer
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices, num_classes=78)

    return results, indices


results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("\nnp.argmax(results[12]) =\n", np.argmax(results[12]))
print("\nnp.argmax(results[17]) =\n", np.argmax(results[17]))
print("\nlist(indices[12:18]) =\n", list(indices[12:18]))

'''

Finally, you are ready to generate music. Your RNN generates a sequence of values
The following code generates music by first calling your predict_and_sample() function
These values are then post-processed into musical chords
    - meaning that multiple values or notes can be played at the same time

Most music algorithms use post-processing because it's hard to generate music that sounds good w/out post-processing
The post-processing does things such as...
    -- clean up the generated audio by making sure the same sound is not repeated too many times
    -- that two successive notes are not too far from each other in pitch, and so on
    -- One could argue that a lot of these post-processing steps are hacks
    -- Also, a lot the music generation literature has also focused on hand-crafting post-processors
    -- The output quality depends on the quality of the post-processing and not just the quality of the RNN
    -- But this post-processing does make a huge difference, so lets use it in our implementation as well

Lets make some music!

Run the following cell to generate music and record it into your out_stream

This can take a couple of minutes...

'''

out_stream = generate_music(inference_model)

IPython.display.Audio('./data/30s_trained_model.mp3')

'''

You have come to the end of the notebook

Here's what you should remember:

    1. A sequence model can be used to generate musical values, which are then post-processed into midi music
    
    2. Fairly similar models can be used to generate dinosaur names or to generate music
        --> The major difference being the input fed to the model
    
    3. In Keras, sequence generation involves defining layers with shared weights
        --> These are then repeated for the different time steps  1,…,Tx

REFERENCES

The ideas presented in this notebook came primarily from three computational music papers cited below
The implementation here also took significant inspiration and used many components from Ji-Sung Kim's github repository

Ji-Sung Kim, 2016, deepjazz
Jon Gillick, Kevin Tang and Robert Keller, 2009. Learning Jazz Grammars
Robert Keller and David Morrison, 2007, A Grammatical Approach to Automatic Improvisation
François Pachet, 1999, Surprising Harmonies

We're also grateful to François Germain for valuable feedback



'''