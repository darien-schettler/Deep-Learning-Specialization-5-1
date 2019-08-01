import numpy as np
from rnn_utils import *

"""

Recurrent Neural Networks (RNN) are very effective for Natural Language Processing (NLP) and other sequence tasks
    -- This is because they have "memory"
    -- They can read inputs  x<t> (such as words) one at a time
 -- Remember some information through the hidden layer activations that get passed from one time-step to the next
    -- This allows a uni-directional RNN to take information from the past to process later inputs

A bi-direction RNN can take context from both the past and the future

Here's how you can implement an RNN:

    Steps:
        
        1. Implement the calculations needed for one time-step of the RNN
        2. Implement a loop over  Tx  time-steps in order to process all the inputs, one at a time

A Recurrent neural network can be seen as the repetition of a single cell
You are first going to implement the computations for a single time-step

                                        --- Basic RNN Cell --- 
        --- Input:      x⟨t⟩  (current input) and  a⟨t−1⟩ (previous hidden state containing information from the past)
        --- Output:     a⟨t⟩  which is given to the next RNN cell and also used to predict  y⟨t⟩
        
Instructions:

1. Compute:   Hidden state with tanh activation:  a⟨t⟩ = tanh( Waa * a⟨t−1⟩ + Wax * x⟨t⟩ + ba )

2. Use:       New hidden state   a⟨t⟩,  compute the prediction  ŷ⟨t⟩ = softmax( Wya * a⟨t⟩ + by )
                    --- We provided you a function: <<<softmax>>>> in rnn_utils

3. Store:     a⟨t⟩, a⟨t−1⟩, x⟨t⟩, parameters  --->  in cache

4. Return:    a⟨t⟩,  y⟨t⟩  &  cache


    We will vectorize over  mm  examples
        --- Thus,  x⟨t⟩  will have dimension   (nx,m),  and  a⟨t⟩  will have dimension  (na,m)

"""


def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """

    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # compute next activation state using the formula given above
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


np.random.seed(1)
xt = np.random.randn(3, 10)
a_prev = np.random.randn(5, 10)
Waa = np.random.randn(5, 5)
Wax = np.random.randn(5, 3)
Wya = np.random.randn(2, 5)
ba = np.random.randn(5, 1)
by = np.random.randn(2, 1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
print("\n\na_next[4] = \n", a_next[4])
print("\na_next.shape = \n", a_next.shape)
print("\nyt_pred[1] = \n", yt_pred[1])
print("\nyt_pred.shape = \n", yt_pred.shape)

'''
RNN FORWARD PASS

- You can see an RNN as the repetition of the cell you've just built
- If your input sequence of data is carried over 10 time steps, then you will copy the RNN cell 10 times
- Each cell takes as input the hidden state from the previous cell a⟨t−1⟩ and the current time-step's input data x⟨t⟩
- It outputs a hidden state ( a⟨t⟩a⟨t⟩ ) and a prediction ( y⟨t⟩y⟨t⟩ ) for this time-step.

Instructions for Forward Propagation:

1. Create a vector of zeros ( aa ) that will store all the hidden states computed by the RNN.

2. Initialize the "next" hidden state as  a0a0  (initial hidden state).

3. Start looping over each time step, your incremental index is  tt  :
    -- Update the "next" hidden state and the cache by running rnn_cell_forward
    -- Store the "next" hidden state in  aa  ( tthtth  position)
    -- Store the prediction in y
    -- Add the cache to the list of caches

4. Return  a ,  y  and caches

'''


def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """

    # Initialize "caches" which will contain the list of all caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # initialize "a" and "y" with zeros
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])

    # Initialize a_next
    a_next = a0

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:, :, t] = a_next
        # Save the value of the prediction in y
        y_pred[:, :, t] = yt_pred
        # Append "cache" to "caches"
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y_pred, caches


np.random.seed(1)
x = np.random.randn(3, 10, 4)
a0 = np.random.randn(5, 10)
Waa = np.random.randn(5, 5)
Wax = np.random.randn(5, 3)
Wya = np.random.randn(2, 5)
ba = np.random.randn(5, 1)
by = np.random.randn(2, 1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
print("\n\na[4][1] = \n", a[4][1])
print("\na.shape = \n", a.shape)
print("\ny_pred[1][3] \n=", y_pred[1][3])
print("\ny_pred.shape = \n", y_pred.shape)
print("\ncaches[1][1][3] = \n", caches[1][1][3])
print("\nlen(caches) = \n", len(caches))

'''

In the next part, you will build a more complex LSTM model, which is better at addressing vanishing gradients

The LSTM will be better able to remember a piece of information and keep it saved for many timesteps

    --- LSTM-Cell ---
        - This tracks and updates a "cell state" or memory variable  c⟨t⟩  at every time-step
            --> This can be different from  a⟨t⟩

Similar to the RNN example above, you will start by implementing the LSTM cell for a single time-step
    - Then you can iteratively call it from inside a for-loop to have it process an input with  Tx  time-steps

                                                --- ABOUT THE GATES ---
--- Forget gate
    - Lets assume we are reading words in a piece of text
    - We want use an LSTM to keep track of grammatical structures (such as whether the subject is singular or plural)
    - If it changes from singular to plural, we need a way to get rid of the previously stored memory value 
        --> The memory 'remembers' the singular/plural state
    - In an LSTM, the forget gate lets us do this:

                                                Γ[f]⟨t⟩ = σ(Wf[a⟨t−1⟩, x⟨t⟩] + bf)
                Here:
                    ~~~ Wf are weights that govern the forget gate's behavior                                       ~~~
                    ~~~ We concatenate  [a⟨t−1⟩,x⟨t⟩] and multiply by  Wf                                             ~~~
                    ~~~ The equation above results in a vector  Γ[f]⟨t⟩  with values between 0 and 1                 ~~~
                    ~~~ This forget gate vector will be multiplied element-wise by the previous cell state  c⟨t−1⟩   ~~~
                    ~~~ So if one of the values of  Γ[f]⟨t⟩  is 0 (or close to 0)...                                 ~~~
                        *** That means that the LSTM should remove that piece of information                        ***
                        *** (e.g. the singular subject) in the corresponding component of  c⟨t−1⟩                    ***
                    ~~~ If one of the values is 1 (or close to 1)...                                                ~~~
                        *** then it will keep the information                                                       ***


--- Update gate
    - Once we forget that the subject being discussed is singular, we need to find a way to update it
        --> We want to update it to reflect that the new subject is now plural
    - Here is the formula for the update gate:

                                               Γ[u]⟨t⟩ = σ(Wu[a⟨t−1⟩, x{t}] + bu)
 
                Here:
                    ~~~ Γ[u]⟨t⟩  is again a vector of values between 0 and 1                                         ~~~
                    ~~~ Γ[u]⟨t⟩ will be multiplied element-wise with  c~⟨t⟩, in order to compute  c⟨t⟩                 ~~~


--- Updating the cell
    - To update the new subject we need to create a new vector of numbers that we can add to our previous cell state
    - The equation we use is:

                                              c~⟨t⟩ = tanh(Wc[a⟨t−1⟩, x⟨t⟩] + bc)
 
                Finally, the new cell state is:

                                            c⟨t⟩ = Γ[f]⟨t⟩ ∗ c⟨t−1⟩ + Γ[u]⟨t⟩ ∗ c~⟨t⟩

--- Output gate
    - To decide which outputs we will use, we will use the following two formulas:

                                            1. Γ[o]⟨t⟩ = σ(Wo[a⟨t−1⟩, x⟨t⟩] + bo)
                                            
                                            2. a⟨t⟩ = Γ[o]⟨t⟩ ∗ tanh(c⟨t⟩)
 
                Where:
                    ~~~ In Eqn 1. above you decide what to output using a sigmoid function                          ~~~
                    ~~~ In Eqn 2. above you multiply that by the  tanh  of the previous state                       ~~~    



Exercise: Implement the LSTM cell described above...

Instructions:

1. Concatenate  a⟨t−1⟩  and  x⟨t⟩  in a single matrix:  
    --> concat=[a⟨t−1⟩x⟨t⟩]concat=[a⟨t−1⟩x⟨t⟩] 

2. Compute all the formulas above
    --> You can use sigmoid() (provided) and np.tanh()

3. Compute the prediction  y⟨t⟩
    --> You can use softmax() (provided)

'''


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.concatenate((a_prev, xt), axis=0)

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = ft*c_prev + it*cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
print("\n\na_next[4] = \n", a_next[4])
print("\na_next.shape = \n", c_next.shape)
print("\nc_next[2] = \n", c_next[2])
print("\nc_next.shape = \n", c_next.shape)
print("\nyt[1] = \n", yt[1])
print("\nyt.shape = \n", yt.shape)
print("\ncache[1][3] = \n", cache[1][3])
print("\nlen(cache) = \n", len(cache))

'''
Now having implemented one step of an LSTM, you can iterate over a for-loop to process a sequence of  Tx  inputs
'''


def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros([n_a, m, T_x])
    c = np.zeros(a.shape)
    y = np.zeros([n_y, m, T_x])

    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros([n_a, m])

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:, :, t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y[:, :, t] = yt
        # Save the value of the next cell state (≈1 line)
        c[:, :, t] = c_next
        # Append the cache into caches (≈1 line)
        caches.append(cache)    # store values needed for backward propagation in cache

    caches = (caches, x)

    return a, y, c, caches


np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)
print("\n\na[4][3][6] = \n", a[4][3][6])
print("\na.shape = \n", a.shape)
print("\ny[1][4][3] = \n", y[1][4][3])
print("\ny.shape = \n", y.shape)
print("\ncaches[1][1[1]] = \n", caches[1][1][1])
print("\nc[1][2][1] = \n", c[1][2][1])
print("\nlen(caches) = \n", len(caches))

'''
                                        --- BACK-PROPAGATION IN RNN ---
In modern DL frameworks, you only have to implement the forward pass, & the framework takes care of the rest (BACK-PROP)
    - This means most deep learning engineers do not need to bother with the details of the backward pass
    - If however you are an expert in calculus and want to see the details of back-prop in RNNs, continue below...
    

When implementing a simple (FC) NN, back-prop is used to compute the derivatives w.r.t the cost to update the parameters
Similarly, in RNNs you can to calculate the derivatives with respect to the cost in order to update the parameters

The backprop equations are quite complicated and we did not derive them in lecture

However, if you wish to see them there are images that contain the formulas and they will be used in the code below...

Just like in a FC NN, the derivative of the cost function  J  back-props through the RNN using the chain-rule (calculus)
The chain-rule is also used to calculate:  
    -- (∂J∂Wax,∂J∂Waa,∂J∂b)  
    -- to update the parameters  ---> (Wax,Waa,ba)

To compute the rnn_cell_backward you need to compute the following equations:
It is a good exercise to derive them by hand

The derivative of  tanh  is  1−tanh(x)^2                NOTE --> sech(x)^2 = 1 − tanh(x)^2

Similarly for  ∂a⟨t⟩ / ∂Wax, ∂a⟨t⟩ / ∂Waa, ∂a⟨t⟩/∂b, the derivative of  tanh(u)  is  (1−tanh(u)^2)du

The final two equations also follow same rule and are derived using the  tanh  derivative

Note that the arrangement is done in a way to get the same dimensions to match

'''

