import numpy as np
from utils import *
import random

"""

Welcome to Dinosaur Island!

65 million years ago, dinosaurs existed, and in this assignment they are back

You are in charge of a special task...
Leading biology researchers are creating new breeds of dinosaurs and bringing them to life on earth.

--- Your job is to give names to these dinosaurs ---
    --> If a dinosaur does not like its name, it might go beserk, so choose wisely!

Luckily you have learned some deep learning and you will use it to save the day
Your assistant has collected a list of all the dinosaur names they could find, and compiled them into this dataset

To create new dinosaur names, you will build a character level language model to generate new names
Your algorithm will learn the different name patterns, and randomly generate new names

Hopefully this algorithm will keep you and your team safe from the dinosaurs' wrath!

By completing this assignment you will learn:

1. How to store text data for processing using an RNN
2. How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
3. How to build a character-level text generation recurrent neural network
4. Why clipping the gradients is important

We will begin by loading in some functions that we have provided for you in rnn_utils
    --> Specifically, you have access to functions such as rnn_forward and rnn_backward

"""

data = open('dinos.txt', 'r').read()  # Reads a text file (character level not string)
data = data.lower()  # Make all text lowercase
chars = list(set(data))  # Set reduces the input to a set of all the individual components that make it up (unordered)
data_size, vocab_size = len(data), len(chars)
print('\nThere are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

'''

The characters are a-z (26 characters) plus the "\n" (or newline character)
    --> "\n" plays a role similar to the <EOS> (or "End of sentence") token
    --> only here it indicates the end of the dinosaur name rather than the end of a sentence
    
In the code below, we create a python dictionary (i.e., a hash table) to map each character to an index from 0-26
We also create a second python dictionary that maps each index back to the corresponding character character
    --> This helps find what index corresponds to what char in the probability distribution output of the softmax layer

Below, char_to_ix and ix_to_char are the python dictionaries

'''

char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}  # Dictionary -- Key  (Char) -- Val (Index)
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}  # Dictionary -- Key (Index) -- Val  (Char)

'''

OVERVIEW OF THE MODEL

Your model will have the following structure:

1. Initialize parameters

2. Run the optimization loop
    a. Forward propagation to compute the loss function
    b. Backward propagation to compute the gradients with respect to the loss function
    c. Clip the gradients to avoid exploding gradients
    d. Using the gradients, update your parameter with the gradient descent update rule
    
3. Return the learned parameters



At each time-step, the RNN tries to predict what is the next character given the previous characters
The dataset  X=(x⟨1⟩,x⟨2⟩,...,x⟨Tx⟩)  is a list of characters in the training set
The dataset  Y=(y⟨1⟩,y⟨2⟩,...,y⟨Tx⟩)  is such that at every time-step  t, we have y⟨t⟩=x⟨t+1⟩



BUILDING BLOCKS OF THE MODEL
In this part, you will build two important blocks of the overall model:

1. Gradient clipping: to avoid exploding gradients
2. Sampling: a technique used to generate characters

You will then apply these two functions to build the model

1. CLIPPING THE GRADIENTS IN THE OPTIMIZATION LOOP
In this section you will implement the clip function that you will call inside of your optimization loop. Recall that your overall loop structure usually consists of a forward pass, a cost computation, a backward pass, and a parameter update. Before updating the parameters, you will perform gradient clipping when needed to make sure that your gradients are not "exploding," meaning taking on overly large values.

Below we implement a function clip that takes in a dict of gradients & returns a clipped version of gradients if needed
There are different ways to clip gradients; 
    --> we will use a simple element-wise clipping procedure
        --> in which every element of the gradient vector is clipped to lie between some range [-N, N]
        --> More generally, you will provide a maxValue (say 10)
        --> In this example, if any component of the gradient vector is greater than 10, it would be set to 10
        --> In this example, if any component of the gradient vector is less than -10, it would be set to -10
        --> In this example, If it is between -10 and 10, it is left alone


Implement the function below to return the clipped gradients of your dictionary gradients

Your function takes in a maximum threshold and returns the clipped versions of your gradients

'''


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']

    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


np.random.seed(3)
dWax = np.random.randn(5, 3) * 10
dWaa = np.random.randn(5, 5) * 10
dWya = np.random.randn(2, 5) * 10
db = np.random.randn(5, 1) * 10
dby = np.random.randn(2, 1) * 10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, 10)
print("\ngradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])

'''
2. SAMPLING
Now assume that your model is trained. You would like to generate new text (characters)

The process of generation is known as sampling

mplement the sample function below to sample characters. You need to carry out 4 steps:

Step 1: Pass the network the first "dummy" input  x⟨1⟩=0⃗ (the vector of zeros)
    --> This is the default input before we've generated any characters
    --> We also set  a⟨0⟩=0⃗

Step 2: Run one step of forward propagation to get  a⟨1⟩  and  ŷ⟨1⟩ 

Here are the equations:

       i.   a⟨t + 1⟩ = tanh( Waxx⟨t⟩ + Waaa⟨t⟩ + b )
 
       ii.  z⟨t + 1⟩ = Wyaa⟨t + 1⟩ + by
 
       iii. ŷ⟨t + 1⟩ = softmax( z⟨t + 1⟩ )
 
    Note that  ŷ⟨t + 1⟩ is a (softmax) probability vector (its entries are between 0 and 1 and sum to 1)
        -- ŷ[i]⟨t+1⟩ represents the probability that the character indexed by "i" is the next character
        -- We have provided a softmax() function that you can use

Step 3: Carry out sampling: 
    --> Pick the next character's index according to the probability distribution specified by  ŷ ⟨t+1⟩
    --> This means that if  ŷ[i]⟨t+1⟩ = 0.16 , you will pick the index "i" with 16% probability
    --> To implement it, you can use np.random.choice

    --> Here is an example of how to use np.random.choice():
        
                np.random.seed(0)
                p = np.array([0.1, 0.0, 0.7, 0.2])
                index = np.random.choice([0, 1, 2, 3], p = p.ravel())
                
                This means that you will pick the index according to the distribution: 
                           - P(index = 0) = 0.1
                           - P(index = 1) = 0.0
                           - P(index = 2) = 0.7
                           - P(index = 3) = 0.2
                           - P(index = 0) = 0.1
                           - P(index = 1) = 0.0
                           - P(index = 2) = 0.7
                           - P(index = 3) = 0.2
                
Step 4: The last step to implement in sample() is to overwrite the variable x
    --> x is currently stored in  x⟨t⟩ , with the value of  x⟨t+1⟩
    --> You will represent  x⟨t+1⟩ by creating a one-hot vector corresponding to the char you've chosen as the prediction
    --> You will then forward propagate  x⟨t+1⟩  in Step 1 and keep repeating the process until you get a "\n" character
    --> The "\n" character indicates you've reached the end of the dinosaur name
    
'''


# GRADED FUNCTION: sample

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """

    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation). (≈1 line)
    x = np.zeros([vocab_size, 1])
    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros([n_a, 1])

    # Create an empty list of indices which will contain the list of indices of the characters to generate
    indices = []

    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1

    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
    # trained model), which helps debugging and prevents entering an infinite loop.
    counter = 0
    newline_character = char_to_ix['\n']

    while idx != newline_character and counter != 50:
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        # for grading purposes
        np.random.seed(counter + seed)
        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(range(len(y)), p=y.ravel())
        # Append the index to "indices"
        indices.append(idx)

        # Step 4: Overwrite the input character as the one corresponding to the sampled index
        x = np.zeros([vocab_size, 1])
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # for grading purposes
        seed += 1
        counter += 1

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices


np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


indices = sample(parameters, char_to_ix, 0)
print("\nSampling:")
print("\nlist of sampled indices :\n", indices)
print("\nlist of sampled characters:\n", [ix_to_char[i] for i in indices])

'''

BUILDING THE LANGUAGE MODEL

It is time to build the character-level language model for text generation

GRADIENT DESCENT

In this section you will implement a function performing 1 step of stochastic gradient descent (with clipped gradients)

You'll go through the training examples one at a time, so the optimization algorithm will be stochastic gradient descent

As a reminder, here are the steps of a common optimization loop for an RNN:

1. Forward propagate through the RNN to compute the loss
2. Backward propagate through time to compute the gradients of the loss with respect to the parameters
3. Clip the gradients if necessary
4. Update your parameters using gradient descent

Exercise: Implement this optimization process (one step of stochastic gradient descent)

We provide you with the following functions:

def rnn_forward(X, Y, a_prev, parameters):
    """ Performs the forward propagation through the RNN and computes the cross-entropy loss.
    It returns the loss' value as well as a "cache" storing values to be used in the backpropagation."""
    ....
    return loss, cache

def rnn_backward(X, Y, parameters, cache):
    """ Performs the backward propagation through time to compute the gradients of the loss with respect
    to the parameters. It returns also all the hidden states."""
    ...
    return gradients, a

def update_parameters(parameters, gradients, learning_rate):
    """ Updates parameters using the Gradient Descent Update Rule."""
    ...
    return parameters

'''


# GRADED FUNCTION: optimize

def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """

    # Forward propagate through time (≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)

    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]


np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])

'''

TRAINING THE MODEL

Given the dataset of dinosaur names, we use each line of the dataset (one name) as one training example
Every 100 steps of stochastic gradient descent, you sample 10 randomly chosen names to see how the algorithm is doing

Remember to shuffle the dataset, so that stochastic gradient descent visits the examples in random order

Exercise: Follow the instructions and implement model()
    --> When examples[index] contains one dinosaur name (string), to create an example (X, Y), you can use this:

        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]] 
        Y = X[1:] + [char_to_ix["\n"]]

    Note that we use: index= j % len(examples)
    --> where j = 1....num_iterations, to make sure that examples[index] is a valid statement (index < len(examples))
    --> The first entry of X being None will be interpreted by rnn_forward() as setting  x⟨0⟩=0⃗
        - Further, this ensures that Y is equal to X but shifted one step to the left
        - With an additional "\n" appended to signify the end of the dinosaur name

'''


# GRADED FUNCTION: model

def model(data, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """

    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples).
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):

        # Use the hint above to define one training example (X,Y) (≈ 2 lines)
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)

                seed += 1  # To get the same result for grading purposed, increment the seed by one.

            print('\n')

    return parameters


parameters = model(data, ix_to_char, char_to_ix)