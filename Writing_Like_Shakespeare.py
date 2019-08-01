from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io

"""

A similar (but more complicated) task is to generate Shakespeare poems
Instead of learning from a dataset of Dinosaur names you can use a collection of Shakespearian poems

Using LSTM cells, you can learn longer term dependencies that span many characters in the text
    --> i.e. Characters appearing somewhere in sequence can influence wording much later in the sequence
    --> These long term dependencies were less important with dinosaur names, since the names were quite short

We have implemented a Shakespeare poem generator with Keras!!

"""

'''

To save time, we have trained a model for ~1000 epochs on a collection of Shakespearian poems called "The Sonnets"

Let's train the model for one more epoch
When it finishes training for an epoch you can run generate_output, which will prompt asking you for an input

The poem will start with your sentence, and our RNN-Shakespeare will complete the rest of the poem for you! 

For example, try "Forsooth this maketh no sense" (don't enter the quotation marks)

Depending on whether you include the space at the end, your results might also differ
    --> Try it both ways, and try other inputs as well

'''

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

# Run this cell to try with different inputs without having to re-train the model
generate_output()

'''

The RNN-Shakespeare model is very similar to the one you have built for dinosaur names

The only major differences are:

1. LSTMs instead of the basic RNN to capture longer-range dependencies
2. The model is a deeper, stacked LSTM model (2 layer)
3. Using Keras instead of python to simplify the code

If you want to learn more, you can also check out the Keras Team's text generation implementation on GitHub: 
https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

Congratulations on finishing this notebook!
'''
