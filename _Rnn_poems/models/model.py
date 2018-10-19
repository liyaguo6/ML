import tensorflow as tf
def rnn_model(input_data,output_data,vocab_size,model="lstm"
              ,rnn_size=128,num_layers=2,batch_size=64,learning_rate=0.01):
    end_points = {}

    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
        cell_fun = tf.contrib.rnn
    elif model == 'gru':
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell
    cell = cell_fun(rnn_size, state_is_tuple=True)


    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)