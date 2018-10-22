import tensorflow as tf
def rnn_model(input_data,output_data,vocab_size,model="lstm"
              ,rnn_size=128,num_layers=2,batch_size=64,learning_rate=0.01):
    end_points = {}

    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell
    cell = cell_fun(rnn_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.random_uniform(shape=[vocab_size+1,rnn_size],minval=-1.0,maxval=1.0))
        inputs = tf.nn.embedding_lookup(embedding, input_data)
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, rnn_size])