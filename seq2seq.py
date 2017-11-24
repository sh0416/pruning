import numpy as np
import tensorflow as tf
import helper


PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

# Make placeholder
encoder_inputs = tf.placeholder(shape=(None,None), dtype=tf.int32, 
        name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None,None), dtype=tf.int32, 
        name='decoder_targets')
decoder_inputs = tf.placeholder(shape=(None,None), dtype=tf.int32, 
        name='decoder_inputs')

# Word embedding for input
embeddings = tf.Variable(
        tf.random_uniform([vocab_size,input_embedding_size], -1.0, 1.0), 
        dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

# Encoder part
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, 
        encoder_inputs_embedded, dtype=tf.float32, time_major=True)

# Only encoder_final_state is needed
del encoder_outputs

# Decoder part
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, 
        decoder_inputs_embedded, initial_state=encoder_final_state,
        dtype=tf.float32, time_major=True, scope='plain_decoder')

# Convert to logit for training
decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
# Predict for inferencing
decoder_prediction = tf.argmax(decoder_logits, 2)

# Loss
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits)
loss = tf.reduce_mean(stepwise_cross_entropy)

# Train
train_op = tf.train.AdamOptimizer().minimize(loss)

# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Get the size of graph
    global_var = tf.global_variables()
    size = 0
    for var in global_var:
        size += np.prod(var.get_shape().as_list())
    print('the size of graph: {}'.format(size))

    # Simple execution of this model
    batch_ = [[6], [3,4], [9,8,7]]

    batch_, batch_length_ = helper.batch(batch_)
    print('batch_encoded:\n' + str(batch_))

    din_, dlen_ = helper.batch(np.ones(shape=(3,1), dtype=np.int32),
            max_sequence_length=4)
    print('decoder inputs:\n' + str(din_))

    pred_ = sess.run(decoder_prediction, 
            feed_dict={encoder_inputs: batch_, decoder_inputs: din_})
    print('decoder prediction:\n' + str(pred_))

    # Simple train process of this model
    batch_size = 100

    batches = helper.random_sequences(length_from=3, length_to=8, vocab_lower=2,
            vocab_upper=10, batch_size=batch_size)

    print('head of the batch:')
    for seq in next(batches)[:10]:
        print(seq)

    loss_track = []
    max_batches = 30001
    batches_in_epoch = 1000

    try:
        for batch in range(max_batches):
            fd = helper.next_feed(batches, EOS)
            fd = {encoder_inputs: fd['encoder_inputs'],
                    decoder_inputs: fd['decoder_inputs'],
                    decoder_targets: fd['decoder_targets']}
            _, l = sess.run([train_op, loss], feed_dict=fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('  minibatch loss: {}'.format(sess.run(loss, feed_dict=fd)))
                predict_ = sess.run(decoder_prediction, feed_dict=fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print('  sample: {}'.format(i+1))
                    print('    input    -> {}'.format(inp))
                    print('    predicted-> {}'.format(pred))
                    if i >= 2:
                        break
                print()
    except KeyboardInterrupt:
        print('training interrupted')

