import tensorflow as tf

def attention(inputs,attention_size):
    inputs=tf.concat(inputs,2)

    v=tf.layers.dense(inputs,attention_size,activation=tf.nn.tanh)
    vu=tf.layers.dense(v,1,use_bias=False)
    alphas=tf.nn.softmax(vu)

    output=tf.reduce_mean(alphas*inputs,axis=1)
    return output