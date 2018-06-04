from utils import *
import tensorflow as tf
from attention import attention
class hier_rnn():
    def __init__(self,args):
        self.args=args
        self.sentence=tf.placeholder(tf.int32,[self.args.batch_size,None,None])
        self.target=tf.placeholder(tf.int64,[self.args.batch_size])
        self.seq_len=tf.placeholder(tf.int32,[None])
        self.max_len=tf.placeholder(tf.int32,shape=())

    def word_embedding(self,input):
        def cell():
            return tf.nn.rnn_cell.GRUCell(128)

        cell_bw=cell_fw=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(3)])
        outputs,_=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,input,
                                                  sequence_length=self.seq_len,dtype=tf.float32,
                                                  scope='word_embedding')
        return attention(outputs,128)

    def sentence_embedding(self,input):
        def cell():
            return tf.nn.rnn_cell.GRUCell(128)

        cell_bw=cell_fw=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(3)])
        cell_fw_initial=cell_fw.zero_state(self.args.batch_size,tf.float32)
        cell_bw_initial=cell_bw.zero_state(self.args.batch_size,tf.float32)
        outputs,_=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,input,
                                                  initial_state_fw=cell_fw_initial,
                                                  initial_state_bw=cell_bw_initial,
                                                  scope='sentence_embedding')
        return attention(outputs,128)

    def forward(self):
        # time_step=self.sentence.shape[2].value
        sen_in=tf.reshape(self.sentence,[self.args.batch_size*self.max_len,-1])
        with tf.device("/cpu:0"):
            embedding=tf.get_variable('embedding',shape=[89526,256])
            inputs=tf.nn.embedding_lookup(embedding,sen_in)
        word_embedding=self.word_embedding(inputs)
        word_embedding=tf.reshape(word_embedding,[self.args.batch_size,-1,256])
        sen_embedding=self.sentence_embedding(word_embedding)
        logits=tf.layers.dense(sen_embedding,2)
        cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.target,logits=logits))
        optimizer=tf.train.AdamOptimizer().minimize(cross_entropy)
        correct=tf.equal(self.target,tf.argmax(tf.nn.softmax(logits),axis=1))
        accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
        return cross_entropy,optimizer,accuracy

    def train(self):
        cross_entropy,optimizer,accuracy=self.forward()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10):
                x_batch,y_batch,seq_len,max_len=next_batch(self.args.batch_size)
                for step in range(len(x_batch)):
                    # print(y_batch[step])
                    # print(y_batch[step])
                    loss,_,acc=sess.run([cross_entropy,optimizer,accuracy],
                                        feed_dict={self.sentence:x_batch[step],
                                                   self.target:y_batch[step],
                                                   self.seq_len:seq_len[step],
                                                   self.max_len:max_len[step]})
                    if step%10==0:
                        print("Epoch %d,Step %d,loss is %f"%(epoch,step,loss))
                        print("Epoch %d,Step %d,accuracy is %f"%(epoch,step,acc))
                x_batch,y_batch,seq_len,max_len=next_batch(self.args.batch_size,mode='test')
                test_accuracy=0
                for step in range(len(x_batch)):
                    acc=sess.run(accuracy,feed_dict={self.sentence:x_batch[step],
                                                     self.target:y_batch[step],
                                                     self.seq_len:seq_len[step],
                                                     self.max_len:max_len[step]})
                    test_accuracy+=acc
                print('test accuracy is %f'%(test_accuracy/len(x_batch)))
