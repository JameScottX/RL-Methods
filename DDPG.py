import tensorflow as tf
import numpy as np
import os
import shutil

np.random.seed(1)
tf.set_random_seed(1)


class DDPG_Actor(object):

    def __init__(self, sess,action_dim,action_bound,state_dim,learning_rate,replace_iter):

        self.sess =sess
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.replace_iter =replace_iter
        self.replace_count = 0
           
        with tf.variable_scope('Actor'):

            self.states = tf.placeholder(tf.float32,shape = [None,state_dim],name = 'state')
            self.states_next = tf.placeholder(tf.float32,shape = [None,state_dim],name ='state_next' )

            self.actions = self.build_net(self.states,scope = 'eval_net',trainable = True)
            self.actions_next = self.build_net(self.states_next,scope = 'target_net',trainable = False)

        self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Actor/eval_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope ='Actor/target_net')


    def build_net(self,state,scope,trainable):

        with tf.variable_op_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            net1  = tf.layers.dense(state,200,
                                    activation = tf.nn.relu6,kernel_initializer = init_w,
                                    bias_initializer = init_b,name = 'net1',trainable  =trainable)
            net2 = tf.layers.dense(net1,200,
                                   activation = tf.nn.relu6,kernel_initializer = init_w,
                                   bias_initializer = init_b,name = 'net2',trainable  =trainable)
            net3 = tf.layers.dense(net2,10,
                                   activation = tf.nn.relu,kernel_initializer = init_w,
                                   bias_initializer = init_b,name = 'net3',trainable  =trainable)
            with tf. variable_scope('action'):

                actions = tf.layers.dense(net3,self.action_dim,activation = tf.nn.tanh,
                                          kernel_initializer = init_w,bias_initializer = init_b,
                                          name = 'actions',trainable  =trainable)

                scaled_actions = tf.multiply(actions,self.action_bound,name = 'scaled_actions')

        return scaled_actions

    def add_grad_to_graph(self,action_grads):

        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys  = self.actions,xs = self.eval_params,grad_ys =action_grads)
        with tf.variable_scope('Actor_train'):
            self.train_op = tf.train.RMSPropOptimizer(-self.learning_rate).apply_gradients(zip(self.policy_grads,self.eval_params))

    def learn(self,s):
        self.sess.run(self.train_op, feed_dict = {self.states:s})
        self.replace_count +=1
        if self.replace_count % self.replace_iter == 0:
            self.sess.run([tf.assign(t,e) for t ,e in zip(self.target_params,self.eval_params)])

    def choose_ations(self,s):
         s =s[np.newaxis,:]
         return self.sess.run(self.actions, feed_dict={self.states : s})[0]


class DDPG_Critic(object):

    def __init__(self,sess,state_dim,action_dim,learning_rate,gmma,replace_iter,action,action_next):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.replace_iter = replace_iter
        self.replace_count =  0

        with tf.variable_scope('Critic'):

            
            self.action  = action
            self.states = tf.placeholder(tf.float32,shape = [None,state_dim],name = 'Critic_state')
            self.states_next = tf.placeholder(tf.float32,shape = [None,state_dim],name ='Critic_state_next' )

            self.q_val = self.build_net(self.states,self.action,scope = 'eval_net',trainable = True)
            self.q_next_val = self.buid_net(self.states_next,action_next, scope = 'target_net',trainable = True)

            self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Critic/eval_net')
            self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Critic/target_net')

        with tf.variable_scope('error'):

            self.reward = tf.placeholder(tf.float32,shape = [None,1],name = 'reward')

            self.target_q = self.reward + gmma*self.q_next_val

            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q,self.q_val))

        with tf.variable_scope('train'):

            self.train_op  = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

        with tf.variable_scope('a_grad'):

            self.a_grad = tf.gradients(self.q_val,action)[0]


    def buid_net(self,state,action,scope,trainable):

        with tf.variable_scope(scope):

            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_op_scope('net1'):

                n_net = 200
                w_s = tf.get_variable('w_s',[self.state_dim,n_net],initializer =init_w,trainable = trainable )
                w_a = tf.get_variable('w_a',[self.action_dim,n_net],initialzer = init_w,trainable = trainable)
                b1 = tf.get_variable('b1',[1, n_net],initializer = init_b,trainable  = Ture)
                net1  = tf.nn.relu6(tf.matmul(state,w_s) + tf.matmul(action,w_a) + b1)

            net2  = tf.layers.dense(net1,200,activation = tf.nn.relu6,
                                    kernel_initializer = init_w,bias_initializer = init_b,
                                    name = 'net2',trainable  = trainable)
            net3  =tf.layers.dense(net2,10,activation = tf.nn.relu,
                                   kernel_initializer = init_w,bias_initializer = init_b,
                                   name = 'net3',trainable = trainable)
            net_q = tf.layers.dense(net3,1, kernel_initializer = init_w,bias_initializer = init_b,
                                     name = 'net_q',trainable  =trainable )

            return net_q

    def learn(self,states,action,reward,state_next):

        self.sess.run(train_op,feed_dict ={self.states :states,self.action:action,self.reward :reward,self.states_next:state_next})
        replace_count+=1
        if replace_count % replace_iter == 0 and replace_count ==1:
            sess.run([tf.assign(t,e) for t ,e in zip(self.target_params,self.eval_params)])
        



class DDPG_Memory(object):

    def __init__(self, capacity,dims):

        self.capacity =capacity
        self.data = np.zeros((capacity,dims))
        self.data_loc = 0

    def store_memry(self,states,action,reward,state_next):
        transition = np.hstack((states,action,[reward],state_next))
        index = self.data_loc % self.capacity
        self.data[index,:] = transition
        self.data_loc +=1

    def sample(self,size):
        assert self.data_loc >= self.capacity , 'Memory Loc overflow '
        indices = np.random.choice(self.capacity,size = size)
        return self,data[indices,:]

