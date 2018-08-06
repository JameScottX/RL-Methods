
import numpy as np
import pandas as pd
import tensorflow as tf


class DQN(object):

   def __init__(self,
                n_actions,
                n_features,
                learning_rate = 0.01,
                reward_decay = 0.9,
                e_greedy = 0.9,
                replace_target_iter = 200,
                memory_size = 500,
                batch_size = 32,
                e_greedy_increment = 0,
                output_graph = False,
                q_double = False,
                layer1_elmts = 20,
                use_e_greedy_increment = 1000
                ):

       self.n_actions = n_actions
       self.n_features = n_features
       self.lr = learning_rate    
       self.gamma = reward_decay
       self.epsilon_max = e_greedy
       self.replace_target_iter =replace_target_iter
       self.memory_size = memory_size
       self.batch_size = batch_size
       self.epsilon_increment = e_greedy_increment
       self.epsilon = e_greedy

       self.q_double = q_double
       self.layer1_elmts = layer1_elmts

       self.use_e_greedy_increment = use_e_greedy_increment
       

       self.learn_step_counter = 0

       self.memory = np.zeros((self.memory_size,self.n_features * 2 +2))
       
       self.build_net()

       e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'eval_net')
       t_params =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'target_net')

       with tf.variable_scope('soft_replacement'):
           self.target_replace_op = [tf.assign(t,e) for t ,e in zip(t_params,e_params)]

       self.sess  = tf.Session()

       self.sess.run(tf.global_variables_initializer())
       self.cost_his = []

   def build_net(self):

        #********************* build evaluate_net ********************

        self.state = tf.placeholder(tf.float32,[None,self.n_features],name = 'state')

        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name = 'QTarget')

        w_initializer,b_initializer=tf.random_normal_initializer(0.,0.3),tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):

           e_layer1 = tf.layers.dense( self.state,self.layer1_elmts,tf.nn.relu,kernel_initializer = w_initializer,\
                bias_initializer = b_initializer,name = 'e_layer1' )
           self.q_eval = tf.layers.dense(e_layer1,self.n_actions,kernel_initializer = w_initializer,\
                bias_initializer =b_initializer,name = 'q_eval')

        with tf.variable_scope('loss'):
               self.loss = tf.reduce_mean(tf.squared_difference( self.q_target,self.q_eval))
        with tf.variable_scope('train'):
               self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        
        self.state_next = tf.placeholder(tf.float32,[None,self.n_features],name = 'state_next')

         #********************* build target_net ********************
        with tf.variable_scope('target_net'):

            t_layer1 = tf.layers.dense(self.state_next,self.layer1_elmts,tf.nn.relu,kernel_initializer  = w_initializer,\
                bias_initializer = b_initializer,name = 't_layer1')
            self.q_next = tf.layers.dense(t_layer1,self.n_actions,kernel_initializer = w_initializer,\
                bias_initializer =b_initializer,name = 'q_next')


   def store_transition(self,state,action,reward,state_next):

        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((state,[action,reward],state_next))
        tmp_index  =  self.memory_counter % self.memory_size

        self.memory[tmp_index ,:] = transition

        self.memory_counter  += 1


   def choose_action(self,observation):

        observation = observation[np.newaxis,:]

        if np.random.uniform() < self.epsilon:

            action_val  = self.sess.run(self.q_eval,feed_dict={self.state :observation })
            action = np.argmax(action_val)
        else:
            action = np.random.randint(0,self.n_actions)

        return action


   def learn(self):

        if  self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run( self.target_replace_op)

        print('\target_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,size  = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size =self.batch_size)

        batch_memory = self.memory[sample_index,:]


        q_next, q_eval4next,q_eval = self.sess.run(
            [self.q_next,self.q_eval,self.q_eval],
            feed_dict = {
                self.state_next: batch_memory[:,-self.n_features:],
                self.state: batch_memory[:,-self.n_features:],
                self.state: batch_memory[:,:self.n_features]
                }
            )
        
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size,dtype  = np.int32)
        eval_act_index = batch_memory[:,self.n_features].astype(int)
        reward = batch_memory[:,self.n_features+1]


        if(self.q_double):
            max_act4next = np.argmax(q_eval4next,axis =1)
          
            select_q_next = q_next[batch_index,max_act4next]
        else:
            select_q_next = np.max(q_next,axis =1)

        q_target[batch_index,eval_act_index] = reward + self.gamma * select_q_next

        _,self.cost = self.sess.run([self.train_op,self.loss],feed_dict = {
            self.state: batch_memory[:,:self.n_features],self.q_target:q_target
            
            })

        self.cost_his.append(self.cost)

        if self.learn_step_counter>self.use_e_greedy_increment:
         self.epsilon = self.epsilon + self.epsilon_increment #if self.epsilon < self.epsilon_max else self.epsilon_increment
        

        self.learn_step_counter +=1

   def plot_cost(self):

        import matplotlib.pyplot as plt 
        plt.plot(np.arange(len(self.cost_his)),self.cost_his)

        plt.ylabel('Cost')
        plt.xlabel('training step')
        plt.show()
