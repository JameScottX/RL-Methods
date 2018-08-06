import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym


ACTOR_UPDATE_STEPS = 10
CRITIC_UPDATE_STEPS =10
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RARTE = 0.0002



METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]     

class PPO(object):

    def __init__(self,ep_max,ep_length,batch,gmma,state_dim,action_dim,action_bound):

        
        self.ep_max = ep_max
        self.ep_length = ep_length
        self.batch = batch
        self.gmma = gmma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.sess = tf.Session()
        self.states  =tf.placeholder(tf.float32,[None,self.state_dim],name = 'states')

        self.critic_init()   #critic

        #actor
        pi,pi_params  =self.build_net(train_able = True,name = 'pi')
        oldpi,oldpi_params = self.build_net(train_able = False,name = 'oldpi')
        
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1),axis =0)

        with tf.variable_scope('upadte_oldpi'):
            self.update_oldpi_pi = [oldp.assign(p) for p ,oldp in zip(pi_params,oldpi_params)]
        
        self.action = tf.placeholder(tf.float32,[None,self.action_dim],'action')
        self.action_advantage = tf.placeholder(tf.float32,[None,1],'advantage')

        with tf.variable_scope('loss'):

            with tf.variable_scope('surrogate'):

                ratio = pi.prob(self.action)/oldpi.prob(self.action)
                surr   = ratio * self.action_advantage

                if METHOD['name'] == 'k1_pen':
                    pass
                else:

                    self.action_loss = - tf.reduce_mean(tf.minimum(
                        surr,
                        tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.action_advantage)) 

        with tf.variable_scope('action_train'):
            self.action_train_op = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).minimize(self.action_loss)
        

        #self.sess.run(tf.global_variables_initializer())



    def critic_init(self):

         with tf.variable_scope( 'critic'):

             layer1 = tf.layers.dense(self.states,200,tf.nn.relu);             

             self.v_ = tf.layers.dense(layer1,1)

             self.discount_reward  =  tf.placeholder(tf.float32,[None,1],'discount_reward')

             self.advantage = self.discount_reward - self.v_
             self.cross  = tf.reduce_mean(tf.square(self.advantage))
             self.critic_train_op = tf.train.AdamOptimizer(CRITIC_LEARNING_RARTE).minimize(self.cross)


    def build_net(self,train_able,name):

        with tf.variable_scope(name):

            l1 = tf.layers.dense(self.states,200,tf.nn.relu,trainable = train_able)
            l2 = tf.layers.dense(l1,200,tf.nn.relu,trainable = train_able)

            mu = self.action_bound * tf.layers.dense(l2,self.action_dim ,tf.nn.tanh,trainable = train_able)

            sigmu = tf.layers.dense(l2,self.action_dim ,tf.nn.softplus,trainable = train_able)
            norm_dist = tf.distributions.Normal(loc = mu,scale =sigmu)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = name)


        return norm_dist, params


    def choose_action(self,s):

        s = s[np.newaxis,:]
        a = self.sess.run(self.sample_op,{self.states:s})[0]
       
        return np.clip(a,-self.action_bound ,self.action_bound )


    def update(self,s,a,r):

        self.sess.run(self.update_oldpi_pi)
        advantage = self.sess.run(self.advantage,{self.states:s,self.discount_reward:r})

        if METHOD['name'] == 'kl_pen':

            pass

        else:

            [self.sess.run(self.action_train_op,{self.states:s,self.action:a,self.action_advantage:advantage})for _ in range(ACTOR_UPDATE_STEPS)]
            [self.sess.run(self.critic_train_op,{self.states:s,self.discount_reward:r})for _ in range(CRITIC_UPDATE_STEPS)]

    def get_value(self,s):

        if s.ndim<2: s=s[np.newaxis,:]

        return self.sess.run(self.v_,{self.states:s})[0,0]





       
    

