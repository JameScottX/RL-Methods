import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGredient:

    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate = 0.001,
                 reward_decay = 0.95,
                 output_graph = False):
        
      self.n_actions = n_actions
      self.n_features = n_features
      self.lr = learning_rate
      self.gamma = reward_decay

      self.ep_obs,self.ep_as,self.ep_rs = [],[],[]

      self.bulid_net()
      self.sess = tf.Session()

      if output_graph:

          tf.summary.FileWriter("log/",self.sess.graph)

      self.sess.run(tf.global_variables_initializer())

    def bulid_net(self):
        
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32,[None,self.n_features],name = 'observation')
            self.tf_acts = tf.placeholder(tf.int32,[None,],name = 'actions_num')
            self.tf_acval = tf.placeholder(tf.float32,[None,],name = 'actions_value')

        layer1  = tf.layers.dense(
            inputs = self.tf_obs,
            units = 10,
            activation = tf.nn.tanh,
            kernel_initializer = tf.random_normal_initializer(mean = 0,stddev = 0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name = 'layer1'
            )

        layer2 = tf.layers.dense(
            inputs = layer1,
            units = self.n_actions,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean = 0,stddev = 0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name = 'layer2'
            )

        self.all_act_prob  = tf.nn.softmax(layer2,name = 'act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = layer2,labels =self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_acval)

        with tf.name_scope('train'):
            self.train_op  = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self,observation):

        prob_weights = self.sess.run(self.all_act_prob,feed_dict = {
            self.tf_obs: observation[np.newaxis,:]
            })
        action = np.random.choice(range(prob_weights.shape[1]),p =prob_weights.ravel() )
        return action

    def store_transition(self,s,a,r):

        self.ep_obs.append(s) 
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):

        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.sess.run(self.train_op,feed_dict = {
            self.tf_obs:np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_acval: discounted_ep_rs_norm
        
            })
        self.ep_obs,self.ep_as,self.ep_rs = [],[],[]

        return discounted_ep_rs_norm
    
    
    def _discount_and_norm_rewards(self):

        discounterde_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0,len(self.ep_rs))):
            running_add = running_add*self.gamma + self.ep_rs[t]
            discounterde_ep_rs[t] = running_add

        discounterde_ep_rs-= np.mean(discounterde_ep_rs)
        discounterde_ep_rs /= np.std(discounterde_ep_rs)

        return discounterde_ep_rs

