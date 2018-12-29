from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
from src.config.config import Config
import tensorflow as tf
import itertools
from src.model.tensorflowBasedModel import TensorflowBasedModel
from src.model.utils.networkCreator import NetworkCreator
import tensorflow.contrib as tfcontrib


class DQNModel(TensorflowBasedModel):
    key_list = Config.load_json(file_path=None)

    def __init__(self, config, action_bound):
        super(DQNModel, self).__init__(config=config)
        self.proposed_action_list = []
        self.action_bound = action_bound
        action_list = []
        for i in range(len(action_bound[0])):
            low = action_bound[0][i]
            high = action_bound[1][i]
            action_list.append(np.arange(start=low,
                                         stop=high,
                                         step=(high - low) / self.config.config_dict['ACTION_SPLIT_COUNT']))
        action_iterator = itertools.product(*action_list)
        self.action_selection_list = []
        for action_sample in action_iterator:
            self.action_selection_list.append(tf.constant(action_sample))

        self.reward_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.state_input = tf.placeholder(shape=[None] + list(self.config.config_dict['STATE_SPACE']), dtype=tf.float32)
        self.next_state_input = tf.placeholder(shape=[None] + list(self.config.config_dict['STATE_SPACE']),
                                               dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None] + list(self.config.config_dict['ACTION_SPACE']),
                                           dtype=tf.float32)
        self.done_input = tf.placeholder(shape=[None, 1], dtype=tf.bool)
        self.input = tf.concat([self.state_input, self.action_input])
        self.done = tf.cast(self.done_input, dtype=tf.float32)

        self.q_value_list = []
        var_list = None
        for action_sample in self.action_selection_list:
            q_net, q_output, var_list = NetworkCreator.create_network(input=tf.concat(self.state_input, action_sample),
                                                                      network_config=self.config.config_dict[
                                                                          'NET_CONFIG'],
                                                                      net_name=self.config.config_dict['NAME'])
            self.q_value_list.append(q_output)
        self.var_list = var_list

        self.target_q_value_list = []
        for action_sample in self.action_selection_list:
            q_net, q_output, var_list = NetworkCreator.create_network(
                input=tf.concat(self.next_state_input, action_sample),
                network_config=self.config.config_dict[
                    'NET_CONFIG'],
                net_name='TARGET' + self.config.config_dict['NAME'])
            self.target_var_list.append(q_output)
        self.target_var_list = var_list

        self.loss, self.optimizer, self.optimize = self.create_training_method()
        self.update_target_q_op = self.create_target_q_update()
        self.memory = Memory(limit=1e100,
                             action_shape=self.config.config_dict['ACTION_SPACE'],
                             observation_shape=self.config.config_dict['STATE_SPACE'])
        self.sess = tf.get_default_session()

    def update(self):
        for i in range(self.config.config_dict['ITERATION_EVER_EPOCH']):
            batch_data = self.memory.sample(batch_size=self.config.config_dict['BATCH_SIZE'])
            loss = self.sess.run(fetches=[self.loss, self.optimize],
                                 feed_dict={
                                     self.reward_input: batch_data['rewards'],
                                     self.action_input: batch_data['actions'],
                                     self.state_input: batch_data['obs0'],
                                     self.done_input: batch_data['terminals1']
                                 })

    def predict(self, obs, q_value):
        pass

    def print_log_queue(self, status):
        self.status = status
        while self.log_queue.qsize() > 0:
            log = self.log_queue.get()
            print("%s: Critic loss %f: " %
                  (self.name, log[self.name + '_CRITIC']))
            log['INDEX'] = self.log_print_count
            self.log_file_content.append(log)
            self.log_print_count += 1

    def create_training_method(self):
        l1_l2 = tfcontrib.layers.l1_l2_regularizer()
        loss = tf.reduce_sum((self.predict_q_value - self.q_output) ** 2) + \
               tfcontrib.layers.apply_regularization(l1_l2, weights_list=self.var_list)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config.config_dict['LEARNING_RATE'])
        optimize_op = optimizer.minimize(loss=loss, var_list=self.var_list)
        return loss, optimizer, optimize_op

    def create_predict_q_value_op(self):

        predict_q_value = (1. - self.done) * self.config.config_dict['DISCOUNT'] * self.target_q_output \
                          + self.reward_input
        return predict_q_value

    def create_target_q_update(self):
        op = []
        for var, target_var in zip(self.var_list, self.target_var_list):
            ref_val = self.config.config_dict['DECAY'] * target_var + (1.0 - self.config.config_dict['DECAY']) * var
            op.append(tf.assign(ref_val, var))
        return op

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        self.memory.append(obs0=state,
                           obs1=next_state,
                           action=action,
                           reward=reward,
                           terminal1=done)
