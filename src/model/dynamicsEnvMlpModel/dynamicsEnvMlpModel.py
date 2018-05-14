from src.model.simpleMlpModel.simpleMlpModel import SimpleMlpModel
import tensorflow as tf
from src.model.utils.networkCreator import NetworkCreator
import numpy as np
from src.config.config import Config
from config.key import CONFIG_KEY
from src.model.model import Model
import src.model.utils.utils as utl
from src.model.tensorflowBasedModel import TensorflowBasedModel
import easy_tf_log
from src.model.trpoModel.trpo.utils import Scaler


class DynamicsEnvMlpModel(TensorflowBasedModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/dynamicsEnvMlpModelKey.json')

    def __init__(self, config, output_bound):
        # TODO ADD NORMALIZATION
        # TODO THE PLACEHODER SHOULD MOVE TO AGENT AND USE IT AS INPUT FOR __init__
        super(DynamicsEnvMlpModel, self).__init__(config)

        with tf.variable_scope(name_or_scope=self.config.config_dict['NAME']):
            self.state_means = tf.placeholder(shape=list(self.config.config_dict['STATE_SPACE']),
                                              dtype=tf.float32,
                                              name='state_means')
            self.state_vars = tf.placeholder(shape=list(self.config.config_dict['STATE_SPACE']),
                                             dtype=tf.float32,
                                             name='state_vars')
            self.action_means = tf.placeholder(shape=list(self.config.config_dict['ACTION_SPACE']),
                                               dtype=tf.float32)
            self.action_vars = tf.placeholder(shape=list(self.config.config_dict['ACTION_SPACE']),
                                              dtype=tf.float32)
            self.output_means = tf.placeholder(shape=list(self.config.config_dict['STATE_SPACE']),
                                               dtype=tf.float32,
                                               name='delta_means')
            self.output_vars = tf.placeholder(shape=list(self.config.config_dict['STATE_SPACE']),
                                              dtype=tf.float32,
                                              name='delta_vars')

            self.state_input = tf.placeholder(shape=[None] + list(self.config.config_dict['STATE_SPACE']),
                                              dtype=tf.float32)
            self.action_input = tf.placeholder(shape=[None] + list(self.config.config_dict['ACTION_SPACE']),
                                               dtype=tf.float32)
            self.state_delta_label = tf.placeholder(shape=[None] + list(self.config.config_dict['STATE_SPACE']),
                                                    dtype=tf.float32)

            self.norm_state_input = (self.state_input - self.state_means) / self.state_vars
            self.norm_action_input = (self.action_input - self.action_means) / self.action_vars
            self.norm_state_delta_label = (self.state_delta_label - self.output_means) / self.output_vars

            self.input = tf.concat(values=[self.norm_state_input, self.norm_action_input],
                                   axis=1)

            self.action_scalar = Scaler(obs_dim=self.config.config_dict['ACTION_SPACE'])
            self.state_scalar = Scaler(obs_dim=self.config.config_dict['STATE_SPACE'])
            self.delta_scalar = Scaler(obs_dim=self.config.config_dict['STATE_SPACE'])

            self.net, self.delta_state_output, self.trainable_var_list = \
                NetworkCreator.create_network(input=self.input,
                                              network_config=self.config.config_dict['NET_CONFIG'],
                                              net_name=self.config.config_dict['NAME'])
            # output_low=output_bound[0] - output_bound[1],
            # output_high=output_bound[1] - output_bound[0])

            self.loss, self.optimizer, self.optimize = self.create_training_method()
            self.denorm_delta_state_output = self.delta_state_output * self.output_vars + self.output_means
            self.denorm_state_input = self.norm_state_input * self.state_vars + self.state_means

            self.output = self.state_input + self.denorm_delta_state_output

        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.config_dict['NAME'])

        self.variables_initializer = tf.variables_initializer(var_list=self.var_list)

    def create_training_method(self):
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.trainable_var_list])

        loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.norm_state_delta_label - self.delta_state_output),
                                            reduction_indices=[1])) + 0.0 * l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.config_dict['LEARNING_RATE'])
        optimize = optimizer.minimize(loss=loss, var_list=self.trainable_var_list)
        return loss, optimizer, optimize

    def update_mean_var(self, state_input, action_input, delta_state_label):
        self.state_scalar.update(x=state_input)
        self.action_scalar.update(x=action_input)
        self.delta_scalar.update(x=delta_state_label)

    def update(self, sess, state_input, action_input, delta_state_label):

        state_input = np.reshape(state_input, newshape=[-1] + list(self.config.config_dict['STATE_SPACE']))
        action_input = np.reshape(action_input, newshape=[-1] + list(self.config.config_dict['ACTION_SPACE']))
        delta_state_label = np.reshape(delta_state_label, newshape=[-1] + list(self.config.config_dict['STATE_SPACE']))

        total_loss = 0.0
        batch_count = len(state_input) // self.config.config_dict['BATCH_SIZE']
        if batch_count <= 0:
            raise ValueError('Batch count is zero, input data size: %d, batch size %d' %
                             (len(state_input), self.config.config_dict['BATCH_SIZE']))
        for j in range(batch_count):
            state_intput_j = state_input[self.config.config_dict['BATCH_SIZE'] * j:
                                         self.config.config_dict['BATCH_SIZE'] * (j + 1), :]
            action_input_j = action_input[self.config.config_dict['BATCH_SIZE'] * j:
                                          self.config.config_dict['BATCH_SIZE'] * (j + 1), :]
            delta_state_label_j = delta_state_label[self.config.config_dict['BATCH_SIZE'] * j:
                                                    self.config.config_dict['BATCH_SIZE'] * (j + 1), :]

            _, loss = sess.run(fetches=[self.optimize, self.loss],
                               feed_dict={self.state_input: state_intput_j,
                                          self.action_input: action_input_j,
                                          self.state_delta_label: delta_state_label_j,
                                          self.state_vars: np.sqrt(self.state_scalar.vars),
                                          self.state_means: self.state_scalar.means,
                                          self.action_vars: np.sqrt(self.action_scalar.vars),
                                          self.action_means: self.action_scalar.means,
                                          self.output_means: self.delta_scalar.means,
                                          self.output_vars: np.sqrt(self.delta_scalar.vars)
                                          })
            total_loss += loss
        average_loss = total_loss / batch_count
        self.log_queue.put({self.name + '_LOSS': average_loss})
        easy_tf_log.tflog(key=self.name + '_TRAIN_LOSS', value=average_loss)
        return average_loss

    def test(self, sess, state_input, action_input, delta_state_label):
        state_input = np.reshape(state_input, newshape=[-1] + list(self.config.config_dict['STATE_SPACE']))
        action_input = np.reshape(action_input, newshape=[-1] + list(self.config.config_dict['ACTION_SPACE']))
        delta_state_label = np.reshape(delta_state_label, newshape=[-1] + list(self.config.config_dict['STATE_SPACE']))

        loss = sess.run(fetches=self.loss,
                        feed_dict={self.state_input: state_input,
                                   self.action_input: action_input,
                                   self.state_delta_label: delta_state_label,
                                   self.state_vars: np.sqrt(self.state_scalar.vars),
                                   self.state_means: self.state_scalar.means,
                                   self.action_vars: np.sqrt(self.action_scalar.vars),
                                   self.action_means: self.action_scalar.means,
                                   self.output_means: self.delta_scalar.means,
                                   self.output_vars: np.sqrt(self.delta_scalar.vars)
                                   })

        output = sess.run(fetches=self.output,
                          feed_dict={self.state_input: state_input,
                                     self.action_input: action_input,
                                     self.state_delta_label: delta_state_label,
                                     self.state_vars: np.sqrt(self.state_scalar.vars),
                                     self.state_means: self.state_scalar.means,
                                     self.action_vars: np.sqrt(self.action_scalar.vars),
                                     self.action_means: self.action_scalar.means,
                                     self.output_means: self.delta_scalar.means,
                                     self.output_vars: np.sqrt(self.delta_scalar.vars)
                                     })

        self.log_queue.put({self.name + '_LOSS': np.mean(loss)})
        easy_tf_log.tflog(key=self.name + '_TEST_LOSS', value=loss)

    def predict(self, sess, state_input, action_input):
        # TODO MODIFY THIS RESHAPE PART

        state_input = np.reshape(state_input, newshape=[-1] + list(self.config.config_dict['STATE_SPACE']))
        action_input = np.reshape(action_input, newshape=[-1] + list(self.config.config_dict['ACTION_SPACE']))
        # norm_state_input = (state_input - self.state_scalar.means) / self.state_scalar.vars
        # norm_action_input = (action_input - self.action_scalar.means) / self.action_scalar.vars

        res = sess.run(fetches=[self.output],
                       feed_dict={self.state_input: state_input,
                                  self.action_input: action_input,
                                  self.state_vars: np.sqrt(self.state_scalar.vars),
                                  self.state_means: self.state_scalar.means,
                                  self.action_vars: np.sqrt(self.action_scalar.vars),
                                  self.action_means: self.action_scalar.means,
                                  self.output_means: self.delta_scalar.means,
                                  self.output_vars: np.sqrt(self.delta_scalar.vars)
                                  })

        return utl.squeeze_array(res, dim=1 + len(self.config.config_dict['STATE_SPACE']))

    def init(self):
        sess = tf.get_default_session()
        sess.run(self.variables_initializer)
        super().init()


if __name__ == '__main__':
    from config import CONFIG

    conf = Config(standard_key_list=DynamicsEnvMlpModel.key_list)
    conf.load_config(path=CONFIG + '/dynamicsEnvMlpModelTestConfig.json')
    a = DynamicsEnvMlpModel(config=conf)
    state_input = np.zeros(shape=[10, 20])
    action_input = np.zeros(shape=[10, 6])
    sess = tf.Session()
    with sess.as_default():
        a.init()
        a.load_snapshot()
        a.save_snapshot()
        a.save_model(path='/home/linsen/.tmp/model.ckpt', global_step=1)
        a.load_model(file='/home/linsen/.tmp/model.ckpt-1')
