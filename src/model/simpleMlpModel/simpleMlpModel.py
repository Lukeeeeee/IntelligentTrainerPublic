from src.model.model import Model
import tensorflow as tf
from src.model.utils.networkCreator import NetworkCreator
import numpy as np
from src.config.config import Config
from config.key import CONFIG_KEY
from src.model.tensorflowBasedModel import TensorflowBasedModel
import easy_tf_log


class SimpleMlpModel(TensorflowBasedModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/simpleMlpModelKey.json')

    def __init__(self, config, INPUT_SPACE=0, OUTPUT_SPACE=0):
        # TODO ADD NORMALIZATION
        super(SimpleMlpModel, self).__init__(config)
        #        if INPUT_SPACE==0:
        self.input = tf.placeholder(shape=[None] + list(self.config.config_dict['INPUT_SPACE']),
                                    dtype=tf.float32)
        self.label = tf.placeholder(shape=[None] + list(self.config.config_dict['OUTPUT_SPACE']),
                                    dtype=tf.float32)
        #        else:
        #            self.input = tf.placeholder(shape=[None,INPUT_SPACE],
        #                                        dtype=tf.float32)
        #            self.label = tf.placeholder(shape=[None,OUTPUT_SPACE],
        #                                        dtype=tf.float32)
        #            self.config.config_dict['NET_CONFIG'][0]['N_UNITS'] = INPUT_SPACE
        #            self.config.config_dict['NET_CONFIG'][-1]['N_UNITS'] = OUTPUT_SPACE
        with tf.variable_scope(name_or_scope=self.config.config_dict['NAME']):
            self.net, self.output, self.trainable_var_list = \
                NetworkCreator.create_network(input=self.input,
                                              network_config=self.config.config_dict['NET_CONFIG'],
                                              net_name=self.config.config_dict['NAME'])

            self.loss, self.optimizer, self.optimize = self.create_training_method()
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.config_dict['NAME'])
        self.variables_initializer = tf.variables_initializer(var_list=self.var_list)

    def create_training_method(self):
        loss = tf.reduce_sum((self.output - self.label) ** 2)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.config_dict['LEARNING_RATE'])
        optimize = optimizer.minimize(loss=loss)
        return loss, optimizer, optimize

    # TODO UPDATE THIS API

    def update(self, sess, input, label):
        average_loss = 0.0
        if len(input) == 0:
            print("len(input)=0 detected")
        batch_count = len(input)  # // self.config.config_dict['BATCH_SIZE']
        for j in np.random.permutation(batch_count):
            input_j = input[j: (j + 1), :]
            label_j = label[j:(j + 1), :]
            _, loss = sess.run(fetches=[self.optimize, self.loss],
                               feed_dict={self.input: input_j,
                                          self.label: label_j
                                          })
            average_loss += loss
        average_loss = average_loss / batch_count
        self.log_queue.put({self.name + '_LOSS': average_loss})
        easy_tf_log.tflog(key=self.name + '_TRAIN_LOSS', value=average_loss)

    def predict(self, sess, input):
        if len(input.shape) == 1:
            input = np.expand_dims(input, 0)

        res = sess.run(fetches=[self.output],
                       feed_dict={self.input: input})
        return np.array(res)

    def init(self):
        sess = tf.get_default_session()
        sess.run(self.variables_initializer)
        super().init()


if __name__ == '__main__':
    from config import CONFIG

    conf = Config(standard_key_list=SimpleMlpModel.key_list)
    conf.load_config(path=CONFIG + '/simpleMlpModelTestConfig.json')
    a = SimpleMlpModel(config=conf)
    state_input = np.zeros(shape=[10, 6])
    action_input = np.zeros(shape=[10, 6])
    sess = tf.Session()
    with sess.as_default():
        a.init()
        res = a.predict(sess=sess, input=state_input)
        res = a.load_snapshot()
        res = a.save_snapshot()
