import tensorflow as tf
import random
import numpy as np
import os
from collections import deque

from game import Game
from DQN_Value import Value

tf.compat.v1.disable_eager_execution()

class Deep_Q_Network:
    def __init__(self, sess, Game):

        self.sess = sess
        v = Value()
        self.game = Game
        self.settings = self.game.settings
        self.explore= v.EXPLORE
        self.n_actions = v.N_ACTIONS
        self.learning_rate = v.LEARNING_RATE
        self.gamma = v.GAMMA
        self.final_epsilon = v.FINAL_EPSILON
        self.initial_epsilon = v.INITIAL_EPSILON
        self.epsilon = v.INITIAL_EPSILON
        self.observe = v.OBSERVE
        self.replace_target_iter = v.REPLACE_TARGET_ITER
        self.memory_size = v.MEMORY_SIZE
        self.batch_size = v.BATCH_SIZE
        self.model_file = v.MODEL_FILE

        # total learning step
        self.learn_step = 0

        self.memory = deque(maxlen=self.memory_size)

        # consist of [target_net, evaluate_net]
        self.build_net()

        t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.compat.v1.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]

        self.loss_list = []
        self.restore_model()

    def conv_network(self, scope_name, state):
        settings = self.settings

        with tf.compat.v1.variable_scope(scope_name):
            conv1 = tf.compat.v1.layers.conv2d(state, filters=32, kernel_size=3,strides=1, padding="SAME",activation=tf.nn.relu, name="conv1")
            conv2 = tf.compat.v1.layers.conv2d(conv1, filters=64, kernel_size=3,strides=1, padding="SAME",activation=tf.nn.relu, name="conv2")
            conv3 = tf.compat.v1.layers.conv2d(conv2, filters=128, kernel_size=3,strides=1, padding="SAME",activation=tf.nn.relu, name="conv3")
            conv4 = tf.compat.v1.layers.conv2d(conv3, filters=4, kernel_size=1,strides=1, padding="SAME",activation=tf.nn.relu, name="conv4")

            conv4_flat = tf.reshape(conv4, shape=[-1, 4 * (settings.w+2) * (settings.h+2)])

            h_fc1 = tf.compat.v1.layers.dense(conv4_flat, 128, activation=tf.nn.relu)
            q_value = tf.compat.v1.layers.dense(h_fc1, self.n_actions)

        return q_value

    def build_net(self):
        settings = self.settings
        with tf.compat.v1.name_scope("inputs"):
            self.s = tf.compat.v1.placeholder(tf.float32, shape=[None, settings.w+2, settings.h+2, 8], name="s")
            self.r = tf.compat.v1.placeholder(tf.float32, [None, ], name='r')  # input Reward
            self.a = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='a')  # input Action

        # evaluate_net
        self.q_eval = self.conv_network('eval_net', self.s)

        # target_net
        self.q_next = self.conv_network('target_net', self.s)

        action_value = tf.reduce_sum(input_tensor=tf.multiply(self.q_eval, self.a), axis=1)
        self.loss = tf.reduce_mean(input_tensor=tf.square(self.r - action_value))
        self.train_step = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(self.loss)

    def restore_model(self):
        sess = self.sess
        model_file = self.model_file
        self.saver = tf.compat.v1.train.Saver()

        if os.path.exists(model_file + '.meta'):
            self.saver.restore(sess, model_file)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())

    def get_model_params(self):
        gvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self.sess.run(gvars))}

    def choose_action(self, s_t):
        a_t = np.zeros([self.n_actions])
        action_index = 0

        if random.random() <= self.epsilon:
            action_index = np.random.choice(4)
        else:
            q_eval = self.sess.run(self.q_eval, feed_dict={self.s : [s_t]})[0]
            action_index = np.argmax(q_eval)

        a_t[action_index] = 1

        return a_t, action_index

    def update_epsilon(self):
        # scale down epsilon
        if self.epsilon > self.final_epsilon and self.learn_step > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

    def play_a_game(self):
        game = self.game
        game.restart_game()
        score = 0
        step = 0
        game_state = game.current_state()
        s_t = np.concatenate((game_state, game_state, game_state, game_state), axis=2)

        while not game.g_end():
            a_t, action_index = self.choose_action(s_t, score)
            # run the selected action and observe next state and reward
            move = action_index
            r_t = game.do_move(move)

            if r_t == 1:
                score += 1

            game_state = game.current_state()
            end = game.g_end()
            s_t1 = np.append(game_state, s_t[:, :, :-2], axis=2)

            self.memory.append((s_t, a_t, r_t, s_t1, end))
            self.update_epsilon()
            self.learn_step += 1

            s_t = s_t1
            step += 1

            # only train if done observing
            if self.learn_step % 50 == 0 and len(self.memory) > self.observe:
                self.train_batch()

        return step, score

    def train_batch(self):
        m = self.memory #memory
        batch_size = self.batch_size

        # sample a minibatch to train on
        minibatch = random.sample(m, batch_size)
        # get the batch variables
        s_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s1_batch = [d[3] for d in minibatch]

        y_batch = []
        q_next = sess.run(self.q_next, feed_dict={self.s : s1_batch})

        for i in range(batch_size):
            end = minibatch[i][4]
            # if terminal, only equals reward
            if end:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + self.game * np.max(q_next[i]))

        # perform gradient step
        sess.run(self.train_step, feed_dict = { self.r : y_batch,self.a : a_batch,self.s : s_batch})

        if self.learn_step % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        if self.learn_step % 100000 == 0:
            self.saver.save(sess, self.model_file)

        if self.learn_step % 10000 == 0:
            batch_loss = sess.run(self.loss, feed_dict = {self.r: y_batch, self.a : a_batch,self.s : s_batch})
            self.loss_list.append(batch_loss)



    def train(self):
        try:
            g_num = 0
            scores = []
            score_means = []
            while self.learn_step < self.explore:
                step, score = self.play_a_game()
                g_num += 1
                scores.append(score)

                if g_num % 10 == 0:
                    score_mean = np.mean(scores)
                    score_means.append(score_mean)
                    print("game: {} step length: {} score: {:.2f}".format(g_num, step, score_mean))
                    scores = []

            self.plot_loss(score_means)

        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            self.saver.save(self.sess, './model/snake')
            self.plot_loss(score_means)

    def plot_loss(self, score_means):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(score_means)), score_means)
        plt.ylabel('Score')
        plt.xlabel('training steps')
        plt.show()

if __name__ == "__main__":

    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    game = Game()
    dqn = Deep_Q_Network(sess, game)
    dqn.train()
