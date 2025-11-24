import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras
mnist = tf.keras.datasets.mnist
from scipy import special
from keras.utils.np_utils import to_categorical



class CmiBoundSimulator:
    def __init__(self):
        seed_all = np.random.randint(low=0, high=2 ** 20 - 1, size=3)
        self.num_traindp = 20000  # it is not 50000 so that we can estimate the impact of different datasets
        self.seed_ds = seed_all[0]
        self.seed_J = seed_all[1]
        self.seed_U = seed_all[2]
        self.z_Jc, self.z_J1, self.z_J2, self.uJ, self.test_ds = self.cmi_data_preprocess()
        self.model = self.model_def()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.05,
            decay_steps=40,
            decay_rate=0.90,
            staircase=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        self.theta_fn_name = 'erf'  # it can be 'tanh', 'erf','sigmoid' or 'indicator'.
        self.var_0 = 1e-5
        self.var_inf = 1e-8
        self.var_rate_v = 0.5
        self.var_steps_v = 40

    def cmi_data_preprocess(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
        indx_shuffle = np.random.RandomState(seed=self.seed_ds).permutation(len(x_train))  # Data shuffling
        # before sampling from it.
        x_train, y_train = x_train[indx_shuffle], y_train[indx_shuffle]
        x_super, y_super = x_train[:self.num_traindp], y_train[:self.num_traindp]  # first we take num_traindp samples
        J = np.random.RandomState(seed=self.seed_J).randint(self.num_traindp - 1,
                                                            size=1)  # we select the index J between
        # [0,num_traindp-1]
        x_J1_, y_J1_ = x_super[J], y_super[J]  # this is J1
        x_J2_, y_J2_ = x_train[[self.num_traindp]], y_train[[self.num_traindp]]  # the is n+1 points for J2
        x_Jc_, y_Jc_ = np.delete(x_super, [J], axis=0), np.delete(y_super, [J], axis=0)  # this is the rest of point
        [uJ] = np.random.RandomState(seed=self.seed_U).binomial(1, 1 / 2, 1) + 1  # uJ decides whether z_J1  or z_J2 in
        # the training set
        z_Jc = [tf.convert_to_tensor(x_Jc_), tf.convert_to_tensor(y_Jc_)]
        z_J1 = [tf.convert_to_tensor(x_J1_), tf.convert_to_tensor(y_J1_)]
        z_J2 = [tf.convert_to_tensor(x_J2_), tf.convert_to_tensor(y_J2_)]
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        return z_Jc, z_J1, z_J2, uJ, test_ds

    def theta_fn(self, arg):
        if self.theta_fn_name == 'tanh':
            return 1 / 2 + 1 / 2 * tf.math.tanh(arg)
        elif self.theta_fn_name == 'erf':
            return 1 / 2 + 1 / 2 * tf.math.erf(arg)
        elif self.theta_fn_name == 'sigmoid':
            return tf.math.sigmoid(arg)
        elif self.theta_fn_name == 'indicator':
            return 1 / 2 * tf.math.sign(arg) + 1 / 2

    @tf.function
    def norm_list(self, ww_list):  # this function is for computing norm of params. params in TF is list of tensors
        norm_ww_sq = 0
        for ww in ww_list:
            norm_ww_sq = norm_ww_sq + tf.math.square(tf.norm(ww))
        return tf.sqrt(norm_ww_sq)

    @tf.function
    def inner_prod(self, ww_1, ww_2):  # this function is for inner product
        inner_prod_out = 0
        for ww_1_i, ww_2_i in zip(ww_1, ww_2):
            inner_prod_out = inner_prod_out + tf.math.reduce_sum(tf.multiply(ww_1_i, ww_2_i))
        return inner_prod_out

    def model_def(self):  # model definition
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu',
                                   kernel_initializer='he_uniform'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu',
                                   kernel_initializer='he_uniform'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(10, kernel_initializer='he_uniform')
        ])
        return model

    @tf.function
    def compute_gradients(self):  # this function computer gradients for other functions
        x_Jc, y_Jc = self.z_Jc
        with tf.GradientTape() as tape:
            predictions_Jc = self.model(x_Jc)
            loss_Jc = self.loss_fn(y_Jc, predictions_Jc)
        grad_Jc = tape.gradient(loss_Jc, self.model.trainable_variables)
        self.train_accuracy(y_Jc, predictions_Jc)  # the training error and accuracy
        self.train_loss(loss_Jc)  # are updated here

        x_J1, y_J1 = self.z_J1
        with tf.GradientTape() as tape:
            predictions_J1 = self.model(x_J1)
            loss1_cmi = self.loss_fn(y_J1, predictions_J1)
        grad_J1 = tape.gradient(loss1_cmi, self.model.trainable_variables)
        if self.uJ == 1:
            self.train_accuracy(y_J1, predictions_J1)
            self.train_loss(loss1_cmi)

        x_J2, y_J2 = self.z_J2
        with tf.GradientTape() as tape:
            predictions_J2 = self.model(x_J2)
            loss2_cmi = self.loss_fn(y_J2, predictions_J2)
        grad_J2 = tape.gradient(loss2_cmi, self.model.trainable_variables)
        if self.uJ == 2:
            self.train_accuracy(y_J2, predictions_J2)
            self.train_loss(loss2_cmi)

        return grad_Jc, grad_J1, grad_J2

    def learning_rate_tracker(self, iter_current_):
        decay_step_ = self.lr_schedule.decay_steps
        decay_rate_ = self.lr_schedule.decay_rate
        init_learning_rate_ = self.lr_schedule.initial_learning_rate
        if self.lr_schedule.staircase:
            p = np.floor(iter_current_ / decay_step_)
        else:
            p = iter_current_ / decay_step_
        return init_learning_rate_ * np.power(decay_rate_, p)

    def var_noise(self, iter_=0):  
        var_n = self.var_inf + (self.var_0 - self.var_inf) * np.exp(
            -self.var_rate_v * np.floor(iter_ / self.var_steps_v))
        return var_n

    @tf.function
    def cmi_bound_sq(self, grad_J1, grad_J2, beta_t, lr_t, arg_theta, bound_sq):
        incoh = []
        for grad_J1_i, grad_J2_i in zip(grad_J1, grad_J2):
            incoh.append(tf.math.subtract(grad_J1_i, grad_J2_i))
        incoh_norm_sq = tf.square(self.norm_list(incoh))  
        bound_sq = bound_sq + beta_t * lr_t * incoh_norm_sq * tf.square(self.theta_fn(arg_theta))
        return bound_sq, incoh_norm_sq

    @tf.function
    def cmi_update_ml(self, grad_J1, grad_J2, noise_ld, lr_t, beta_t, arg_theta):
        incoh_vec = []
        if self.uJ == 1:
            for grad_J1_i, grad_J2_i in zip(grad_J1, grad_J2):
                incoh_vec.append(-grad_J1_i + grad_J2_i)
            incoh_norm_sq = tf.square(self.norm_list(incoh_vec))
            inner_prod_term = self.inner_prod(incoh_vec, noise_ld)
            term1_arg = tf.multiply(beta_t, lr_t) / (2 * self.num_traindp ** 2) * incoh_norm_sq
            term2_arg = tf.sqrt(tf.multiply(beta_t, lr_t * 2)) / self.num_traindp * inner_prod_term
            arg_theta = arg_theta - 1 / 2 * (term1_arg + term2_arg)
        elif self.uJ == 2:
            for grad_J1_i, grad_J2_i in zip(grad_J1, grad_J2):
                incoh_vec.append(grad_J1_i - grad_J2_i)
            incoh_norm_sq = tf.square(self.norm_list(incoh_vec))
            inner_prod_term = self.inner_prod(incoh_vec, noise_ld)
            term1_arg = tf.multiply(beta_t, lr_t) / (2 * self.num_traindp ** 2) * incoh_norm_sq
            term2_arg = tf.sqrt(tf.multiply(beta_t, lr_t * 2)) / self.num_traindp * inner_prod_term
            arg_theta = arg_theta - 1 / 2 * (term1_arg + term2_arg)

        return arg_theta

    @tf.function
    def training_step(self, var_n, lr, gradients_):  # This is the training step, base on value of uJ we decide what is
        # the gradient for training.
        grad_Jc, grad_J1, grad_J2 = gradients_
        grad_training = []
        if self.uJ == 1:
            for grad_Jc_i, grad_J1_i in zip(grad_Jc, grad_J1):
                grad_training.append(
                    (self.num_traindp - 1) / self.num_traindp * grad_Jc_i + 1 / self.num_traindp * grad_J1_i)
        elif self.uJ == 2:
            for grad_Jc_i, grad_J2_i in zip(grad_Jc, grad_J2):
                grad_training.append(
                    (self.num_traindp - 1) / self.num_traindp * grad_Jc_i + 1 / self.num_traindp * grad_J2_i)
        processed_grads = []
        noise_ld = []
        for g in grad_training:
            g_process, eps_ld = self.noisy_gradient(g, var_noise=var_n, learning_rate=lr)
            processed_grads.append(g_process)
            noise_ld.append(eps_ld)
        self.optimizer.apply_gradients(zip(processed_grads, self.model.trainable_variables))
        return noise_ld

    @tf.function
    def noisy_gradient(self, g, var_noise, learning_rate):  # making gradient noisy
        # since by applying gd, tf will multiply both side by eta_t.
        noise_ld_ = tf.random.normal(tf.shape(g))
        g = g + tf.sqrt(var_noise) / learning_rate * noise_ld_
        return g, noise_ld_

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images)
        self.test_accuracy(labels, predictions)

    def train(self, num_epochs=701):
        train_loss_all = np.array([])
        train_accuracy_all = np.array([])
        test_accuracy_all = np.array([])
        sq_gen_all = np.array([])
        arg_theta_all = np.array([])
        incoh_norm_sq_all = np.array([])
        gen_bound_sq = tf.convert_to_tensor(0.0)
        arg_theta = 0.0
        for epoch in range(num_epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()
            iter_t = self.optimizer.iterations.numpy()
            lr_t = self.learning_rate_tracker(iter_t)
            lr_t_tensor = tf.convert_to_tensor(lr_t, dtype=tf.float32)
            var_noise_ = self.var_noise(iter_=iter_t)
            var_noise_tensor = tf.convert_to_tensor(var_noise_, dtype=tf.float32)
            beta_t = 2 * lr_t / var_noise_
            beta_t_tensor = tf.convert_to_tensor(beta_t, dtype=tf.float32)
            grad_list = self.compute_gradients()  # accuracy and loss are computed inside this
            # w_cur_iter = self.model.trainable_variables
            gen_bound_sq, incoh_norm_sq = self.cmi_bound_sq(grad_list[1], grad_list[2], beta_t_tensor, lr_t_tensor,
                                                            arg_theta, gen_bound_sq)
            noise_ld = self.training_step(var_noise_tensor, lr_t_tensor, grad_list)
            # w_next_iter = self.model.trainable_variables
            arg_theta = self.cmi_update_ml(grad_list[1], grad_list[2], noise_ld, lr_t_tensor, beta_t_tensor, arg_theta)

            for test_images, test_labels in self.test_ds:
                self.test_step(test_images, test_labels)

            if epoch % 100 == 0:
                template = 'Epoch {}, Loss: {}, Accuracy train: {}, LR: {}, var_noise: {}, arg_theta: {}, cmi_bound: {}'
                print(template.format(epoch + 1,
                                      self.train_loss.result(),
                                      self.train_accuracy.result() * 100,
                                      lr_t,
                                      var_noise_,
                                      arg_theta.numpy(),
                                      1 / (self.num_traindp * np.sqrt(2)) * np.sqrt(gen_bound_sq) * 100))

            train_loss_all = np.append(train_loss_all, self.train_loss.result())
            train_accuracy_all = np.append(train_accuracy_all, self.train_accuracy.result())
            test_accuracy_all = np.append(test_accuracy_all, self.test_accuracy.result())
            incoh_norm_sq_all = np.append(incoh_norm_sq_all, incoh_norm_sq)
            arg_theta_all = np.append(arg_theta_all, arg_theta)
            sq_gen_all = np.append(sq_gen_all, gen_bound_sq)

        return train_loss_all, train_accuracy_all, test_accuracy_all, incoh_norm_sq_all, arg_theta_all, sq_gen_all


class GradBoundSimulator:
    def __init__(self):
        self.num_traindp = 20000  # it is not 50000 so that we can estimate the impact of different datasets
        self.x_train, self.y_train, self.x_train_tensor, self.y_train_tensor, self.test_ds = self.mnist_ds()
        self.model = self.model_def()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.05,
            decay_steps=40,
            decay_rate=0.90,
            staircase=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        self.var_0 = 1e-5
        self.var_inf = 1e-8
        self.var_rate_v = 0.5
        self.var_steps_v = 40

    def mnist_ds(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
        rnd_indices = np.random.permutation(
            len(x_train))  # shuffle training data so that we can simulate different training set
        rnd_indices_trunc = rnd_indices[:self.num_traindp]
        x_train = x_train[rnd_indices_trunc]
        y_train = y_train[rnd_indices_trunc]
        y_train = to_categorical(y_train, num_classes=10)  
        y_test = to_categorical(y_test, num_classes=10)  
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        x_train_tensor = tf.convert_to_tensor(x_train)
        y_train_tensor = tf.convert_to_tensor(y_train)
        return x_train, y_train, x_train_tensor, y_train_tensor, test_ds

    def model_def(self):  # model definition
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu',
                                   kernel_initializer='he_uniform'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu',
                                   kernel_initializer='he_uniform'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(10, kernel_initializer='he_uniform')
        ])
        return model

    def learning_rate_tracker(self, iter_current_):
        decay_step_ = self.lr_schedule.decay_steps
        decay_rate_ = self.lr_schedule.decay_rate
        init_learning_rate_ = self.lr_schedule.initial_learning_rate
        staircase_ = self.lr_schedule.staircase
        if staircase_:
            p = np.floor(iter_current_ / decay_step_)
        else:
            p = iter_current_ / decay_step_
        return init_learning_rate_ * np.power(decay_rate_, p)

    def var_noise(self, iter_=0): # noise variance scheduler
        var_n = self.var_inf + (self.var_0 - self.var_inf) * np.exp(-self.var_rate_v * np.floor(iter_ / self.var_steps_v))
        return var_n

    @tf.function
    def grad_bound(self, learning_rate, beta, sq_gen):
        sampling_size = 2000  # since we need to compute per sample gradient,
        # we do it by sampling a subset of size "sampling_size"
        indx_shuffle = np.random.permutation(len(self.x_train))
        x_train_trunc = self.x_train[indx_shuffle][:sampling_size]
        y_train_trunc = self.y_train[indx_shuffle][:sampling_size]
        indx_rnd_est = np.random.permutation(sampling_size)
        sum_sq_grad = 0.
        for indx_ in np.array_split(indx_rnd_est, 4):
            x_batch_ = tf.convert_to_tensor(x_train_trunc[indx_])
            y_batch_ = tf.convert_to_tensor(y_train_trunc[indx_])
            sum_sq_grad = sum_sq_grad + self.per_example_sq_gradients(x_batch_, y_batch_)
        est_sq_grad = sum_sq_grad / sampling_size
        sq_gen_new = sq_gen + learning_rate * beta * est_sq_grad
        return sq_gen_new

    @tf.function
    def per_example_sq_gradients(self, inputs, labels):
        vec_map = tf.vectorized_map(self.model_fn, (inputs, labels))
        sum_ = 0
        for item in vec_map:
            sum_ = sum_ + tf.math.square(tf.norm(item))
        return sum_

    @tf.function
    def model_fn(self, arg):
        with tf.GradientTape() as g:
            inp, label = arg
            inp_vec = tf.expand_dims(inp, 0)
            label_vec = tf.expand_dims(label, 0)
            prediction_vec = self.model(inp_vec)
            loss_vec = self.loss_fn(label_vec, prediction_vec)
        return g.gradient(loss_vec, self.model.trainable_variables)

    @tf.function
    def noisy_gradient(self, g, var_noise, learning_rate):
        # since by applying gd, tf will multiply both side by eta_t.
        g = g + tf.random.normal(tf.shape(g), stddev=tf.multiply(tf.sqrt(var_noise), tf.pow(learning_rate, -1)))
        return g

    @tf.function
    def training_step(self, x_batch, y_batch, var_n, lr):
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch)
            loss = self.loss_fn(y_batch, predictions)
        gradients_ = tape.gradient(loss, self.model.trainable_variables)
        processed_grads = [self.noisy_gradient(g, var_noise=var_n, learning_rate=lr) for g in gradients_]
        self.optimizer.apply_gradients(zip(processed_grads, self.model.trainable_variables))
        self.train_accuracy(y_batch, predictions)
        self.train_loss(loss)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images)
        self.test_accuracy(labels, predictions)

    def train(self, num_epochs=700):
        train_loss_all = np.array([])
        train_accuracy_all = np.array([])
        test_accuracy_all = np.array([])
        sq_gen_all = np.array([])
        sq_gen = np.float32(0)
        for epoch in range(num_epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()
            iter_t = self.optimizer.iterations.numpy()
            # print("iter num", iter_t)
            lr_t = self.learning_rate_tracker(iter_t)
            # print("learning rate", lr_t)
            var_noise_ = self.var_noise(iter_=iter_t)
            var_noise_tensor = tf.convert_to_tensor(var_noise_, dtype=tf.float32)
            lr_t_tensor = tf.convert_to_tensor(lr_t, dtype=tf.float32)
            beta_t = tf.convert_to_tensor(2 * lr_t / var_noise_, dtype=tf.float32)
            sq_gen = self.grad_bound(learning_rate=lr_t_tensor, beta=beta_t, sq_gen=sq_gen)

            self.training_step(self.x_train_tensor, self.y_train_tensor, var_noise_tensor, lr_t_tensor)

            if epoch % 50 == 0:
                template = 'Epoch {}, Loss: {}, Accuracy train: {}, gen_iclr: {}'
                tf.print(template.format(epoch + 1, self.train_loss.result(), self.train_accuracy.result() * 100,
                                         np.sqrt(2) / self.num_traindp * np.sqrt(sq_gen)*100))

            for test_images, test_labels in self.test_ds:
                self.test_step(test_images, test_labels)

            train_loss_all = np.append(train_loss_all,self.train_loss.result())
            train_accuracy_all = np.append(train_accuracy_all, self.train_accuracy.result())
            test_accuracy_all = np.append(test_accuracy_all, self.test_accuracy.result())
            sq_gen_all = np.append(sq_gen_all, sq_gen)
        return train_loss_all, train_accuracy_all, test_accuracy_all, sq_gen_all


class IncohBoundSimulator:
    def __init__(self):
        self.num_traindp = 20000  # it is not 50000 so that we can estimate the impact of different datasets
        self.z_Jc, self.z_J, self.test_ds = self.incoh_data_preprocess()
        self.model = self.model_def()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.05,
            decay_steps=40,
            decay_rate=0.90,
            staircase=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        self.var_0 = 1e-5
        self.var_inf = 1e-8
        self.var_rate_v = 0.5
        self.var_steps_v = 40

    def incoh_data_preprocess(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
        indx_shuffle = np.random.permutation(len(x_train))  # Data is shuffled
        # before sampling from it.
        x_train, y_train = x_train[indx_shuffle], y_train[indx_shuffle]
        x_train, y_train = x_train[:self.num_traindp], y_train[:self.num_traindp]  # first we take num_traindp samples
        J = np.random.randint(self.num_traindp, size=1)  # we select the index J between
        x_J_, y_J_ = x_train[J], y_train[J]  # this is J
        x_Jc_, y_Jc_ = np.delete(x_train, [J], axis=0), np.delete(y_train, [J], axis=0)  # this is the rest of point
        # the training set
        z_Jc = [tf.convert_to_tensor(x_Jc_), tf.convert_to_tensor(y_Jc_)]
        z_J = [tf.convert_to_tensor(x_J_), tf.convert_to_tensor(y_J_)]
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        return z_Jc, z_J, test_ds

    @tf.function
    def norm_list(self, ww_list):  # this function is for computing norm of params. params in TF is list of tensors
        norm_ww_sq = 0
        for ww in ww_list:
            norm_ww_sq = norm_ww_sq + tf.math.square(tf.norm(ww))
        return tf.sqrt(norm_ww_sq)

    def model_def(self):  # model definition
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu',
                                   kernel_initializer='he_uniform'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu',
                                   kernel_initializer='he_uniform'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(10, kernel_initializer='he_uniform')
        ])
        return model

    @tf.function
    def compute_gradients(self):  # this function computer gradients for other functions
        x_Jc, y_Jc = self.z_Jc
        with tf.GradientTape() as tape:
            predictions_Jc = self.model(x_Jc)
            loss_Jc = self.loss_fn(y_Jc, predictions_Jc)
        grad_Jc = tape.gradient(loss_Jc, self.model.trainable_variables)
        self.train_accuracy(y_Jc, predictions_Jc)  # the training error and accuracy
        self.train_loss(loss_Jc)  # are updated here

        x_J, y_J = self.z_J
        with tf.GradientTape() as tape:
            predictions_J = self.model(x_J)
            lossJ = self.loss_fn(y_J, predictions_J)
        grad_J = tape.gradient(lossJ, self.model.trainable_variables)
        self.train_accuracy(y_Jc, predictions_Jc)  # the training error and accuracy
        self.train_loss(loss_Jc)  # are updated here

        return grad_Jc, grad_J

    def learning_rate_tracker(self, iter_current_):
        decay_step_ = self.lr_schedule.decay_steps
        decay_rate_ = self.lr_schedule.decay_rate
        init_learning_rate_ = self.lr_schedule.initial_learning_rate
        staircase_ = self.lr_schedule.staircase
        if staircase_:
            p = np.floor(iter_current_ / decay_step_)
        else:
            p = iter_current_ / decay_step_
        return init_learning_rate_ * np.power(decay_rate_, p)

    def var_noise(self, iter_=0):  # noise variance scheduler
        var_n = self.var_inf + (self.var_0 - self.var_inf) * np.exp(
            -self.var_rate_v * np.floor(iter_ / self.var_steps_v))
        return var_n

    @tf.function
    def incoh_bound_sq(self, grad_Jc, grad_J, beta_t, lr_t, bound_sq):
        incoh = []
        zip_object = zip(grad_Jc, grad_J)
        for grad_Jc_i, grad_J_i in zip_object:
            incoh.append(grad_J_i - grad_Jc_i)
        incoh_norm_sq = tf.square(self.norm_list(incoh)) 
        bound_sq = bound_sq + beta_t * lr_t * 1 / (self.num_traindp ** 2) * incoh_norm_sq
        return bound_sq

    @tf.function
    def training_step(self, var_n, lr, gradients_):  # This is the training step, base on value of uJ we decide what is
        # the gradient for training.
        grad_Jc, grad_J = gradients_
        grad_training = []
        for grad_Jc_i, grad_J_i in zip(grad_Jc, grad_J):
            grad_training.append(
                (self.num_traindp - 1) / self.num_traindp * grad_Jc_i + 1 / self.num_traindp * grad_J_i)
        processed_grads = [self.noisy_gradient(g, var_noise=var_n, learning_rate=lr) for g in grad_training]
        self.optimizer.apply_gradients(zip(processed_grads, self.model.trainable_variables))

    @tf.function
    def noisy_gradient(self, g, var_noise, learning_rate):  # making gradient noisy
        # since by applying gd, tf will multiply both side by eta_t.
        g = g + tf.random.normal(tf.shape(g), stddev=tf.multiply(tf.sqrt(var_noise), tf.pow(learning_rate, -1)))
        return g

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images)
        self.test_accuracy(labels, predictions)

    def train(self, num_epochs=700):
        train_loss_all = np.array([])
        train_accuracy_all = np.array([])
        test_accuracy_all = np.array([])
        sq_gen_all = np.array([])
        gen_bound_sq = tf.convert_to_tensor(0.0)
        for epoch in range(num_epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()
            iter_t = self.optimizer.iterations.numpy()
            lr_t = self.learning_rate_tracker(iter_t)
            lr_t_tensor = tf.convert_to_tensor(lr_t, dtype=tf.float32)
            var_noise_ = self.var_noise(iter_=iter_t)
            var_noise_tensor = tf.convert_to_tensor(var_noise_, dtype=tf.float32)
            beta_t = 2 * lr_t / var_noise_
            beta_t_tensor = tf.convert_to_tensor(beta_t, dtype=tf.float32)
            grad_list = self.compute_gradients()  # accuracy and loss are computed inside this
            gen_bound_sq = self.incoh_bound_sq(grad_list[0], grad_list[1], beta_t_tensor, lr_t_tensor, gen_bound_sq)
            self.training_step(var_noise_tensor, lr_t_tensor, grad_list)

            for test_images, test_labels in self.test_ds:
                self.test_step(test_images, test_labels)

            if epoch % 100 == 0:
                template = 'Epoch {}, Loss: {}, Accuracy train: {}, incoh_bound: {}'
                print(template.format(epoch + 1,
                                      self.train_loss.result(),
                                      self.train_accuracy.result() * 100,
                                      1 / (2) * np.sqrt(gen_bound_sq) * 100))

            train_loss_all = np.append(train_loss_all, self.train_loss.result())
            train_accuracy_all = np.append(train_accuracy_all, self.train_accuracy.result())
            test_accuracy_all = np.append(test_accuracy_all, self.test_accuracy.result())
            sq_gen_all = np.append(sq_gen_all, gen_bound_sq)

        return train_loss_all, train_accuracy_all, test_accuracy_all, sq_gen_all



