# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Reshape, Flatten, Dropout, BatchNormalization, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
import pandas as pd


class ImgGAN():

    def __init__(self,
                 img_rows=28, img_cols=28, img_channels=1, dim_noise=10, num=None):
        # initialize the basic parameters
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, img_channels)
        self.dim_noise = dim_noise
        optimizer = keras.optimizers.SGD(lr=0.001)

        # load the all X_real data from mnist dataset
        (self.X_real_all, y_real_all), (_, _) = keras.datasets.mnist.load_data()
        self.X_real_all = self.X_real_all / 255.0   # rescale the value of images to between 0 and 1
        self.X_real_all = np.expand_dims(self.X_real_all, axis=3)
        if num is not None:
            self.X_real_all = self.X_real_all[np.where(y_real_all == num)]

        # build and compile the discriminative model
        self.model_D = self.build_D()
        self.model_D.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.set_trainability(self.model_D, False)

        # build the generative model
        self.model_G = self.build_G()

        # build the combined model (GAN)
        noise = Input(shape=(self.dim_noise,))
        img_fake = self.model_G(noise)
        pred_of_fake = self.model_D(img_fake)
        self.model_combined = Model(noise, pred_of_fake)
        self.model_combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def build_G(self):
        '''
        Build the generative model.
        '''
        model = Sequential(name='model_G')
        model.add(Dense(64, input_shape=(self.dim_noise,), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(np.prod(self.img_shape), activation='sigmoid')) # ensure all the output values of G is between 0 and 1
        model.add(Reshape(self.img_shape))
        model.summary()
        noise = Input(shape=(self.dim_noise,))
        output_img = model(noise)
        return Model(noise, output_img)

    def build_D(self):
        '''
        Build the discriminative model.
        '''
        model = Sequential(name='model_D')
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        input_img = Input(shape=self.img_shape)
        output = model(input_img)
        return Model(input_img, output)

    def set_trainability(self, model, trainable=False):
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable
    
    def sample_real_data(self, n_samples=32):
        '''
        Generate a batch of real images from mnist dataset.
        '''
        rand_index = np.random.choice(len(self.X_real_all), n_samples, replace=False)
        return self.X_real_all[rand_index]
    
    def sample_noise_data(self, n_samples=32):
        '''
        Generate a batch of noise data randomly.
        '''
        return np.random.uniform(-1, 1, (n_samples, self.dim_noise))
    
    def pre_train(self, batch_size):
        n_samples = 1024
        self.set_trainability(self.model_D, True)
        X_real = self.sample_real_data(n_samples)
        noise_vector = self.sample_noise_data(n_samples)
        X_fake = self.model_G.predict(noise_vector)
        y_real = np.ones((n_samples, 1))   # y_real = 1
        y_fake = np.zeros((n_samples, 1))  # y_fake = 0
        X = np.concatenate((X_real, X_fake), axis=0)
        y = np.concatenate((y_real, y_fake), axis=0)
        self.model_D.fit(X, y, batch_size=batch_size, epochs=1)

    def train(self, epochs=1000, k_D=1, k_G=1, batch_size=32, verbose=True, v_freq=100):
        '''
        Training the GAN model
        '''
        self.pre_train(batch_size)

        loss_list_D = []
        loss_list_G = []
        acc_list_D = []
        acc_list_G = []
        y_real = np.ones((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))

        for epoch in range(1,epochs+1):

            # training the discriminative model (Model_D)
            self.set_trainability(self.model_D, True)
            for k in range(k_D):
                X_real = self.sample_real_data(n_samples=batch_size)
                noise_vector = self.sample_noise_data(n_samples=batch_size)
                X_fake = self.model_G.predict(noise_vector)
                X = np.concatenate((X_real, X_fake), axis=0)
                y = np.concatenate((y_real, y_fake), axis=0)
                rand_idx = np.random.permutation(len(y))
                X = X[rand_idx]
                y = y[rand_idx]
                loss_D, acc_D = self.model_D.train_on_batch(X, y)
            loss_list_D.append(loss_D)
            acc_list_D.append(acc_D)

            # training the generative model (Model_G)
            self.set_trainability(self.model_D, False)
            for k in range(k_G):
                noise_vector = self.sample_noise_data(n_samples=batch_size)
                # we want to train combined model can predict the fake data equal to 1.
                loss_G, acc_G = self.model_combined.train_on_batch(noise_vector, y_real)
            loss_list_G.append(loss_G)
            acc_list_G.append(acc_G)

            if verbose == True:
                print('epoch: {} => D [loss: {:.3f}, acc: {:.3f}]  G [loss: {:.3f}, acc: {:.3f}]'.format(
                    epoch, loss_D, acc_D, loss_G, acc_G))
                if epoch % v_freq == 0:
                    self.plot_fake_img(num=5, epoch=epoch)
        return loss_list_D, loss_list_G, acc_list_D, acc_list_G

    def plot_fake_img(self, num=5, epoch=-1):
        r, c = num, num
        noise = self.sample_noise_data(r*c)
        fake_imgs = self.model_G.predict(noise)
        # fake_imgs = 0.5 * fake_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(fake_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('./img/%d.png' % epoch)
        plt.close()

    def save_model(self, path_model_d, path_model_g):
        self.model_D.save(path_model_d)
        self.model_G.save(path_model_g)


if __name__ == '__main__':
    np.random.seed(272)
    img_gan = ImgGAN(dim_noise=2, num=2)
    loss_list_D, loss_list_G, acc_list_D, acc_list_G = img_gan.train(
        epochs=10000, k_D=10, k_G=100, batch_size=32, verbose=True, v_freq=100)
    
    # save the model D and G
    img_gan.save_model(path_model_d='./model/img_gan_model_D.h5', path_model_g='./model/img_gan_model_G.h5')

    # save loss and acc
    df_loss_and_acc = pd.DataFrame(
        {
            'loss_D': loss_list_D,
            'loss_G': loss_list_G,
            'acc_D': acc_list_D,
            'acc_G': acc_list_G
        })
    df_loss_and_acc.to_csv('./model/df_loss_and_acc_img_gan.csv', index=False)
