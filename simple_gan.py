# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Model
import pandas as pd


def plot_data(x, x_):
    fig1 = plt.subplot(121)
    for i in range(len(x)):
        plt.plot(x[i])
    fig2 = plt.subplot(122)
    for i in range(len(x_)):
        plt.plot(x_[i])
    return fig1, fig2

# 生成正弦函数数据
def sample_real_data_1(n_samples=1000, dim=50):
    real_data = []
    x_vals = np.arange(0, 10, 10./dim)
    for _ in range(n_samples):
        a = np.random.normal(1, 0.1)
        b = np.random.normal()*100
        v = np.sin(x_vals * a + b)
        real_data.append(v)
    return np.array(real_data)

# 生成正太分布数据
def sample_real_data_2(n_samples=1000, dim=50):
    real_data = []
    for _ in range(n_samples):
        v = np.random.normal(10, 3, dim)
        real_data.append(v)
    return np.array(real_data)

# 生成任意线性函数数据
def sample_real_data(n_samples=1000, dim=50):
    real_data = []
    x_vals = np.arange(0, 10, 10./dim)
    for _ in range(n_samples):
        a = np.random.uniform(-10, 10)
        b = np.random.normal()*10.0
        v = x_vals * a + b
        real_data.append(v)
    return np.array(real_data)

def sample_noise_data(n_samples=1000, dim=2):
    noise_data = []
    for _ in range(n_samples):
        v = np.random.uniform(low=-1.0, high=1.0, size=dim)
        noise_data.append(v)
    return np.array(noise_data)

def get_model_G(inputs, out_dim=50):
    x = Dense(units=16, activation='relu')(inputs)
    G_output = Dense(units=out_dim)(inputs)
    model_G = Model(inputs, G_output)
    model_G.compile(optimizer=keras.optimizers.SGD(lr=1e-3), loss='binary_crossentropy')
    return model_G

def get_model_D(inputs):
    x = Dense(units=32, activation='relu')(inputs)
    x = Dense(units=8, activation='relu')(x)
    D_output = Dense(units=1, activation='sigmoid')(x)
    model_D = Model(inputs, D_output)
    model_D.compile(optimizer=keras.optimizers.SGD(lr=1e-3), loss='binary_crossentropy', metrics=['acc'])
    return model_D

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def make_GAN(GAN_inputs, model_G, model_D):
    set_trainability(model_D, False)
    x = model_G(GAN_inputs)
    GAN_out = model_D(x)
    model_GAN = Model(GAN_inputs, GAN_out)
    model_GAN.compile(optimizer=keras.optimizers.SGD(lr=1e-3), loss='binary_crossentropy', metrics=['acc'])
    return model_GAN

def sample_noise_data_Xy(n_samples=1000, dim=2):
    X = sample_noise_data(n_samples=n_samples, dim=dim)
    y = np.zeros(n_samples)
    return X, y

def sample_real_and_fake_data_Xy(model_G, n_samples=1000, dim_real=100, dim_noise=2):
    X_real = sample_real_data(n_samples=n_samples, dim=dim_real)
    noise = sample_noise_data(n_samples=n_samples, dim=dim_noise)
    X_fake = model_G.predict(noise)
    X = np.concatenate((X_real, X_fake), axis=0)
    y_real = np.ones(shape=(n_samples, 1))  # y_real = 1
    y_fake = np.zeros(shape=(n_samples, 1)) # y_fake = 0
    y = np.concatenate((y_real, y_fake), axis=0)
    return X, y

def pre_train(model_G, model_D, n_samples=1000, dim_real=100, dim_noise=2, batch_size=32, epochs=1):
    X, y = sample_real_and_fake_data_Xy(model_G, n_samples=n_samples, dim_real=dim_real, dim_noise=dim_noise)
    set_trainability(model_D, True)
    model_D.fit(X, y, batch_size=batch_size, epochs=epochs)

def train(model_GAN, model_G, model_D, k_D=1, k_G=1,
          n_samples=1000, dim_real=100, dim_noise=2,
          batch_size=32, epochs=10, verbose=True, v_freq=50):
    loss_list_D = []
    loss_list_G = []
    acc_list_D = []
    acc_list_G = []
    for epoch in range(epochs):

        # train D
        set_trainability(model_D, True)
        for _ in range(k_D):
            X_all, y_all = sample_real_and_fake_data_Xy(model_G, n_samples=n_samples, dim_real=dim_real, dim_noise=dim_noise)
            loss_D, acc_D = model_D.train_on_batch(X_all, y_all)
        loss_list_D.append(loss_D)
        acc_list_D.append(acc_D)

        # train G
        set_trainability(model_D, False)
        for _ in range(k_G):
            X_fake, y_fake = sample_noise_data_Xy(n_samples=n_samples, dim=dim_noise)
            y_fake = np.array([1 for _ in range(len(y_fake))])  # we want GAN can predict the fake data equal to 1.
            loss_G, acc_G = model_GAN.train_on_batch(X_fake, y_fake)
        loss_list_G.append(loss_G)
        acc_list_G.append(acc_G)

        if verbose:
            print('epoch: {} => D_loss: {:.3}, G_loss: {:.3}, acc_D: {:.3}, acc_G: {:.3}'.format(
                epoch+1, loss_D, loss_G, acc_D, acc_G))
            if (epoch) % v_freq == 0:
                x_real = sample_real_data(n_samples=n_samples, dim=dim_real)
                x_noise = sample_noise_data(dim=dim_noise)
                x_fake = model_G.predict(x_noise)
                fig1, fig2 = plot_data(x_real[0:3], x_fake[0:3])
                plt.pause(0.3)
                fig1.cla()
                fig2.cla()
    return loss_list_D, loss_list_G, acc_list_D, acc_list_G


### ---------- main ---------- ###

### hyper parameters ###
dim_real = 50
dim_noise = 2
n_samples = 500
epochs = 100000

G_in = keras.Input(shape=[dim_noise])
model_G = get_model_G(inputs=G_in, out_dim=dim_real)
model_G.summary()

D_in = keras.Input(shape=[dim_real])
model_D = get_model_D(inputs=D_in)
model_D.summary()

GAN_in = keras.Input(shape=[dim_noise])
model_GAN = make_GAN(GAN_in, model_G, model_D)
model_GAN.summary()

# load the pre-trained models
# print('loading the pre-trained models...')
# model_D = keras.models.load_model('./model/model_D.h5')
# model_G = keras.models.load_model('./model/model_G.h5')
# GAN_in = keras.Input(shape=[dim_noise])
# model_GAN = make_GAN(GAN_in, model_G, model_D)
# model_D.summary()
# model_G.summary()
# model_GAN.summary()

input('\npress enter to start training...\n')

print('start pre-training...')
np.random.seed(272)
pre_train(model_G, model_D,
          n_samples=n_samples, dim_real=dim_real, dim_noise=dim_noise,
          batch_size=32, epochs=10)

plt.ion()   # continuous display

print('start GAN training...')
loss_list_D, loss_list_G, acc_list_D, acc_list_G = train(model_GAN, model_G, model_D, k_D=3, k_G=100,
                                 n_samples=n_samples, dim_real=dim_real, dim_noise=dim_noise,
                                 batch_size=32, epochs=epochs, verbose=True)

plt.ioff()
plt.close('all')

x_real = sample_real_data(n_samples=n_samples, dim=dim_real)
x_noise = sample_noise_data(dim=dim_noise)
x_fake = model_G.predict(x_noise)
plot_data(x_real[0:3], x_fake[0:3])

# plot the loss of D and G
df_loss = pd.DataFrame(
    {
        'loss_D': loss_list_D,
        'loss_G': loss_list_G
    })
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(df_loss)
plt.legend(df_loss)
plt.show()

# save loss and acc
df_loss_and_acc = pd.DataFrame(
    {
        'loss_D': loss_list_D,
        'loss_G': loss_list_G,
        'acc_D': acc_list_D,
        'acc_G': acc_list_G
    })
df_loss_and_acc.to_csv('./model/df_loss_and_acc_simple_gan.csv', index=False)

# save the model D, G and GAN
model_D.save('./model/model_D.h5')
model_G.save('./model/model_G.h5')
