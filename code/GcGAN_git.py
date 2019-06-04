# !usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class GcGAN():
    def __init__(self):
        self.noiseDims = 256
        self.batch_size = 64
        self.epochs = 821
        self.embeddingDim = 100*3
        self.inputDim_x = 777*3
        self.inputDim_y = 748
        self.inputDim = 3079
        self.lamda1 = 2.0
        self.lamda2 = 5.0
        self.lamda = 1.0

        self.compressDims = [self.embeddingDim]
        self.decompressDims = [self.inputDim]
        self.generatorDims_mlp = [512,self.embeddingDim]
        self.generatorDims = [self.noiseDims,self.noiseDims,self.embeddingDim]
        self.encoderDims = [512,64]
        self.discriminatorDims = [64,32,1]

        self.learning_rate = 0.0001
        self.l2scale = 0.01
        self.generatorTrainPeriod = 1
        self.discriminatorTrainPeriod = 1

        self.aeActivation = tf.nn.tanh
        self.generatorActivation = tf.nn.relu
        self.discriminatorActivation = tf.nn.relu
        self.encoderActivation = tf.nn.relu
        self.loss_type = 'wgan'


    def discriminator_loss(self,loss_func, real, fake):
        real_loss = 0
        fake_loss = 0

        if loss_func == 'wgan':
            real_loss = -tf.reduce_mean(real)
            fake_loss = tf.reduce_mean(fake)

        loss = real_loss + fake_loss

        return loss

    def generator_loss(self,loss_func, real, fake):
        real_loss = 0
        fake_loss = 0

        if loss_func == 'wgan':
            real_loss = tf.reduce_mean(real)
            fake_loss = -tf.reduce_mean(fake)

        loss = real_loss + fake_loss

        return loss

    def l2_norm(self,v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    def spectral_norm(self,w,name=None, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable(name, [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration,Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = self.l2_norm(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = self.l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm

    def print2file(self,buf, outFile):
        outfd = open(outFile, 'a')
        outfd.write(buf + '\n')
        outfd.close()


    def load_data(self):
        """
        Load data from file
        """
        root1 = "../medical_data_analysis/data/"
        data = pd.read_csv(os.path.join(root1, "medical_data_train.csv")).values
        print("data.shape:", data.shape)
        return data


    def bn(self,x, is_training=True):
        return tf.contrib.layers.batch_norm(x,decay=0.9,updates_collections=None,epsilon=1e-5,
                                       scale=True,is_training=is_training)


    def lrelu(self,x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)


    def GroupedCorrNN(self,input):
        """
        Training Grouped CorrNN
        """
        x_input = tf.slice(input, [0, 0], [self.batch_size, self.inputDim_x])
        y_input = tf.slice(input, [0, self.inputDim_x], [self.batch_size, self.inputDim_y])
        decodeVariables = {}
        with tf.variable_scope('autoencoder', regularizer=l2_regularizer(self.l2scale)):
            tempVec_x_zero = np.reshape(np.zeros(self.batch_size * self.inputDim_x), (self.batch_size, self.inputDim_x))
            tempVec_y_zero = np.reshape(np.zeros(self.batch_size * self.inputDim_y), (self.batch_size, self.inputDim_y))
            tempVec_x = tf.concat([x_input, tempVec_y_zero], axis=1)
            tempVec_y = tf.concat([tempVec_x_zero, y_input], axis=1)
            tempVec = input
            tempDim = self.inputDim
            i = 0

            for compressDim in self.compressDims[:-1]:
                W = tf.get_variable('aee_W_' + str(i), shape=[tempDim, compressDim],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('aee_b_' + str(i), shape=[compressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempVec_x = self.aeActivation(tf.add(tf.matmul(tempVec_x, W), b))
                tempVec_y = self.aeActivation(tf.add(tf.matmul(tempVec_y, W), b))
                tempDim = compressDim
                i += 1
            W = tf.get_variable('aee_W_' + str(i), shape=[tempDim, self.compressDims[-1]],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('aee_b_' + str(i), shape=[self.compressDims[-1]])
            H = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
            H1 = self.aeActivation(tf.add(tf.matmul(tempVec_x, W), b))
            H2 = self.aeActivation(tf.add(tf.matmul(tempVec_y, W), b))

            H_ZERO = np.reshape(np.zeros(self.batch_size * int(self.embeddingDim / 3)),
                                (self.batch_size, int(self.embeddingDim / 3)))

            i = 0
            tempDim = self.compressDims[-1]
            tempVec = H;tempVec_x = H1;tempVec_y = H2
            tempVec_H01, tempVec_H02, tempVec_H03 = tf.split(H, 3, 1)
            tempVec_H11, tempVec_H12, tempVec_H13 = tf.split(H1, 3, 1)
            tempVec_H21, tempVec_H22, tempVec_H23 = tf.split(H2, 3, 1)

            tempVec_h01 = tf.concat([tempVec_H01, H_ZERO, H_ZERO], axis=1)
            tempVec_h02 = tf.concat([H_ZERO, tempVec_H02, H_ZERO], axis=1)
            tempVec_h03 = tf.concat([H_ZERO, H_ZERO, tempVec_H03], axis=1)
            tempVec_h023 = tf.concat([H_ZERO, tempVec_H02, tempVec_H03], axis=1)

            tempVec_h11 = tf.concat([tempVec_H11, H_ZERO, H_ZERO], axis=1)
            tempVec_h12 = tf.concat([H_ZERO, tempVec_H12, H_ZERO], axis=1)
            tempVec_h13 = tf.concat([H_ZERO, H_ZERO, tempVec_H13], axis=1)
            tempVec_h123 = tf.concat([H_ZERO, tempVec_H12, tempVec_H13], axis=1)

            tempVec_h21 = tf.concat([tempVec_H21, H_ZERO, H_ZERO], axis=1)
            tempVec_h22 = tf.concat([H_ZERO, tempVec_H22, H_ZERO], axis=1)
            tempVec_h23 = tf.concat([H_ZERO, H_ZERO, tempVec_H23], axis=1)
            tempVec_h223 = tf.concat([H_ZERO, tempVec_H22, tempVec_H23], axis=1)

            for decompressDim in self.decompressDims[:-1]:
                W = tf.get_variable('aed_W_' + str(i), shape=[tempDim, decompressDim],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('aed_b_' + str(i), shape=[decompressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempVec_x = self.aeActivation(tf.add(tf.matmul(tempVec_x, W), b))
                tempVec_y = self.aeActivation(tf.add(tf.matmul(tempVec_y, W), b))

                tempVec_h01 = self.aeActivation(tf.add(tf.matmul(tempVec_h01, W), b))
                tempVec_h02 = self.aeActivation(tf.add(tf.matmul(tempVec_h02, W), b))
                tempVec_h03 = self.aeActivation(tf.add(tf.matmul(tempVec_h03, W), b))
                tempVec_h023 = self.aeActivation(tf.add(tf.matmul(tempVec_h023, W), b))

                tempVec_h11 = self.aeActivation(tf.add(tf.matmul(tempVec_h11, W), b))
                tempVec_h12 = self.aeActivation(tf.add(tf.matmul(tempVec_h12, W), b))
                tempVec_h13 = self.aeActivation(tf.add(tf.matmul(tempVec_h13, W), b))
                tempVec_h123 = self.aeActivation(tf.add(tf.matmul(tempVec_h123, W), b))

                tempVec_h21 = self.aeActivation(tf.add(tf.matmul(tempVec_h21, W), b))
                tempVec_h22 = self.aeActivation(tf.add(tf.matmul(tempVec_h22, W), b))
                tempVec_h23 = self.aeActivation(tf.add(tf.matmul(tempVec_h23, W), b))
                tempVec_h223 = self.aeActivation(tf.add(tf.matmul(tempVec_h223, W), b))

                tempDim = decompressDim
                decodeVariables['aed_W_' + str(i)] = W
                decodeVariables['aed_b_' + str(i)] = b
                i += 1
            W = tf.get_variable('aed_W_' + str(i), shape=[tempDim, self.decompressDims[-1]],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('aed_b_' + str(i), shape=[self.decompressDims[-1]])
            decodeVariables['aed_W_' + str(i)] = W
            decodeVariables['aed_b_' + str(i)] = b
            xy_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b))
            x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_x, W), b))
            y_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_y, W), b))


            h11x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h11, W), b))
            h11x_reconst = tf.slice(h11x_reconst, [0, 0], [self.batch_size, self.inputDim_x])
            h12y_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h12, W), b))
            h12y_reconst = tf.slice(h12y_reconst, [0, self.inputDim_x], [self.batch_size, self.inputDim_y])
            h13y_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h13, W), b))
            h13y_reconst = tf.slice(h13y_reconst, [0, self.inputDim_x], [self.batch_size, self.inputDim_y])
            h123y_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h123, W), b))
            h123y_reconst = tf.slice(h123y_reconst, [0, self.inputDim_x], [self.batch_size, self.inputDim_y])


            h12x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h12, W), b))
            h12x_reconst = tf.slice(h12x_reconst, [0, 0], [self.batch_size, self.inputDim_x])
            h22x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h22, W), b))
            h22x_reconst = tf.slice(h22x_reconst, [0, 0], [self.batch_size, self.inputDim_x])


            h21x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h21, W), b))
            h21x_reconst = tf.slice(h21x_reconst, [0, 0], [self.batch_size, self.inputDim_x])
            h22y_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h22, W), b))
            h22y_reconst = tf.slice(h22y_reconst, [0, self.inputDim_x], [self.batch_size, self.inputDim_y])
            h23y_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h23, W), b))
            h23y_reconst = tf.slice(h23y_reconst, [0, self.inputDim_x], [self.batch_size, self.inputDim_y])
            h223y_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_h223, W), b))
            h223y_reconst = tf.slice(h223y_reconst, [0, self.inputDim_x], [self.batch_size, self.inputDim_y])

            # Calculated loss
            xys_input = input
            L1 = tf.reduce_mean(-tf.reduce_sum(
                xys_input * tf.log(x_reconst + 1e-12) + (1. - xys_input) * tf.log(1. - x_reconst + 1e-12), 1), 0)
            L2 = tf.reduce_mean(-tf.reduce_sum(
                xys_input * tf.log(y_reconst + 1e-12) + (1. - xys_input) * tf.log(1. - y_reconst + 1e-12), 1), 0)
            L12 = tf.reduce_mean(-tf.reduce_sum(
                xys_input * tf.log(xy_reconst + 1e-12) + (1. - xys_input) * tf.log(1. - xy_reconst + 1e-12), 1), 0)
            L3 = tf.reduce_mean(-tf.reduce_sum(
                x_input * tf.log(h11x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - h11x_reconst + 1e-12),1), 0)
            L31 = tf.reduce_mean(-tf.reduce_sum(
                x_input * tf.log(h12x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - h12x_reconst + 1e-12), 1),0)
            L4 = tf.reduce_mean(-tf.reduce_sum(
                x_input * tf.log(h21x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - h21x_reconst + 1e-12), 1), 0)
            L41 = tf.reduce_mean(-tf.reduce_sum(
                x_input * tf.log(h22x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - h22x_reconst + 1e-12), 1),0)
            L5 = tf.reduce_mean(-tf.reduce_sum(
                y_input * tf.log(h123y_reconst + 1e-12) + (1. - y_input) * tf.log(1. - h123y_reconst + 1e-12), 1), 0)
            L6 = tf.reduce_mean(-tf.reduce_sum(
                y_input * tf.log(h223y_reconst + 1e-12) + (1. - y_input) * tf.log(1. - h223y_reconst + 1e-12), 1), 0)
            L7 = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.square(h12y_reconst - h13y_reconst), y_input), 1), 0) + \
                 tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.square(h22y_reconst - h23y_reconst), y_input), 1), 0)
            L71 = tf.reduce_mean(tf.reduce_sum(tf.multiply(h12y_reconst, 1 - y_input), 1), 0) + \
                  tf.reduce_mean(tf.reduce_sum(tf.multiply(h13y_reconst, 1 - y_input), 1), 0) + \
                  tf.reduce_mean(tf.reduce_sum(tf.multiply(h22y_reconst, 1 - y_input), 1), 0) + \
                  tf.reduce_mean(tf.reduce_sum(tf.multiply(h23y_reconst, 1 - y_input), 1), 0)


            # # Computational correlation
            H_mean1 = tf.reduce_mean(H1, axis=0)
            H_centered1 = H1 - H_mean1
            H_mean2 = tf.reduce_mean(H2, axis=0)
            H_centered2 = H2 - H_mean2
            corr_nr = tf.reduce_sum(tf.multiply(H_centered1, H_centered2), axis=0)  # axis =0
            corr_dr1 = tf.sqrt(tf.reduce_sum(tf.multiply(H_centered1, H_centered1), axis=0) + 1e-8)
            corr_dr2 = tf.sqrt(tf.reduce_sum(tf.multiply(H_centered2, H_centered2), axis=0) + 1e-8)
            corr_dr = tf.multiply(corr_dr1, corr_dr2)
            L8 = tf.reduce_sum(tf.div(corr_nr, corr_dr))


            loss_corrnet = L1 + L2 + L12 - self.lamda * L8
            loss_Lh = self.lamda1 * (L3 + L31 + L4 + L41) + self.lamda2 * (L5 + L6 - L7 + L71)
            loss_corr = - self.lamda * L8
            loss = loss_corrnet + loss_Lh

            loss_list = [L1, L2, L12, L3, L31, L4, L41, L5, L6, L7, L71, L8]

        return loss, loss_corr, decodeVariables, loss_list


    def generator_mlp(self,inputX,reuse=False,is_training=True):
        """
        Training generator with mlp
        """
        Activation=self.generatorActivation
        generatorDims=self.generatorDims_mlp
        tempDim=self.noiseDims
        tempVec=inputX
        with tf.variable_scope("generator",reuse=reuse):
            for i,genDim in enumerate(generatorDims[:-1]):
                W=tf.get_variable("W_"+str(i),shape=[tempDim, genDim],initializer=tf.contrib.layers.xavier_initializer())
                h=tf.matmul(tempVec,W)
                h2=self.bn(h,is_training=is_training)
                h3=Activation(h2)
                tempDim = genDim
                tempVec = h3
            W=tf.get_variable("W_"+str(i+1),shape=[tempDim,generatorDims[-1]],initializer=tf.contrib.layers.xavier_initializer())
            h=self.bn(tf.matmul(tempVec,W),True)
            output=self.aeActivation(h)
        return h,output


    def generator_dense(self,inputz,reuse=False,is_training=True):
        Activation = self.generatorActivation
        tempDim = self.noiseDims
        tempVec = inputz
        with tf.variable_scope("generator", reuse=reuse):
            """DenseNet层"""
            tempVec_tensor_acv = tf.reshape(tempVec, [-1, 1, tempDim, 1])  # acv内需要存储激活后的变量
            for i,temp in enumerate(self.generatorDims_dense[:-1]):
                W = tf.get_variable("W_" + str(i), shape=[self.noiseDims, tempDim],initializer=tf.contrib.layers.xavier_initializer())
                tempVec = tf.matmul(tempVec, W)
                c = tf.reshape(tempVec, [-1, 1, tempDim, 1])

                tempVec_tensor = tf.concat([c, tempVec_tensor_acv], axis=3)
                filter = tf.Variable(tf.random_normal([1, 1, tempVec_tensor.get_shape().as_list()[3], 1]))
                tempVec_tensor_temp = tf.nn.conv2d(tempVec_tensor, filter, strides=[1, 1, 1, 1], padding='VALID')
                tempVec = tf.reshape(tempVec_tensor_temp, [-1, tempDim])

                h = self.bn(tempVec, is_training=is_training)
                tempVec = Activation(h)
                tempVec_tensor_acv = tf.concat([tf.reshape(tempVec, [-1, 1, tempDim, 1]), tempVec_tensor_acv], axis=3)

            """全连接层"""
            W = tf.get_variable("W_" + str(i + 1), shape=[tempDim, self.generatorDims_dense[-1]],initializer=tf.contrib.layers.xavier_initializer())
            h = self.bn(tf.matmul(tempVec, W), is_training=is_training)
            output = self.aeActivation(h)

        return h, output



    def discriminator(self,inputX,reuse=None,sn=False):
        Activation = self.discriminatorActivation
        discriminatorDims= self.discriminatorDims
        tempDim = discriminatorDims[0]
        tempVec = inputX
        i=0
        with tf.variable_scope("discriminator",reuse=reuse):
            for i,discDim in enumerate(discriminatorDims[1:-1]):
                W=tf.get_variable("W_"+str(i),shape=[tempDim,discDim],initializer=tf.contrib.layers.xavier_initializer())
                if sn:
                    W=self.spectral_norm(W,name="u"+str(i))
                h=tf.matmul(tempVec,W)
                h2=Activation(h)
                tempVec = h2
                tempDim = discDim
            W=tf.get_variable("W_"+str(i+1),shape=[tempDim,discriminatorDims[-1]],initializer=tf.contrib.layers.xavier_initializer())
            if sn:
                W = self.spectral_norm(W,name="u"+str(i+1))
            h=tf.matmul(tempVec,W)
            output=tf.nn.sigmoid(tf.matmul(tempVec,W))
            return h,output


    def encoder(self, x, is_training=True, reuse=False, sn=False):
        Activation = self.encoderActivation
        with tf.variable_scope('encoder', reuse = reuse):
            i=0;tempDim_x= self.inputDim
            tempVec_x = x
            for compressDim in self.encoderDims[:-1]:
                W = tf.get_variable('enc_W_' + str(i), shape=[tempDim_x, compressDim],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('enc_b_' + str(i), shape=[compressDim])
                if sn:
                    W=self.spectral_norm(W,name="u"+str(i))
                tempVec_x = Activation(tf.add(tf.matmul(tempVec_x, W), b))
                tempDim_x = compressDim
                i += 1
            W = tf.get_variable('enc_W_' + str(i), shape=[tempDim_x, self.encoderDims[-1]],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('enc_b_' + str(i), shape=[self.encoderDims[-1]])
            if sn:
                W = self.spectral_norm(W,name="u"+str(i))
            tempVec_x = tf.nn.sigmoid(tf.add(tf.matmul(tempVec_x, W), b))

        return tempVec_x


    def buildDecoder(self,embedding,decodeVariables):
        tempVec = embedding
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(
                tf.add(tf.matmul(tempVec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))
            i += 1
        x_fake = tf.nn.sigmoid(
            tf.add(tf.matmul(tempVec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))

        return x_fake

    def buildnet(self,checkpoint=None,modelPath="",net_style=None):
        self.inputs = tf.placeholder(dtype=tf.float32,shape=[None, self.inputDim])
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.noiseDims])

        print("running GcGAN.......")
        # Training Grouped CorrNN
        loss_ae, loss_corr, decodeVariables, loss_list = self.GroupedCorrNN(self.inputs)

        if net_style== "mlp":
            print("===============generator is MLP.........====================")
            _,x_fake_embedding = self.generator_mlp(self.z, is_training=True, reuse=False)
        if net_style== "Dense":
            print("===============generator is Dense.........====================")
            _, x_fake_embedding = self.generator_dense(self.z, is_training=True, reuse=False)
        x_fake = self.buildDecoder(x_fake_embedding,decodeVariables)

        #T-GANs
        x_real_encoder = self.encoder(self.inputs, is_training=True, reuse=False, sn=True)
        x_fake_encoder = self.encoder(x_fake, is_training=True, reuse=True, sn=True)
        x_real_fake = tf.subtract(x_real_encoder, x_fake_encoder)
        x_fake_real = tf.subtract(x_fake_encoder, x_real_encoder)
        x_real_fake_score,_ = self.discriminator(x_real_fake, reuse=False, sn=True)
        x_fake_real_score,_ = self.discriminator(x_fake_real, reuse=True, sn=True)

        # get loss for discriminator
        self.d_loss = self.discriminator_loss(self.loss_type, real=x_real_fake_score, fake=x_fake_real_score)

        # get loss for generator
        self.g_loss = self.generator_loss(self.loss_type, real=x_real_fake_score, fake=x_fake_real_score)


        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars=tf.trainable_variables()
        ae_vars = [var for var in t_vars if "autoencode" in var.name ]
        d_vars = [var for var in t_vars if "discriminator" in var.name or 'encoder' in var.name]
        g_vars = [var for var in t_vars if "generator" in var.name or 'aed_' in var.name]
        aed_vars = [var for var in t_vars if 'aed_' in var.name]
        aee_vars = [var for var in t_vars if 'aee_' in var.name]

        optimize_ae = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0,beta2=0.9).minimize(loss_ae, var_list=ae_vars)
        optimize_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0,beta2=0.9).minimize(self.d_loss, var_list=d_vars)
        optimize_g=tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0,beta2=0.9).minimize(self.g_loss,var_list=g_vars)
        optimize_aed = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0,beta2=0.9).minimize(loss_ae,var_list=aed_vars)
        optimize_aee = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0,beta2=0.9).minimize(loss_corr,var_list=aee_vars)

        initOp = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=0)
        if not os.path.exists("result"):
            os.makedirs("result")
        logFile = "result/result_GcGAN.txt"

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        with tf.Session(config=config) as sess:
            if modelPath=="":
                sess.run(initOp)
            else:
                saver.restore(sess,modelPath)

            data = self.load_data()
            nbatchs = int((data.shape[0] / self.batch_size))

            for epoch in range(self.epochs):
                np.random.shuffle(data)
                for i in range(nbatchs):
                    start=i*self.batch_size;end=(i+1)*self.batch_size
                    batchX=data[start:end]

                    "updata disc"
                    for _ in range(self.discriminatorTrainPeriod):
                        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.noiseDims])
                        discloss,_=sess.run([self.d_loss,optimize_d],feed_dict={self.inputs:batchX,self.z:batch_z})

                    "updata gene"
                    for _ in range(self.generatorTrainPeriod):
                        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.noiseDims])
                        geneloss,_,ae_loss,_,Loss_list,_=sess.run([self.g_loss,optimize_g,loss_ae,optimize_aed,loss_list,
                                                optimize_ae],feed_dict={self.inputs:batchX,self.z:batch_z})

                buf="Epoch:%3d , d_loss:%5f, g_loss:%5f,ae_loss:%s,Loss_list:%s"%(epoch,discloss,geneloss,ae_loss,Loss_list)
                print(buf)
                self.print2file(buf, logFile)
                if epoch>=400 and epoch%20==0:
                    saver.save(sess, checkpoint, global_step=epoch)

    def generateData(self,modelPath=None,nsamples=None,batchsize=100,outFile=None,net_style=None):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.inputDim])
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.noiseDims])
        bn_train = tf.placeholder('bool')

        # Training Grouped CorrNN
        loss_ae, loss_corr, decodeVariables, loss_list = self.GroupedCorrNN(self.inputs)

        if net_style=="mlp":
            _, x_emb = self.generator_mlp(self.z, is_training=bn_train)
        if net_style=="Dense":
            _,x_emb = self.generator_dense(self.z,is_training=bn_train)
        x_fake = self.buildDecoder(x_emb, decodeVariables)

        saver = tf.train.Saver()
        outputVec = []
        with tf.Session() as sess:
            saver.restore(sess,modelPath)
            print('burning in')
            for i in range(1000):
                randomX = np.random.normal(size=(batchsize, self.noiseDims))
                output = sess.run(x_fake, feed_dict={self.z: randomX, bn_train: True})

            print('generating')
            nBatches = int(np.ceil(float(nsamples) / float(batchsize)))
            for i in range(nBatches):
                randomX = np.random.normal(size=(batchsize, self.noiseDims))
                output = sess.run(x_fake, feed_dict={self.z: randomX, bn_train: False})
                outputVec.extend(output)
            outputMat = np.array(outputVec)
            np.save(outFile, outputMat)
            print(outputMat.shape)

        """ translate float into int"""
        print("outputMat.shape:", outputMat.shape)
        data_train = []
        for li in outputMat:
            raw_data = []
            for i in range(int(self.inputDim_x/3)):
                icd_vec = [0, 0, 0]
                lj = li[i * 3:(i + 1) * 3]
                index = np.argsort(lj)[-1]
                if lj[index] > 0.5:
                    icd_vec[index] = 1
                raw_data.extend(icd_vec)

            for lj in li[self.inputDim_x:]:
                if lj >= 0.5:
                    raw_data.append(1)
                else:
                    raw_data.append(0)

            data_train.append(raw_data)
        print(np.array(data_train))
        outputMat = np.array(data_train)

        index1 = list(set(np.where(outputMat[:, :self.inputDim_x])[0]))
        index2 = list(set(np.where(outputMat[:, self.inputDim_x:])[0]))
        index = [li for li in index1 if li in index2]

        print("after translate outputMat.shape:", outputMat[index].shape)
        print("Proportion of qualified data：", len(index), outputMat.shape[0], 1.0 * len(index) / outputMat.shape[0])
        np.save(outFile, outputMat[index])


if __name__=="__main__":
    clf = GcGAN()

    # # #Training network
    # net_style indicates whether the generator uses mlp connection or dense connection
    clf.buildnet(checkpoint="checkpoint_GcGAN_mlp/save_net.ckpt",net_style="mlp")

    # # #Generating data
    # clf.generateData(modelPath='checkpoint_GcGAN_mlp_git/save_net.ckpt-620',
    #                  outFile = 'data/gene_GcGAN_mlp_git.npy',nsamples = 60000,
    #                  net_style="mlp")
