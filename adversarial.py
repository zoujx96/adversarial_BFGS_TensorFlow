'''
!/usr/bin/env python
-*- coding: utf-8 -*-
**************************************
@Time    : 2018/9/27 1:41
@Author  : Jiaxu Zou
@File    : adversarial.c 
**************************************
'''
import pickle
import tensorflow as tf 
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import keras.datasets.cifar10 as cifar10
import matplotlib.pyplot as plt 
import time as tm
import sourcef as sf

#Import MNIST and CIFAR10 datasets as reference

FLAGS = tf.app.flags.FLAGS
np.seterr(divide='ignore', invalid='ignore')
sess = tf.Session()

#Label the training and testing sets of MNIST

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
script_test = mnist.test.images
label_test = mnist.test.labels
script_train = mnist.train.images
label_train = mnist.train.labels

#Label the training and testing sets of CIFAR10

[[photo_train,name_train],[photo_test,name_test]] = cifar10.load_data()
photo_train = photo_train/255.0
photo_train = photo_train.reshape((50000,-1))
photo_test = photo_test/255.0
photo_test = photo_test.reshape((10000,-1))


#Main program

class adversarial(object):

#Construct TensorFlow Graph in self function

#x is the input datasets labeled above
#Classsize is the total number of labels in the dataset
#Architecture is a set of neuron layers defined in sourcef.py
#Activation is the activation function for some layers, you can choose 'relu','tanh' or 'sigmoid'
#Batchsize is the total number of examples processed at the same time to create adversarial examples
#Reshape is the tensor shape you want to reshape the input data as
#Softmax is the softmax layer type, you can choose 'regular' or 'sparse'
#Optimizer is the optimizer in TensorFlow used for training models, you can choose 'Adam' or 'SGD'
#Coeff is the coefficient of the optimization function, you can set it 1.0 for reference
#Epis is the threshold value of the gradient to judge when to stop the optimization
#Perturb is the threshold value of magnitude of noises imposed on original image to judge when to stop the optimization. Since it matters little, you can set it a big number
#Alpha is a hyperparameter in backtracking line search, you can set it 0.25
#Deltabeta is another hyperparameter in backtracking line search, you can set it 0.2
  def __init__(self,x,classsize,architecture,activation,batchsize,reshape,softmax,optimizer,coeff,epis,perturb,alpha,deltabeta):
    self.length = np.shape(x)[1]
    self.batch = batchsize
    self.point5 = tf.ones([self.batch,self.length])*0.5
    self.I = tf.cast(np.tile(np.identity(self.length),self.batch),tf.float32)
    self.c = tf.constant(coeff,tf.float32)
    self.epis = tf.constant(epis,tf.float32,[1,self.batch])
    self.perturb = tf.constant(perturb,tf.float32,[1,self.batch])
    self.alpha = tf.constant(alpha,tf.float32)
    self.deltabeta = tf.constant(deltabeta,tf.float32)
    self.D0 = tf.Variable(self.I,tf.float32)
    self.G0 = tf.Variable(tf.ones([self.length,self.batch])*1.0,tf.float32)
    self.lamda = tf.Variable(tf.ones([1,self.batch]),tf.float32)
    self.fx = tf.Variable(tf.ones([1,self.batch]),tf.float32)
    self.r0 = tf.Variable(tf.zeros([self.batch,self.length]),tf.float32)
    self.r2 = tf.Variable(tf.zeros([self.batch,self.length]),tf.float32)
    self.r = tf.Variable(tf.zeros([self.batch,self.length]),tf.float32)
    self.x = tf.placeholder(tf.float32,[None,self.length])
    self.y = tf.placeholder(tf.int32)
    self.learning_rate = tf.placeholder(tf.float32)
    self.train_or_adv=tf.placeholder(tf.bool)
    self.ximage=tf.cond(self.train_or_adv, lambda:tf.reshape(self.x,[-1,reshape[0],reshape[1],reshape[2]]), lambda:tf.reshape(0.5*(tf.tanh(self.r)+tf.ones([self.batch,self.length])*1.0),[-1,reshape[0],reshape[1],reshape[2]]))
    self.model = sf.dcnn(self.ximage,classsize,architecture,activation)
    if softmax=='regular':
      self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.model.logits)
      self.correct_prediction = tf.equal(tf.argmax(self.model.logits, 1), tf.argmax(self.y, 1))
    else:
      self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.y, [-1]), logits=self.model.logits)
      self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.model.logits,1),tf.int32),tf.reshape(self.y, [-1]))
    self.minimizer = tf.reduce_mean(self.loss)
    if optimizer=='Adam':
      self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.minimizer)
    else:
      self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.minimizer)
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    self.normr = tf.reshape(tf.sqrt(tf.reduce_sum((0.5*(tf.tanh(self.r)+tf.ones([self.batch,self.length])*1.0)-self.x)**2,1)),[1,self.batch])
    self.loss2 = self.c*self.normr
    self.objectfunction = self.loss2+self.loss
    self.G = tf.transpose(tf.reshape(tf.gradients(self.objectfunction,self.r),[self.batch,self.length]))
    for i in range(self.batch):
      if i==0:
	      self.d = (-1)*tf.matmul(tf.cast(self.D0[:,(self.length*i):(self.length*(i+1))],tf.float32),tf.reshape(self.G0[:,i],[self.length,1]))
      else:
	      self.d = tf.concat(1,[self.d,(-1)*tf.matmul(tf.cast(self.D0[:,(self.length*i):(self.length*(i+1))],tf.float32),tf.reshape(self.G0[:,i],[self.length,1]))])
    self.m = tf.reshape(tf.reduce_sum(tf.mul(self.d,self.G0),0),[1,self.batch])
    self.ob2 = self.fx+self.alpha*tf.mul(self.lamda,self.m)
    self.constraint1 = tf.cast(tf.reduce_all(tf.less_equal(tf.zeros([self.batch,self.length]),self.x+self.r0),1),tf.float32)
    self.constraint2 = tf.cast(tf.reduce_all(tf.less_equal(self.x+self.r0,tf.ones([self.batch,self.length])),1),tf.float32)
    self.flag_constraint = tf.reshape(tf.mul(self.constraint1,self.constraint2),[1,self.batch])
    self.normg = tf.reshape(tf.sqrt(tf.reduce_sum(self.G0**2,0)),[1,self.batch])
    self.flag_normg = tf.less_equal(self.epis,self.normg)
    self.normr0 = tf.reshape(tf.sqrt(tf.reduce_sum((0.5*(tf.tanh(self.r0)+tf.ones([self.batch,self.length])*1.0)-self.x)**2,1)),[1,self.batch])
    self.averpert = tf.reduce_mean(self.normr0)
    self.flag_normr = tf.less_equal(self.perturb,self.normr0)
    self.flag_norm = tf.cast(tf.logical_or(self.flag_normg,self.flag_normr),tf.float32)
    self.flag_update = self.flag_norm
    self.judgeoptimize = tf.reduce_sum(self.flag_update)
    self.flag_linesearch = tf.mul(tf.cast(tf.less(self.ob2, self.objectfunction),tf.float32),self.flag_update)
    self.judgelinesearch = tf.reduce_sum(self.flag_linesearch)
    self.deltalamda = self.deltabeta*tf.mul(self.lamda,self.flag_linesearch)
    self.updatelamda = tf.assign(self.lamda,self.lamda-self.deltalamda)
    self.true_incre = tf.mul(self.lamda,self.d)
    self.deltag = self.G-self.G0
    for i in range(self.batch):
      if i==0:
        self.D = tf.matmul(tf.matmul((tf.eye(self.length)-tf.matmul(tf.reshape(self.true_incre[:,i],[self.length,1]),tf.transpose(tf.reshape(self.deltag[:,i],[self.length,1])))/tf.matmul(tf.transpose(tf.reshape(self.deltag[:,i],[self.length,1])),tf.reshape(self.true_incre[:,i],[self.length,1]))),tf.cast(self.D0[:,(self.length*i):(self.length*(i+1))],tf.float32)),(tf.eye(self.length)-tf.matmul(tf.reshape(self.deltag[:,i],[self.length,1]),tf.transpose(tf.reshape(self.true_incre[:,i],[self.length,1])))/tf.matmul(tf.transpose(tf.reshape(self.deltag[:,i],[self.length,1])),tf.reshape(self.true_incre[:,i],[self.length,1]))))+tf.matmul(tf.reshape(self.true_incre[:,i],[self.length,1]),tf.transpose(tf.reshape(self.true_incre[:,i],[self.length,1])))/tf.matmul(tf.transpose(tf.reshape(self.deltag[:,i],[self.length,1])),tf.reshape(self.true_incre[:,i],[self.length,1]))
      else:
        self.D = tf.concat(1,[self.D,tf.matmul(tf.matmul((tf.eye(self.length)-tf.matmul(tf.reshape(self.true_incre[:,i],[self.length,1]),tf.transpose(tf.reshape(self.deltag[:,i],[self.length,1])))/tf.matmul(tf.transpose(tf.reshape(self.deltag[:,i],[self.length,1])),tf.reshape(self.true_incre[:,i],[self.length,1]))),tf.cast(self.D0[:,(self.length*i):(self.length*(i+1))],tf.float32)),(tf.eye(self.length)-tf.matmul(tf.reshape(self.deltag[:,i],[self.length,1]),tf.transpose(tf.reshape(self.true_incre[:,i],[self.length,1])))/tf.matmul(tf.transpose(tf.reshape(self.deltag[:,i],[self.length,1])),tf.reshape(self.true_incre[:,i],[self.length,1]))))+tf.matmul(tf.reshape(self.true_incre[:,i],[self.length,1]),tf.transpose(tf.reshape(self.true_incre[:,i],[self.length,1])))/tf.matmul(tf.transpose(tf.reshape(self.deltag[:,i],[self.length,1])),tf.reshape(self.true_incre[:,i],[self.length,1]))])
  	self.updateg = tf.assign(self.G0,self.G)
  	self.updater = tf.assign(self.r,self.r2+tf.transpose(tf.mul(self.true_incre,self.flag_update)))
    self.updated = tf.assign(self.D0,self.D)
    self.initiald = tf.assign(self.D0,self.I)
    self.initvalue = tf.cast(tf.less(self.x,self.point5),tf.float32)*0.1+tf.cast(tf.less_equal(self.point5,self.x),tf.float32)*(-0.1)
    self.initialr = tf.assign(self.r,initvalue)
    self.initialr0 = tf.assign(self.r0,initvalue)
    self.updater0 = tf.assign(self.r0,self.r)
    self.initiallamda = tf.assign(self.lamda,tf.ones([1,self.batch])*1.0)
    self.memor = tf.assign(self.r2,self.r)
    self.updatefx = tf.assign(self.fx,self.objectfunction)
    self.cheat = 0.5*(tf.tanh(self.r0)+tf.ones([self.batch,self.length])*1.0)
    sess.run(tf.global_variables_initializer())
    self.saver=tf.train.Saver(max_to_keep=1)
#X_train is the training set of images
#Y_train is the training set of labels
#X_test is the testing set of images
#Y_test is the testing set of labels
#Train_batch is the total number of examples trained at the same time
#Epoch is the total update times for the model
#Num_data is the total piece number of training data
#Keep_prob is the dropout rate of the dropout layer, you can set it 0.5 for reference
#Learning_rate is the learning rate of the training algorithm, you can set it 1e-4 for Adam optimizer for reference
#Rate_period is the changing period of the learning rate, you can set it 3000 for reference
#Decay_rate is the decay rate of the learning rate, you can set it 0.9 for reference
  def train(self,x_train,y_train,x_test,y_test,train_batch,epoch,num_data,keep_prob,learning_rate,rate_period,decay_rate):
    self.j = 0
    self.current_rate = learning_rate
    for i in range(epoch):
      if i % rate_period == 0:
        self.current_rate = self.current_rate*decay_rate
      if self.j==num_data/train_batch:
        self.j = 0
      self.x_train = x_train[train_batch*self.j:train_batch*(self.j+1)]
      self.y_train = y_train[train_batch*self.j:train_batch*(self.j+1)]
      if i % 100 == 0:
        self.train_accuracy = sess.run(self.accuracy,feed_dict={self.x: self.x_train, self.y: self.y_train, self.model.keep_prob: 1.0, self.train_or_adv:True})
        print('step %d, training accuracy %g' % (i, self.train_accuracy))
      sess.run(self.train_step, feed_dict={self.x: self.x_train, self.y: self.y_train, self.model.keep_prob: keep_prob, self.learning_rate: self.current_rate, self.train_or_adv:True})
      self.j = self.j+1
    self.saver.save(sess,'model/cifar10.ckpt')
    print('test accuracy %g' % sess.run(self.accuracy,feed_dict={self.x: x_test[1000:4000], self.y: y_test[1000:4000], self.model.keep_prob: 1.0, self.train_or_adv: True}))
  
#Num_examp is the total number of examples to process
#X_examp is the input set of images
#Y_examp is the input set of labels
  def create_adverexamp(self,num_examp,x_examp,y_examp):
    self.saver.restore(sess,'model/cifar10.ckpt')
    start = tm.time()
    self.output = 0
    self.perturbation = 0
    self.averaccuracy = 0
    initvalue = 0
    for i in range(num_examp/self.batch):
      self.x_examp = np.mat(x_examp[self.batch*i:self.batch*(i+1)])
      if i==num_examp/self.batch-1:
        self.y_examp = np.mat(y_examp[self.batch*i+1:self.batch*(i+1)])
        self.y_examp = np.concatenate((self.y_examp,np.mat(y_examp[0])),axis=0)
      else:
        self.y_examp = np.mat(y_examp[self.batch*i+1:self.batch*(i+1)+1])
      self.y_true = np.mat(y_examp[self.batch*i:self.batch*(i+1)])
      self.iteration = 0
      sess.run(self.initiald,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
      initvalue = np.arctanh(2.0*(sess.run(self.x,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})+sess.run(self.initvalue,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False}))-np.ones((self.batch,self.length))*1.0)
      sess.run(self.initialr,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
      sess.run(self.initialr0,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
      sess.run(self.updateg,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
      while True:
        sess.run(self.initiallamda,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
        sess.run(self.memor,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
        sess.run(self.updatefx,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
        self.correction = 0
        while True:
          sess.run(self.updater,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
          if sess.run(self.judgelinesearch,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})==0:
            break
          sess.run(self.updatelamda,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
          self.correction = self.correction+1
        sess.run(self.updater,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
        sess.run(self.updater0,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
        sess.run(self.updated,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
        sess.run(self.updateg,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
        if sess.run(self.judgeoptimize,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})==0:
          break
        self.iteration = self.iteration+1
      if i==0:
        self.output = sess.run(self.cheat,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
        self.perturbation = np.mat(sess.run(self.averpert,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False}))
        self.averaccuracy = np.mat(sess.run(self.accuracy,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False}))
      else:
        self.output = np.concatenate((self.output,sess.run(self.cheat,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})),axis=0)
        self.perturbation = np.concatenate((self.perturbation,np.mat(sess.run(self.averpert,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False}))),axis=0)
        self.averaccuracy = np.concatenate((self.averaccuracy,np.mat(sess.run(self.accuracy,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False}))),axis=0)
      print sess.run(self.accuracy,feed_dict={self.x: self.x_examp, self.y: self.y_examp, self.model.keep_prob: 1.0, self.train_or_adv: False})
      print sess.run(self.accuracy,feed_dict={self.x: self.x_examp, self.y: self.y_true, self.model.keep_prob: 1.0, self.train_or_adv: True})
    print np.mean(self.perturbation)
    print np.mean(self.averaccuracy)
    end = tm.time()
    print end-start
#Store the adversarial examples in a certain file (here is testadver.txt)
    foo=open('testadver.txt','wb')
    pickle.dump(self.output,foo)
    foo.close()
	
#Architecture for MNIST for reference
#Architecture for CIFAR10 for reference

architecture_mnist = [sf.conv(32,5,1,'SAME'),sf.pool(2,2),sf.conv(64,5,1,'SAME'),sf.pool(2,2),sf.conv_to_fc('flatten'),sf.dense(1024),sf.drop()]
architecture_cifar10 = [sf.conv(64,5,1,'SAME'),sf.pool(3,2),sf.conv(64,5,1,'SAME'),sf.pool(3,2),sf.conv_to_fc('flatten'),sf.dense(384),sf.dense(192),sf.drop()]

#An example main function to create adversarial examples for testing datasets of MNIST

advermodel = adversarial(script_train,10,architecture_mnist,'relu',100,[28,28,1],'regular','Adam',5.0,3.0,100.0,0.1,0.2)
advermodel.train(script_train,label_train,script_test,label_test,100,2000,50000,0.5,1e-4,3000,1.0)
advermodel.create_adverexamp(10000,script_test,label_test)
