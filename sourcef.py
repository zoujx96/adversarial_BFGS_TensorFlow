import pickle
import tensorflow as tf
import numpy as np


np.seterr(divide='ignore', invalid='ignore')
sess = tf.Session()

#Convolution layer

class conv(object):
	def __init__(self,n_channels,kernel_size=3,stride=1,padding='SAME'):
		self.n_channels=n_channels
		self.k=kernel_size
		self.s=stride
		self.pad=padding

#Pooling layer

class pool(object):
	def __init__(self,window_size=2,stride=None):
		self.k=window_size
		self.s=stride

#Dense layer

class dense(object):
	def __init__(self,n_neurons):
		self.n_neurons=n_neurons

#Dropout layer
		
class drop(object):
	def __init__(self):
		pass

class conv_to_fc(object):
	def __init__(self,transformation='flatten'):
		self.transformation=transformation


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

class dcnn(object):
	def __init__(self,input,num_class,arch,activate):
		self.keep_prob=tf.placeholder(tf.float32)
		current=input
		#build graph here
		for layer in arch:
			if isinstance(layer,conv):
				n_ch_in=int(current.get_shape()[-1])
				W=weight_variable(shape=[layer.k,layer.k,n_ch_in,layer.n_channels])
				B=bias_variable(shape=[layer.n_channels])
				current=tf.nn.convolution(current,W,padding=layer.pad,strides=(layer.s,layer.s))
				current=current+B
				if activate=='relu':
					current=tf.nn.relu(current)
				elif activate=='tanh':
					current=tf.nn.tanh(current)
				elif activate=='sigmoid':
					current=tf.nn.sigmoid(current)
				
			elif isinstance(layer,pool):
				if layer.s==None:
					layer.s=layer.k
				current=tf.nn.max_pool(current,[1,layer.k,layer.k,1],[1,layer.s,layer.s,1],'SAME')
			
			elif isinstance(layer,conv_to_fc):
				if layer.transformation=='flatten':
					current=tf.contrib.layers.flatten(current)
				elif layer.transformation=='averagepool':
					ksize=[1,current.get_shape()[1],current.get_shape()[2],1]
					strides=[1,current.get_shape()[1],current.get_shape()[2],1]
					current=tf.nn.avg_pool(current,ksize=ksize,strides=strides,padding='VALID',data_format='NHWC')
					current=tf.contrib.layers.flatten(current)
				else:
					raise(ValueError('unknown conv to fc transformation'))
			elif isinstance(layer,dense):
				n_neurons_in=int(current.get_shape()[-1])
				W=weight_variable(shape=[n_neurons_in,layer.n_neurons])
				B=bias_variable(shape=[layer.n_neurons])
				current=tf.matmul(current,W)
				current=current+B
				if activate=='relu':
					current=tf.nn.relu(current)
				elif activate=='tanh':
					current=tf.nn.tanh(current)
				elif activate=='sigmoid':
					current=tf.nn.sigmoid(current)

			elif isinstance(layer,drop):
				current=tf.nn.dropout(current,self.keep_prob)
			else:
				raise(ValueError('unknown layer added'))
		#one last regressor layer:
		n_neurons_in=int(current.get_shape()[-1])
		W=weight_variable(shape=[n_neurons_in,num_class])
		B=bias_variable(shape=[num_class])
		current=tf.matmul(current,W)
		current=tf.add(current,B)

		self.logits=current
		








