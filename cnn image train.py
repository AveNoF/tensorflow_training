#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  cnn image train.py
#  
#  Copyright 2017 AveNoF-AI <avenof-ai@avenofai-Inspiron-15-7000-Gaming>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import os
import scipy.misc
import scipy.io
import numpy as np
import tensorflow as tf

sess = tf.Session()

#imagesfile

original_image_file = 'book_cover.jpg'
style_image_file = 'starry_night.jpg'

#info image

vgg_path ='imagenet-vgg-verydeep-19.mat'
original_image_weight = 5.0
style_image_weight = 500.0
regularization_weight = 100
learning_rate = 0.001
generations = 5000
output_generations = 250

#loadimage
original_image = scipy.misc.imread(original_image_file)
style_image = scipy.misc.imread(style_image_file)
#adapt
target_shape = original_image.shape
style_image = scipy.misc.imresize(style_image,target_shape[1] / style_image.shape[1])

vgg_layers = ['conv1_1','relu1_1',
			  'conv1_2','relu1_2','pool1',
			  'conv2_1','relu2_1',
			  'conv2_2','relu2_2','pool2',
			  'conv3_1','relu3_1',
			  'conv3_2','relu3_2',
			  'conv3_3','relu3_3','pool3',
			  'conv4_1','relu4_1',
			  'conv4_2','relu4_2',
			  'conv4_3','relu4_3',
			  'conv4_4','relu4_4','pool4',
			  'conv5_1','relu5_1',
			  'conv5_2','relu5_2',
			  'conv5_3','relu5_3',
			  'conv5_4','relu5_4']

#mat

def extract_net_info(path_to_params):
	vgg_data = scipy.io.loadmat(path_to_params)
	normalization_matrix = vgg_data['normalization'][0][0][0]
	mat_mean = np.mean(normalization_matrix, axis=(0,1))
	network_weights = vgg_data['layers'][0]
	return(mat_mean, network_weights)

def vgg_network(network_weights, init_image):
	network = {}
	image = init_image
	
	for i, layer in enumerate(vgg_layers):
		if layer[0] == 'c':
			weights, bias = network_weights[i][0][0][0][0]
			weights = np.transpose(weights, (1, 0, 2, 3))
			bias = bias.reshape(-1)
			conv_layer = tf.nn.conv2d(image,
									  tf.constant(weights),
									  (1, 1, 1, 1),
									  'SAME')
			images = tf.nn.bias_add(conv_layer, bias)
		elif layer[0] == 'r':
			image = tf.nn.relu(image)	
		else:
			image = tf.nn.max_pool(image,
								   (1, 2, 2, 1),
								   (1, 2, 2, 1),
								   'SAME')
		network[layer] = image
	return(network)

original_layer = 'relu4_2'
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

normalization_mean, network_weights = extract_net_info(vgg_path)

shape = (1,) + original_image.shape
style_shape = (1,) + style_image.shape
original_features = {}
style_features = {}


image = tf.placeholder('float', shape=shape)
vgg_net = vgg_network(network_weights, image)



original_minus_mean = ariginal_image - normalization_mean
original_norm = np.array([original_minus_mean])
original_features[original_layer] = \
	sess.run(vgg_net[original_layer], feed_dict={image: original_norm})



image = tf.placeholder('float', shape=style_shape)
vgg_net = vgg_network(network_weights, image)
style_minus_mean = style_image - normalization_mean
style_norm =  np.array([style_minus_mean])

for layer in style_layers:
	layer_output = sess.run(vgg_net[layer], feed_dict={image: style_norm})
	layer_output = np.reshape(layer_output, (-1, layer_output.shape[3]))
	style_gram_matrix = np.matmul(layer_output.T,
								  layer_output) / layer_output.size
	style_features[layer] = style_gram_matrix

initial = tf.random_normal(shape) * 0.256
image = tf.Variable(initial)
vgg_net = vgg_network(network_weights, image)


original_loss = original_image_weight * (2 * tf.nn.l2_lpss(
	vgg_net[original_layer] - original_features[original_layer])/
	original_features[original_layer].size)

style_loss = 0
style_losses = []
for style_layer in style_features:
	


def main(args):
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
