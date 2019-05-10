import tensorflow as tf 
import tensorflow.contrib.slim as slim
import numpy as np 
import cv2
import random
import math
from utils import *
import os
from sklearn.utils import shuffle
from roi_pool import roi_pool
import pandas as pd
from shutil import copyfile
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _base_inference(inputs):
	with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu , trainable = False, \
			weights_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01), \
			weights_regularizer = slim.l2_regularizer(0.0005)):
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding = 'SAME', scope='conv1', trainable = False)
		net = slim.max_pool2d(net, [2, 2], padding='VALID', scope = 'pool1')
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], padding = 'SAME', scope='conv2', trainable = False)
		net = slim.max_pool2d(net, [2, 2], padding='VALID', scope = 'pool2')
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding = 'SAME', scope='conv3')
		net = slim.max_pool2d(net, [2, 2], padding='VALID', scope = 'pool3')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], padding = 'SAME', scope='conv4')
		net = slim.max_pool2d(net, [2, 2], padding='VALID', scope = 'pool4')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], padding = 'SAME', scope='conv5')
	return net

def name_in_checkpoint(v):
	return 'vgg_16/' + v.op.name

def fine_turning_vgg16():
	variables_to_restore = slim.get_model_variables()
	variables_to_restore = {name_in_checkpoint(v):v for v in variables_to_restore if v.name.startswith('conv')}
	restorer = tf.train.Saver(variables_to_restore)
	return restorer

def _inference(x, k_anchors, name=None, dim_reduce = 16, aspect_ratios = [1, 2, 4], scales = [32, 64, 128], cls_thresh = 0.7):

	with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu, trainable = False, \
			weights_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01), \
			weights_regularizer = slim.l2_regularizer(0.0005)):

		# Itermediate layer
		itermediate_layer = slim.conv2d(x, 512, [3, 3], padding = 'SAME', scope = 'itermediate_layer')

		# Class regression
		RPN_cls = slim.conv2d(itermediate_layer, 2*k_anchors, [1, 1], padding = 'VALID', scope = 'classification', activation_fn = None)

		# Bounding box regression
		RPN_reg = slim.conv2d(itermediate_layer, 4*k_anchors, [1, 1], padding = 'VALID', scope = 'regression', activation_fn = None)

	# Get feature map shape
	ftmap_shape = tf.shape(itermediate_layer)

	confidence_list = tf.reshape(RPN_cls, [-1, 2])
	regression_list = tf.reshape(RPN_reg, [-1, 4])
	print('Itermediate Layer shape: ')
	print(itermediate_layer.shape)
	print('Confidence Layer shape: ')
	print(RPN_cls.shape)
	print('Bounding boxes Regression Layer shape: ')
	print(RPN_reg.shape)
	print('Confidence list shape: ')
	print(confidence_list.shape)
	print('Bounding boxes Regression list shape: ')
	print(regression_list.shape)

	probabilities = tf.nn.softmax(confidence_list, axis = 1)
	probabilities = probabilities[:, 1]

	return confidence_list, regression_list, probabilities

def _predict_boxes(probabilities, regression_list, ft_shape, placeholder_anchors, placeholder_anchors_cleaned, cls_thresh = 0.7, top_k = 128, nms_threshold = 0.7, dim_reduce = 16):


	# Remove unclean boxes
	cleaned_probabilites = tf.gather(probabilities, placeholder_anchors_cleaned)
	cleaned_regression = tf.gather(regression_list, placeholder_anchors_cleaned)
	cleaned_anchors = tf.gather(placeholder_anchors, placeholder_anchors_cleaned)
	
	# Remove boxes by threshold
	indices = tf.where(cleaned_probabilites > cls_thresh)
	cleaned_probabilites = tf.gather(cleaned_probabilites, indices)
	cleaned_probabilites = tf.reshape(cleaned_probabilites, [-1])
	cleaned_regression = tf.gather(cleaned_regression, indices)
	cleaned_regression = tf.reshape(cleaned_regression, [-1, 4])
	cleaned_anchors = tf.gather(cleaned_anchors, indices)
	cleaned_anchors = tf.reshape(cleaned_anchors, [-1, 4])

	# Get predict box
	predict_boxes_x = tf.add(tf.multiply(cleaned_anchors[:, 2], cleaned_regression[:, 0]), cleaned_anchors[:, 0])
	predict_boxes_y = tf.add(tf.multiply(cleaned_anchors[:, 3], cleaned_regression[:, 1]), cleaned_anchors[:, 1])
	predict_boxes_w = tf.multiply(cleaned_anchors[:, 2], tf.exp(cleaned_regression[:, 2]))
	predict_boxes_h = tf.multiply(cleaned_anchors[:, 3], tf.exp(cleaned_regression[:, 3]))
	
	# Clip to fit image boundary
	predict_boxes_x1 = predict_boxes_x - predict_boxes_w/2
	predict_boxes_x1 = tf.clip_by_value(predict_boxes_x1, 0.0, ft_shape[2]*dim_reduce)
	predict_boxes_y1 = predict_boxes_y - predict_boxes_h/2
	predict_boxes_y1 = tf.clip_by_value(predict_boxes_y1, 0.0, ft_shape[1]*dim_reduce)
	predict_boxes_x2 = predict_boxes_x + predict_boxes_w/2
	predict_boxes_x2 = tf.clip_by_value(predict_boxes_x2, 0.0, ft_shape[2]*dim_reduce)
	predict_boxes_y2 = predict_boxes_y + predict_boxes_h/2
	predict_boxes_y2 = tf.clip_by_value(predict_boxes_y2, 0.0, ft_shape[1]*dim_reduce)

	# Concatenate
	predict_boxes_x1 = tf.reshape(predict_boxes_x1, (-1, 1))
	predict_boxes_y1 = tf.reshape(predict_boxes_y1, (-1, 1))
	predict_boxes_x2 = tf.reshape(predict_boxes_x2, (-1, 1))
	predict_boxes_y2 = tf.reshape(predict_boxes_y2, (-1, 1))
	img_ind = tf.zeros_like(predict_boxes_x1)

	predict_boxes = tf.concat([img_ind, predict_boxes_x1, predict_boxes_y1, predict_boxes_x2, predict_boxes_y2], axis = -1)

	# Reduce boxes by NMS
	# Using tf.image.non_max_suppression
	cleaned_indices = tf.image.non_max_suppression(predict_boxes[:, 1:], cleaned_probabilites, max_output_size = top_k, iou_threshold = nms_threshold)
	nms_cleaned_probabilites = tf.gather(cleaned_probabilites, cleaned_indices)
	nms_cleaned_probabilites = tf.reshape(nms_cleaned_probabilites, [-1])
	nms_predict_boxes = tf.gather(predict_boxes, cleaned_indices)

	#return predict_boxes, cleaned_probabilites
	return nms_predict_boxes, nms_cleaned_probabilites

def format_boxes_proposal(boxes_proposal):
	# Convert from x1, y1, x2, y2 to x, y, w, h
	x = (boxes_proposal[:, 2] + boxes_proposal[:, 0])/2
	x = tf.reshape(x, [-1, 1])
	y = (boxes_proposal[:, 3] + boxes_proposal[:, 1])/2
	y = tf.reshape(y, [-1, 1])
	w = boxes_proposal[:, 2] - boxes_proposal[:, 0]
	w = tf.reshape(w, [-1, 1])
	h = boxes_proposal[:, 3] - boxes_proposal[:, 1]
	h = tf.reshape(h, [-1, 1])
	return tf.concat([x, y, w, h], axis = -1)

def get_real_result(boxes_proposal, offsets):
	boxes_proposal = format_boxes_proposal(boxes_proposal)
	# Box proposal in x, y, w, h format
	x = tf.subtract(boxes_proposal[:, 0], tf.multiply(offsets[:, 0], boxes_proposal[:, 2]))
	x = tf.reshape(x, [-1, 1])
	y = tf.subtract(boxes_proposal[:, 1], tf.multiply(offsets[:, 1], boxes_proposal[:, 3]))
	y = tf.reshape(y, [-1, 1])
	w = tf.div(boxes_proposal[:, 2], tf.exp(offsets[:, 2]))
	w = tf.reshape(w, [-1, 1])
	h = tf.div(boxes_proposal[:, 3], tf.exp(offsets[:, 3]))
	h = tf.reshape(h, [-1, 1])
	return tf.concat([x, y, w, h], axis = -1)

def _fastRCNN(feature_map, boxes_proposal, pooled_width = 7, pooled_height = 7, total_class = 10):
	pooled_features = roi_pool(feature_map, boxes_proposal)
	print('ROI Layer shape: {}'.format(pooled_features.shape))
	with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu, \
			weights_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01), \
			weights_regularizer = slim.l2_regularizer(0.0005)):
		rcnn_feature = slim.conv2d(pooled_features, 256, [1, 1], scope = 'rcnn_Feature')
		print('RCNN Feature shape: {}'.format(rcnn_feature.shape))
		flatted = tf.contrib.layers.flatten(rcnn_feature)
		iter_fl = tf.contrib.layers.fully_connected(flatted, 2048, activation_fn = tf.nn.relu, scope='rcnn_iter')
		
	iter_fl_cls = tf.contrib.layers.fully_connected(iter_fl, 2048, activation_fn = tf.nn.relu, scope='rcnn_iter_cls')
	drop_out = tf.nn.dropout(iter_fl_cls, keep_prob = 1)
	print('Add dropout 0.25 !')
	rcnn_cls = tf.contrib.layers.fully_connected(drop_out, total_class+1, activation_fn = None, scope='rcnn_cls')
	print('Classification shape: {}'.format(rcnn_cls.shape))

	iter_fl_reg = tf.contrib.layers.fully_connected(iter_fl, 1024, activation_fn = tf.nn.relu, scope='rcnn_iter_reg')
	rcnn_reg = tf.contrib.layers.fully_connected(iter_fl_reg, 4, activation_fn = None, scope='rcnn_reg')
	print('Regression shape: {}'.format(rcnn_reg.shape))

	# Get net result
	softmax_classes = tf.nn.softmax(rcnn_cls, axis = 1)
	classes = tf.argmax(softmax_classes, axis = 1)
	probabilities = tf.reduce_max(softmax_classes, axis = 1)
	# Get object
	obj_ind = tf.where(classes > 0)
	classes = tf.gather(classes, obj_ind)
	classes = tf.reshape(classes, [-1])
	probabilities = tf.gather(probabilities, obj_ind)
	probabilities = tf.reshape(probabilities, [-1])
	# Get offset
	offsets = tf.gather(rcnn_reg, obj_ind)
	offsets = tf.reshape(offsets, [-1, 4])
	cleaned_boxes_proposal = tf.gather(boxes_proposal[:, 1:], obj_ind)

	cleaned_boxes_proposal = tf.reshape(cleaned_boxes_proposal, [-1, 4])

	predict_boxes = get_real_result(cleaned_boxes_proposal, offsets)

	return rcnn_cls, rcnn_reg, classes, probabilities, predict_boxes


def rpn_generate_batch_label(gt_boxes, dim_reduce, imageSize, aspect_ratios, scales, batch_size, visualize = False, minibatch_size = 256, repeat_positives = True, \
	positives_threshold = 0.7, negatives_threshold = 0.3):

	# Prepare
	feature_map_width = int(imageSize[0]/dim_reduce)
	feature_map_height = int(imageSize[1]/dim_reduce)
	anchors = rpn_generate_anchor_boxes((feature_map_width, feature_map_height), imageSize, aspect_ratios, scales)
	#print('Anchors shape {}'.format(anchors.shape))
	anchors_cleaned = remove_cross_boundaries_anchors(anchors, imageSize[0], imageSize[1])

	# Anchors're in center, width, height format
	# Convert anchors from center width, height -> topleft bottom right
	A = format_anchors(anchors)
	n_anchors = len(A)
	batch_anchors = np.reshape(anchors, (-1, 4))
	batch_anchors = np.tile(batch_anchors, (batch_size, 1))

	#print('Anchor shape: ' + str(batch_anchors.shape))
	#print('GT: ' + str(gt_boxes))
	#print('Batch anchors: {} {}'.format(batch_anchors[6163], batch_anchors[6463]) )
	# Calculate iou between each groundtruth box & anchor box
	gt_each_img = np.zeros((batch_size), dtype = np.int32)
	for i in range(batch_size):
		gt_each_img[i] = len(gt_boxes[i])
	#print('Number of groundtruth: ')
	#print(gt_each_img)
	gt_boxes_flatten = [item for sublist in gt_boxes for item in sublist]
	gt_boxes_flatten = np.reshape(gt_boxes_flatten, (-1, 4))
	# Convert Groundtruth box from c, w, h -> tl br
	gt_boxes_flatten_formated = format_anchors(gt_boxes_flatten)

	#print('Groundtruth box: ')
	#print(gt_boxes_flatten)

	total_gt = len(gt_boxes_flatten_formated)
	IoU = iou_tensorial(gt_boxes_flatten_formated, A)
	idx_of_GT = np.argmax(IoU, axis = 0)
	A_scores_list = np.max(IoU, axis = 0)

	# Get safety object
	safety_object = np.argmax(IoU, axis = 1)
	A_scores_list[A_scores_list>positives_threshold] = 1
	A_scores_list[A_scores_list<negatives_threshold] = 0
	A_scores_list[np.logical_and(A_scores_list>0, A_scores_list<1)] = 0.5
	#print(A_scores_list)
	A_scores_list_cleaned = A_scores_list[anchors_cleaned]

	object_indices = anchors_cleaned[A_scores_list_cleaned > 0.9]
	# print('Object indices: {}'.format(object_indices))
	non_object_indices = anchors_cleaned[A_scores_list_cleaned < 0.1]
	# print('non_object indices: {}'.format(non_object_indices))
	# include at least one object
	if (len(object_indices) == 0):
		object_indices = np.array(safety_object)
		#print('Safety obj: {}'.format(safety_object))
		A_scores_list[safety_object] = 1
		idx = np.argwhere(non_object_indices == safety_object)
		non_object_indices = np.delete(non_object_indices, idx)

	assert(len(object_indices) > 0)

	if repeat_positives:
		chosen_idx_obj = np.random.choice(object_indices, int(minibatch_size/2),replace = True)
		assert(len(chosen_idx_obj) + len(non_object_indices) >= minibatch_size)
	else:
		assert(len(object_indices) + len(non_object_indices) >= minibatch_size)
		chosen_idx_obj = np.random.choice(object_indices, min(len(object_indices), int(minibatch_size/2)),replace = False)
	chosen_idx_non_obj = np.random.choice(non_object_indices, minibatch_size - len(chosen_idx_obj), replace = False)
	#print('Chosen Object indices: {}'.format(chosen_idx_obj))
	#print('Chosen NonObject indices: {}'.format(chosen_idx_non_obj))
	minibatch_indices = np.concatenate((chosen_idx_obj, chosen_idx_non_obj))

	# minibatch only
	ground_truth_anchors_list = np.zeros((len(minibatch_indices), 4))
	ground_truth_anchors_list[:,0] = (gt_boxes_flatten[idx_of_GT[minibatch_indices],0] - batch_anchors[minibatch_indices,0]) / batch_anchors[minibatch_indices,2]
	ground_truth_anchors_list[:,1] = (gt_boxes_flatten[idx_of_GT[minibatch_indices],1] - batch_anchors[minibatch_indices,1]) / batch_anchors[minibatch_indices,3]
	ground_truth_anchors_list[:,2] = np.log(gt_boxes_flatten[idx_of_GT[minibatch_indices],2]/batch_anchors[minibatch_indices,2])
	ground_truth_anchors_list[:,3] = np.log(gt_boxes_flatten[idx_of_GT[minibatch_indices],3]/batch_anchors[minibatch_indices,3])
	#print(ground_truth_anchors_list)
	a = A_scores_list[minibatch_indices]
	A_scores_one_hot = np.zeros((a.shape[0], 2))
	A_scores_one_hot[np.arange(a.shape[0]), a.astype(int)] = 1
	#print(A_scores_one_hot)

	return minibatch_indices, A_scores_one_hot, ground_truth_anchors_list

def fastRCNN_generate_batch_labels(boxes_proposal, groundtruth, labels, cls_thresh = 0.5, number_samples = 128, positive_percent = 0.25, n_classes = 10):
	# boxes_proposal - array [?, 4] in topleft bottomright format
	# groundtruth - array [?, 4] in center, width, height format
	# labels - array [?, n]	- one hot embedding
	# n = number of classes

	# Compute IOU
	fgroundtruth = format_anchors(groundtruth)
	iou = iou_tensorial(fgroundtruth, boxes_proposal)
	idx_of_GT = np.argmax(iou, axis = 0)
	bp_labels = labels[idx_of_GT]
	probabilities = np.max(iou, axis = 0)

	# Generate positive & negative
	positive = np.arange(len(idx_of_GT))[probabilities > cls_thresh]
	negative = np.arange(len(idx_of_GT))[probabilities <= cls_thresh]

	#assert len(negative) > 0 , 'Not found negative box proposal !'
	if len(negative) == 0 or len(positive) == 0:
		return [], 0, 0, 0

	# Get idx
	choosen_positive = np.random.choice(positive, int(number_samples*positive_percent), replace = True)
	choosen_negative = np.random.choice(negative, number_samples - int(number_samples*positive_percent), replace = True)
	choosen_idx = np.concatenate((choosen_positive, choosen_negative))

	# Get classes label
	positive_labels = list(bp_labels[choosen_positive])
	sample_labels = positive_labels + list((number_samples - int(number_samples*positive_percent))*[0])
	sample_labels_onehot = np.zeros((len(sample_labels), n_classes+1), dtype = np.uint8)
	sample_labels_onehot[np.arange(len(sample_labels)), sample_labels] = 1
	sample_labels_onehot = np.array(sample_labels_onehot, dtype = np.float32)

	# Get regression labels - called 'offsets'
	fboxes_proposal = format_anchors_inv(boxes_proposal)
	offsets = np.zeros((len(choosen_positive), 4))
	offsets[:, 0] = (fboxes_proposal[choosen_positive, 0] - groundtruth[idx_of_GT[choosen_positive], 0]) / fboxes_proposal[choosen_positive, 2]
	offsets[:, 1] = (fboxes_proposal[choosen_positive, 1] - groundtruth[idx_of_GT[choosen_positive], 1]) / fboxes_proposal[choosen_positive, 3]
	offsets[:, 2] = np.log(fboxes_proposal[choosen_positive, 2] / groundtruth[idx_of_GT[choosen_positive], 2])
	offsets[:, 3] = np.log(fboxes_proposal[choosen_positive, 3] / groundtruth[idx_of_GT[choosen_positive], 3])
	
	return choosen_positive, choosen_idx, sample_labels_onehot, offsets


def rpn_calculate_loss(confidence_list, regression_list, minibatchIndices, confidence_gt, regression_gt, name=None):

	with tf.name_scope(name, 'RPN_Loss', [confidence_list, regression_list, minibatchIndices, confidence_gt, regression_gt]):
		minibatchIndices = tf.convert_to_tensor(minibatchIndices)
		confidence_gt = tf.convert_to_tensor(confidence_gt)
		regression_gt = tf.convert_to_tensor(regression_gt)

		# Confidence loss: Cross Entropy
		cls_minibatch = tf.gather(confidence_list, minibatchIndices)
		confidence_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = confidence_gt, logits = cls_minibatch), name = 'confidence_loss')
		#confidence_loss = tf.divide(confidence_loss, n_classes)

		# Regression loss: Smooth L1
		reg_minibatch = tf.gather(regression_list, minibatchIndices)
		abs_deltaT = tf.abs(reg_minibatch - regression_gt)
		L1smooth = tf.where(tf.less(abs_deltaT, 1.0), 0.5 * abs_deltaT * abs_deltaT, abs_deltaT - 0.5)
		reg_loss = tf.reduce_mean(confidence_gt[:,1] * tf.reduce_sum(L1smooth, axis=-1))
		reg_loss = 10*reg_loss

		# Total Loss
		total_loss = tf.add(confidence_loss, reg_loss)

	return total_loss, confidence_loss, reg_loss

def fastRCNN_calculate_loss(rcnn_cls, rcnn_reg, positive_idx, choosen_idx, gt_cls_onehot, gt_reg, name = None, lamda = 10):
	with tf.name_scope(name, 'FastRCNN_Loss', [rcnn_cls, rcnn_reg, positive_idx, choosen_idx, gt_cls_onehot, gt_reg]):
		# Convert
		positive_idx = tf.convert_to_tensor(positive_idx)
		choosen_idx = tf.convert_to_tensor(choosen_idx)
		gt_cls_onehot = tf.convert_to_tensor(gt_cls_onehot)
		gt_reg = tf.convert_to_tensor(gt_reg)

		# Calculate confidence loss
		cls_minibatch = tf.gather(rcnn_cls, choosen_idx)
		cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = gt_cls_onehot, logits = cls_minibatch), name = 'RCNN_loss')

		# Calculate regression loss
		reg_minibatch = tf.gather(rcnn_reg, positive_idx)
	
		abs_deltaT = tf.abs(reg_minibatch - gt_reg)
		L1smooth = tf.where(tf.less(abs_deltaT, 1.0), 0.5 * abs_deltaT * abs_deltaT, abs_deltaT - 0.5)
		sum_L1smooth = tf.reduce_sum(L1smooth, axis=1)
		reg_loss = tf.reduce_mean(sum_L1smooth)
		reg_loss = lamda*reg_loss

		total_loss = tf.add(cls_loss, reg_loss)

	return total_loss, cls_loss, reg_loss

def get_region_of_interest(ft_shape, boxes_proposal, dim_reduce = 16):
	# Normalize boxes_proposal
	boxes_proposal = boxes_proposal/dim_reduce
	

	# Convert from center, width, height to topleft bottom right
	boxes_proposal_x1 = boxes_proposal[:, 0]
	boxes_proposal_x1 = tf.reshape(boxes_proposal_x1, (-1, 1))
	boxes_proposal_x1 = tf.clip_by_value(boxes_proposal_x1, 0.0, boxes_proposal_x1)
	
	boxes_proposal_y1 = boxes_proposal[:, 1]
	boxes_proposal_y1 = tf.reshape(boxes_proposal_y1, (-1, 1))
	boxes_proposal_y1 = tf.clip_by_value(boxes_proposal_y1, 0.0, boxes_proposal_y1)

	boxes_proposal_x2 = boxes_proposal[:, 2]
	boxes_proposal_x2 = tf.reshape(boxes_proposal_x2, (-1, 1))
	boxes_proposal_x2 = tf.clip_by_value(boxes_proposal_x2, boxes_proposal_x1, ft_shape[2])

	boxes_proposal_y2 = boxes_proposal[:, 3]
	boxes_proposal_y2 = tf.reshape(boxes_proposal_y2, (-1, 1))
	boxes_proposal_y2 = tf.clip_by_value(boxes_proposal_y2, boxes_proposal_y1, ft_shape[1])
	
	# Clear small boxes (width < 48px, height < 48px)
	wx = tf.subtract(boxes_proposal_x2, boxes_proposal_x1)
	wx = tf.reshape(wx, [-1])
	wy = tf.subtract(boxes_proposal_y2, boxes_proposal_y1)
	wy = tf.reshape(wy, [-1])
	cleaned = tf.where(tf.logical_and(tf.greater_equal(wx, 3), tf.greater_equal(wy, 3)))
	cleaned = tf.reshape(cleaned, [-1])

	# Format to int
	boxes_proposal_cell = tf.cast(tf.concat([boxes_proposal_x1, boxes_proposal_y1, boxes_proposal_x2, boxes_proposal_y2], axis = -1), tf.int32)
	boxes_proposal_cell = tf.gather(boxes_proposal_cell, cleaned)

	return boxes_proposal_cell

class RPN:
	def __init__(self):
		self.BATCH_SIZE = 1
		self.rpn_lr = 0.00001
		self.rcnn_lr = 0.00001
		self.max_size = 1024
		self.min_size = 608
		self.aspect_ratios = [1, 2, 4]
		self.scales = [32, 64, 128]		# In pixel
		self.k_anchors = len(self.aspect_ratios)*len(self.scales)
		self.n_classes = 10
		self.max_epoches = 10
		self.epoches_to_save = 5
		self.global_step = 0
		self.valid = 0.1
		self.check_point = 'vgg_16.ckpt'
		self.eval_iou = [0.5, 0.75]
		self.eval_cls = [0.4, 0.6]
		self.epoch_to_validate = 5
		self.rpn_checkpoint = 'graphs/model.ckpt'
		self.rpn_trained = 'model.ckpt'

	def summary(self):
		with tf.name_scope('summaries'):
			tf.summary.scalar('confidence_loss', self.confidence_loss)
			tf.summary.scalar('regression_loss', self.regression_loss)
			tf.summary.scalar('total_loss', self.total_loss)
			self.summary_op = tf.summary.merge_all()

	def build_net(self):
		
		# Forward
		self.image_batch_placeholder = tf.placeholder(shape = [None, None, None, 3],dtype='float32')
		self.placeholder_anchors = tf.placeholder(shape = [None, 4], dtype='float32', name = 'PL_anchors')
		self.placeholder_anchors_cleaned = tf.placeholder(shape = [None],dtype='int32', name = 'PL_anchors_cleaned')

		# Inference
		self.feature_map = _base_inference(self.image_batch_placeholder)
		self.ft_shape = tf.cast(tf.shape(self.feature_map), tf.float32)
		self.confidence_list, self.regression_list, self.pp_probabilities = _inference(self.feature_map, self.k_anchors)
		self.prediction, self.probabilities = _predict_boxes(self.pp_probabilities, self.regression_list, self.ft_shape, self.placeholder_anchors, self.placeholder_anchors_cleaned)
		# self.boxes_proposal_cell = get_region_of_interest(self.ft_shape, self.prediction)
		# self.boxes_proposal_cell = tf.convert_to_tensor([self.boxes_proposal_cell])
		self.rcnn_cls, self.rcnn_reg, self.rcnn_classes, self.rcnn_probabilities, self.rcnn_predict_boxes = _fastRCNN(self.feature_map, self.prediction)

		# Calculate loss
		self.minibatch_indices = tf.placeholder(dtype='int32', name = 'PL_RPN_indices')
		self.A_scores_one_hot = tf.placeholder(dtype='float32', name = 'PL_RPN_onehot_label')
		self.gt_anchors_list = tf.placeholder(dtype='float32', name = 'PL_RPN_gt')
		self.total_loss, self.confidence_loss, self.regression_loss = rpn_calculate_loss(self.confidence_list, self.regression_list, \
			self.minibatch_indices, \
			self.A_scores_one_hot, self.gt_anchors_list)
		self.saver = tf.train.Saver()

		# Calculate fastRCNN loss
		self.placeholder_choosen_idx = tf.placeholder(shape = [None], dtype='int32', name = 'PL_indices')
		self.placeholder_positive_idx = tf.placeholder(shape = [None], dtype='int32', name = 'PL_positive_indices')
		self.placeholder_gt_reg = tf.placeholder(shape = [None, 4], dtype='float32', name = 'PL_gt_reg')
		self.placeholder_gt_cls = tf.placeholder(shape = [None, self.n_classes+1], dtype='float32', name = 'PL_gt_cls')
		self.rcnn_total_loss, self.rcnn_cls_loss, self.rcnn_reg_loss = fastRCNN_calculate_loss(self.rcnn_cls, self.rcnn_reg, self.placeholder_positive_idx, \
			self.placeholder_choosen_idx, self.placeholder_gt_cls, self.placeholder_gt_reg)

		# Accuracy
		self.summary()

		# Optimizer
		#self.RPN_trainer = tf.train.MomentumOptimizer(learning_rate = self.lr, momentum = 0.9).minimize(self.total_loss, name = 'RPN_trainer')
		self.session = tf.Session()

	def get_data(self, visualize = True):
		# data_imgs, data_labels, data_coordinates, self.train_sizes = get_data('logo/images', 'logo/xml', self.min_size, self.max_size, visualize = False)
		# self.train_imgs, self.train_labels, self.train_coordinates, self.val_imgs, self.val_labels, self.val_coordinates = split_dataset(data_imgs, data_labels, data_coordinates, self.valid)
		# print('Train: {}	Val: {}'.format(len(self.train_imgs), len(self.val_imgs)))
		self.train_imgs, self.train_labels, self.val_imgs, self.val_labels = get_list_data('logo/images', 'logo/xml', val_percent = 0.2)

	def shuffle_data(self):
		ind = [i for i in range(len(self.train_labels))]
		ind = shuffle(ind)
		self.train_imgs, self.train_labels, self.train_coordinates = shuffle(self.train_imgs, self.train_labels, self.train_coordinates)

	def rpn_train(self):
		print('Max epoches: ' + str(self.max_epoches))
		print('Learning rate: ' + str(self.rpn_lr))
		self.RPN_trainer = tf.train.AdamOptimizer(learning_rate = self.rpn_lr).minimize(self.total_loss, name = 'RPN_trainer')
		max_steps = int(len(self.train_labels) / self.BATCH_SIZE)
		writer = tf.summary.FileWriter('graphs', self.session.graph)
		self.session.run([tf.global_variables_initializer()])

		# Restore parameter
		print('Fineturning with VGG16-ImageNet ...')
		restorer = fine_turning_vgg16()
		restorer.restore(self.session, self.check_point)

		for epoch in range(self.max_epoches):

			# Shuffle data
			#self.shuffle_data()

			print('Epoches {} / {} :'.format(epoch, self.max_epoches))
			sumary_total_loss = []
			sumary_confidence_loss = []
			sumary_regression_loss = []
			for step in range(max_steps):
				minibatch_imgs, minibatch_labels, minibatch_gt_coors = read_one_data(self.train_imgs[step], self.train_labels[step])
				
				height, width,_ = minibatch_imgs.shape
				#print('Image shape: {}x{}'.format(width, height))
				minibatch_imgs = np.array([minibatch_imgs])
				minibatch_indices, A_scores_one_hot, ground_truth_anchors_list = rpn_generate_batch_label(minibatch_gt_coors, 16, (width, height), self.aspect_ratios, self.scales, self.BATCH_SIZE)
				_, total_loss, confidence_loss, regression_loss, conf_list, summaries = self.session.run([self.RPN_trainer, self.total_loss, self.confidence_loss, self.regression_loss, self.confidence_list, self.summary_op], \
						feed_dict = { self.minibatch_indices: minibatch_indices, \
								self.A_scores_one_hot: A_scores_one_hot, \
								self.gt_anchors_list: ground_truth_anchors_list, \
								self.image_batch_placeholder: minibatch_imgs})
				#print('Confidence list shape {}'.format(conf_list.shape))
				writer.add_summary(summaries, global_step = epoch*max_steps + step)
				sumary_total_loss.append(total_loss)
				sumary_confidence_loss.append(confidence_loss)
				sumary_regression_loss.append(regression_loss)
				print('Step: {}/{} | Conf_loss: {} | Reg_loss: {} | Total_loss: {}'.format(step, max_steps, confidence_loss, regression_loss, total_loss))
				with open('logs.txt', 'a') as f:
					f.write('Step: {}/{} | Conf_loss: {} | Reg_loss: {} | Total_loss: {}\n'.format(step, max_steps, confidence_loss, regression_loss, total_loss))
			mean_total_loss = np.mean(np.array(sumary_total_loss))
			mean_confidence_loss = np.mean(np.array(sumary_confidence_loss))
			mean_regression_loss = np.mean(np.array(sumary_regression_loss))
			print('-----------------------------------------------------------------------------------------------------------------------------')
			print('Epoch: {}      Conf_loss: {}      Reg_loss: {}		Total_loss:  {}'.format(epoch, mean_confidence_loss, mean_regression_loss, mean_total_loss))
			print('-----------------------------------------------------------------------------------------------------------------------------')
			with open('logs.txt', 'a') as f:
					f.write('Epoch: {}      Conf_loss: {}      Reg_loss: {}		Total_loss:  {}\n'.format(epoch, mean_confidence_loss, mean_regression_loss, mean_total_loss))
		
		self.saver.save(self.session, 'graph/final_model.ckpt')
		writer.close()

	def fastRCNN_train(self, dim_reduce = 16):
		max_epoches = 20
		self.FastRCNN_trainer = tf.train.AdamOptimizer(learning_rate = self.rcnn_lr).minimize(self.rcnn_total_loss, name = 'FastRCNN_trainer')
		max_steps = int(len(self.train_labels) / self.BATCH_SIZE)
		writer = tf.summary.FileWriter('FastRCNN_graphs', self.session.graph)
		print('Fast R-CNN training ...')
		self.session.run([tf.global_variables_initializer()])

		print('Restore RPN parameters ...')
		# Restore parameter
		all_vars = slim.get_model_variables()
		vars_to_restore = {v.op.name:v for v in all_vars if not v.name.startswith('rcnn')}
		print(vars_to_restore)
		restorer = tf.train.Saver(vars_to_restore)
		restorer.restore(self.session, 'graph/final_model.ckpt')

		print('Total step: {}'.format(max_steps))
		for epoch in range(max_epoches):
			#self.shuffle_data()
			print('Epoches {} / {} :'.format(epoch, max_epoches))
			sumary_total_loss = []
			sumary_confidence_loss = []
			sumary_regression_loss = []
			for step in range(max_steps):
				minibatch_imgs, minibatch_labels, minibatch_gt_coors = read_one_data(self.train_imgs[step], self.train_labels[step])
				
				height, width,_ = minibatch_imgs.shape
				minibatch_imgs = np.array([minibatch_imgs])
				minibatch_labels = np.array(minibatch_labels, dtype = np.uint8)
				minibatch_gt_coors = np.array(minibatch_gt_coors)

				# Get proposal boxes
				anchors = rpn_generate_anchor_boxes((int(width/dim_reduce), int(height/dim_reduce)), (width, height), self.aspect_ratios, self.scales)
				anchors = anchors.reshape((-1, 4))
				anchors_cleaned = remove_cross_boundaries_anchors(anchors, width, height)
				boxes_proposal = self.session.run(self.prediction, \
						feed_dict = {self.image_batch_placeholder : minibatch_imgs, \
						self.placeholder_anchors : anchors,\
						self.placeholder_anchors_cleaned: anchors_cleaned})
				boxes_proposal = boxes_proposal[:, 1:]
				# Training
				if len(boxes_proposal) == 0:
					continue
				choosen_positive, choosen_idx, sample_labels_onehot, offsets = fastRCNN_generate_batch_labels(boxes_proposal, minibatch_gt_coors, minibatch_labels)
				if len(choosen_positive) == 0:
					continue
				_, total_loss, confidence_loss, regression_loss = self.session.run([self.FastRCNN_trainer, self.rcnn_total_loss, self.rcnn_cls_loss, self.rcnn_reg_loss], \
					feed_dict = {self.image_batch_placeholder : minibatch_imgs, \
					self.placeholder_positive_idx: choosen_positive, \
					self.placeholder_choosen_idx: choosen_idx, \
					self.placeholder_gt_cls: sample_labels_onehot, \
					self.placeholder_gt_reg: offsets, \
					self.placeholder_anchors : anchors,\
					self.placeholder_anchors_cleaned: anchors_cleaned })
				#print('Confidence list shape {}'.format(conf_list.shape))
				#writer.add_summary(summaries, global_step = epoch*max_steps + step)
				
				#print(pooled_features)
				sumary_total_loss.append(total_loss)
				sumary_confidence_loss.append(confidence_loss)
				sumary_regression_loss.append(regression_loss)
				
				print('Step: {}/{} | Conf_loss: {} | Reg_loss: {} | Total_loss: {}'.format(step, max_steps, confidence_loss, regression_loss, total_loss))
			mean_total_loss = np.mean(np.array(sumary_total_loss))
			mean_confidence_loss = np.mean(np.array(sumary_confidence_loss))
			mean_regression_loss = np.mean(np.array(sumary_regression_loss))
			print('-----------------------------------------------------------------------------------------------------------------------------')
			print('Epoch: {}      Conf_loss: {}      Reg_loss: {}		Total_loss:  {}'.format(epoch, mean_confidence_loss, mean_regression_loss, mean_total_loss))
			print('-----------------------------------------------------------------------------------------------------------------------------')


		self.saver.save(self.session, 'fineturning/final_model.ckpt')


	def eval_FasterRCNN(self, dim_reduce = 16):
		print('Validating ...')
		total_val_img = len(self.val_imgs)
		list_precision = []
		list_recall = []
		self.session.run([tf.global_variables_initializer()])
		print('Loading pretrained model ...')
		self.saver.restore(self.session, 'fineturning/final_model.ckpt')
		pres_x = 0
		pres_y = 0
		res_x = 0
		res_y = 0
		cs = 0.4
		iou = 0.7
		for i in range(total_val_img):
			print('Current: ' + self.val_imgs[i])
			precision, recall = [], []
			minibatch_imgs, minibatch_labels, minibatch_gt_coors = read_one_data(self.val_imgs[i], self.val_labels[i])

			height, width,_ = minibatch_imgs.shape
			minibatch_imgs = np.array([minibatch_imgs])
			minibatch_labels = np.array(minibatch_labels, dtype = np.uint8)
			minibatch_gt_coors = np.array(minibatch_gt_coors)
			
			anchors = rpn_generate_anchor_boxes((int(width/dim_reduce), int(height/dim_reduce)), (width, height), self.aspect_ratios, self.scales)
			anchors = anchors.reshape((-1, 4))
			anchors_cleaned = remove_cross_boundaries_anchors(anchors, width, height)

			clss, probabilities, boxes = self.session.run([self.rcnn_classes, self.rcnn_probabilities, self.rcnn_predict_boxes], \
				feed_dict = {self.image_batch_placeholder:minibatch_imgs, \
							self.placeholder_anchors: anchors,\
							self.placeholder_anchors_cleaned: anchors_cleaned})
			try:
				boxes, probabilities, clss = nms(boxes, probabilities, clss)
			except ValueError:
				res_y += len(minibatch_gt_coors)
				continue
			#print(clss)
			#print(minibatch_labels)
			
			pre_x, pre_y, re_x, re_y = calculate_eval(probabilities, clss, boxes, minibatch_labels, minibatch_gt_coors, cs, iou)
			pres_x += pre_x
			pres_y += pre_y
			res_x += re_x
			res_y += re_y
		print('Precision: ' + str(pres_x/pres_y))
		print('Recall: ' + str(recall))

	def eval_logo_trademark(self, dim_reduce = 16, cls_thresh = 0.4):
		print('Trademark Logo Validating ...')
		df_name = []
		tmp_df = []
		total_val_img = len(self.val_imgs)
		self.session.run([tf.global_variables_initializer()])
		print('Loading pretrained model ...')
		self.saver.restore(self.session, 'fineturning/final_model.ckpt')
		times = []
		for i in range(total_val_img):
			bk_time = time.time()
			print('Current: ' + self.val_imgs[i])
			df_name.append(self.val_imgs[i])
			copyfile(self.val_imgs[i], 'val_' + self.val_imgs[i])
			# Get prediction
			minibatch_imgs, minibatch_labels, minibatch_gt_coors = read_one_data(self.val_imgs[i], self.val_labels[i])

			height, width,_ = minibatch_imgs.shape
			minibatch_imgs = np.array([minibatch_imgs])
			minibatch_labels = np.array(minibatch_labels, dtype = np.uint8)
			minibatch_gt_coors = np.array(minibatch_gt_coors)
			
			anchors = rpn_generate_anchor_boxes((int(width/dim_reduce), int(height/dim_reduce)), (width, height), self.aspect_ratios, self.scales)
			anchors = anchors.reshape((-1, 4))
			anchors_cleaned = remove_cross_boundaries_anchors(anchors, width, height)

			clss, probabilities, boxes = self.session.run([self.rcnn_classes, self.rcnn_probabilities, self.rcnn_predict_boxes], \
				feed_dict = {self.image_batch_placeholder:minibatch_imgs, \
							self.placeholder_anchors: anchors,\
							self.placeholder_anchors_cleaned: anchors_cleaned})
			# Remove by threshold
			try:
				boxes, probabilities, clss = nms(boxes, probabilities, clss, overlapThresh = 0.7)
			except ValueError:
				clss = []
			result = get_trademark(clss)
			tmp_df.append(result)
			times.append(time.time()-bk_time)

		times = np.array(times)
		print('Average time: {}'.format(np.mean(times)))
		tmp_df = np.array(tmp_df, dtype = np.uint8)
		df_vietin = tmp_df[:, 0]
		df_vietcom = tmp_df[:, 1]
		df_bidv = tmp_df[:, 2]
		df_agri = tmp_df[:, 3]
		df_vp = tmp_df[:, 4]
		print('Create Data Frame ...')
		dictionary = {'Image':df_name, 'Vietinbank':df_vietin, 'Vietcombank':df_vietcom, 'BIDV':df_bidv, 'Agribank':df_agri, 'VPBank':df_vp}
		df = pd.DataFrame(dictionary)
		export_csv = df.to_csv('export_dataframe.csv', header=True) 

	def predict_logo_trademark(self, list_img, dim_reduce = 16, cls_thresh = 0.4):
		# Return 3 list include logo
		print('Trademark Logo Predictor ')
		vietin = []
		vietcom = []
		bidv = []
		for i in range(len(list_img)):
			print('Current {}/{}: '.format(i+1, len(list_img)) + list_img[i])
			img = read_and_rescale(list_img[i])
			height, width, _ = img.shape
			image = np.array([img])
			anchors = rpn_generate_anchor_boxes((int(width/dim_reduce), int(height/dim_reduce)), (width, height), self.aspect_ratios, self.scales)
			anchors = anchors.reshape((-1, 4))
			anchors_cleaned = remove_cross_boundaries_anchors(anchors, width, height)
			clss, probabilities, boxes = self.session.run([self.rcnn_classes, self.rcnn_probabilities, self.rcnn_predict_boxes], \
					feed_dict = {self.image_batch_placeholder:image, \
								self.placeholder_anchors: anchors,\
								self.placeholder_anchors_cleaned: anchors_cleaned})
			try:
				boxes, probabilities, clss = nms(boxes, probabilities, clss, overlapThresh = 0.7)
			except ValueError:
				clss = []
			result = get_trademark(clss)
			if result[0] == 1:
				vietin.append(list_img[i])
			if result[1] == 1:
				vietcom.append(list_img[i])
			if result[2] == 1:
				bidv.append(list_img[i])
		print('----------------')
		print('Vietinbank: {}'.format(len(vietin)))
		print('VietcomBank: {}'.format(len(vietcom)))
		print('BIDV: {}'.format(len(bidv)))
		print('Complete !')
		return vietin, vietcom, bidv


	def rpn_predict(self, image_path, cls_thresh = 0.7, dim_reduce = 16):
		img = read_and_rescale(image_path)
		height, width, _ = img.shape
		print('Resized: {}x{}'.format(width, height))
		self.session.run([tf.global_variables_initializer()])
		print('Loading pretrained model ...')
		#saver_temp = tf.train.Saver(v.op.name for v in slim.get_model_variables() if not v.startswith('rcnn'))
		self.saver.restore(self.session, 'fineturning/final_model.ckpt')
		image = np.array([img])
		print('Predict: ')

		# Get anchors
		anchors = rpn_generate_anchor_boxes((int(width/dim_reduce), int(height/dim_reduce)), (width, height), self.aspect_ratios, self.scales)
		anchors = anchors.reshape((-1, 4))
		anchors_cleaned = remove_cross_boundaries_anchors(anchors, width, height)
		predict_boxes, probabilities, pp_cell = self.session.run([self.prediction, self.probabilities, self.rcnn_reg], \
				feed_dict = {self.image_batch_placeholder:image, \
							self.placeholder_anchors: anchors,\
							self.placeholder_anchors_cleaned: anchors_cleaned})

		#print(pp_cell)
		img_base = read_and_rescale_test(image_path)
		for i in range(len(predict_boxes)):
			_, x1, y1, x2, y2 = predict_boxes[i]
			if x1>0 and y1>0 and x2>0 and y2>0:
				img_base = cv2.rectangle(img_base, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
		cv2.imshow('predict', img_base)
		img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
		cv2.imwrite('rpn_test.jpg', img_base)
		cv2.waitKey(0)

	def rcnn_predict(self, image, class_id, cls_thresh = 0.7, dim_reduce = 16):
		img_org = read_and_rescale_notpre(image)
		img = read_and_rescale(image)
		height, width, _ = img.shape
		print('Resized: {}x{}'.format(width, height))
		self.session.run([tf.global_variables_initializer()])
		print('Loading pretrained model ...')
		self.saver.restore(self.session, 'fineturning/final_model.ckpt')
		image = np.array([img])
		print('Predict: ')
		anchors = rpn_generate_anchor_boxes((int(width/dim_reduce), int(height/dim_reduce)), (width, height), self.aspect_ratios, self.scales)
		anchors = anchors.reshape((-1, 4))
		anchors_cleaned = remove_cross_boundaries_anchors(anchors, width, height)
		clss, probabilities, boxes = self.session.run([self.rcnn_classes, self.rcnn_probabilities, self.rcnn_predict_boxes], \
				feed_dict = {self.image_batch_placeholder:image, \
							self.placeholder_anchors: anchors,\
							self.placeholder_anchors_cleaned: anchors_cleaned})
		#print(probabilities.shape)
		#print(clss.shape)
		#print(boxes.shape)


		img_org = draw_boxes(img_org, boxes, probabilities, clss, classes_id)
		img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
		cv2.imshow('Result', img_org)
		cv2.imwrite('rcnn_test.jpg', img_org)
		cv2.waitKey(0)

if __name__ == '__main__':
	net = RPN()
	net.get_data(visualize = False)
	net.build_net()
	#net.summary()
	#net.fastRCNN_train()
	#net.rpn_predict('55_agribank.jpg')
	classes_id = {
		'inbankText': 1,\
		'inbankLogo': 2,\
		'combankText': 3,\
		'combankLogo': 4,\
		'vpText': 5,\
		'vpLogo': 6,\
		'AgriText': 7,\
		'AgriLogo': 8,\
		'agriText': 7,\
		'agriLogo': 8,\
		'bidvText': 9,\
		'bidvLogo': 10 }
	net.rcnn_predict('vp_144.jpg', classes_id)
	#net.eval_logo_trademark()
