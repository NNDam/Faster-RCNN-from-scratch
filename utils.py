import xml.etree.ElementTree as ET
import numpy as np
import glob
import cv2
import random
import os
from PIL import Image
#from random import shuffle
from sklearn.utils import shuffle
import math

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
	'bidvLogo': 10
}

def get_trademark(clss):
	vietin = 0
	vietcom = 0
	bidv = 0
	for i in clss:
		if i == 1 or i == 2:
			vietin = 1
		if i == 3 or i == 4:
			vietcom = 1
		if i == 9 or i == 10:
			bidv = 1
	return [vietin, vietcom, bidv]
def calculate_eval(predict_proba, predict_cls, predict_reg, gt_boxes_cls, gt_boxes, cls_thresh, iou_thresh):

	total_reg_boxes = len(predict_reg)

	if total_reg_boxes == 0:
		return 0, 0, 0, len(gt_boxes_cls)

	# Remove box by cls
	choosed = np.argwhere(predict_proba >= cls_thresh )
	predict_proba = np.take(predict_proba, choosed)
	predict_reg = np.take(predict_reg, choosed, axis = 0)
	predict_cls = np.take(predict_cls, choosed)

	total_reg_boxes = len(predict_reg)

	if total_reg_boxes == 0:
		return 0, 0, 0, len(gt_boxes_cls)

	reg_respone = format_anchors(predict_reg)   # Convert to tl-br

	gt_boxes = np.array(gt_boxes)
	gt_boxes_formated = format_anchors(gt_boxes)

	iou_matrix = iou_tensorial(gt_boxes_formated, reg_respone)
	#print(iou_matrix)
	
	# Calculate Precision
	max_iou = np.max(iou_matrix, axis = 0)
	argmax_iou = np.argmax(iou_matrix, axis = 0)
	choosed = np.argwhere(max_iou >= iou_thresh)
	
	choosed_predict_boxes = np.take(predict_cls, choosed).flatten()
	choosed_predict_labels = gt_boxes_cls[np.take(argmax_iou, choosed)].flatten()

	tmp = choosed_predict_boxes - choosed_predict_labels
	right = tmp[tmp == 0]
	precision_x = len(right)
	precision_y = len(max_iou)
	
	# Calculate Recall
	max_iou = np.max(iou_matrix, axis = 1)
	argmax_iou = np.argmax(iou_matrix, axis = 1)
	choosed = np.argwhere(max_iou >= iou_thresh)
	
	gt_choosed = np.take(gt_boxes_cls, choosed).flatten()
	predict_choosed = predict_cls[np.take(argmax_iou, choosed)].flatten()
	tmp = gt_choosed - predict_choosed
	right = tmp[tmp == 0]
	recall_x = len(right)
	recall_y = len(max_iou)
	return precision_x, precision_y, recall_x, recall_y

def get_correct_label(idx, classes_id):
	return [k for (k, v) in classes_id.items() if v==idx][0]

def draw_boxes(img, boxes, probabilities, clss, classes_id):
	# Cleaned by NMS
	boxes, probabilities, clss = nms(boxes, probabilities, clss, overlapThresh = 0.7)
	print(clss)
	# Draw
	for i in range(len(boxes)):
		x, y, w, h = boxes[i]
		color = (255, int(50*(clss[i]%2)), int(50*(clss[i]%2)))
		text = get_correct_label(clss[i], classes_id)
		img = cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 1)
		img = cv2.putText(img, text, (int(x-w/2), int(y-h/2)-1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA )
	return img

def rpn_generate_anchor_boxes(featureMapSize, imageSize, aspect_ratios, scales, visualize = False):
	# Return collection of anchors in (center_x, center_y, width, height) format
	k = len(aspect_ratios)*len(scales)
	A = np.zeros((featureMapSize[1], featureMapSize[0], k, 4))
	anchors = []
	for s in scales:
		area = s*s
		for ar in aspect_ratios:
			h = math.sqrt(area/ar)
			w = area/h
			anchors.append((round(w), round(h)))
	for i in range(featureMapSize[1]):
		for j in range(featureMapSize[0]):
			# Coordinate of group anchors 's center
			x = (j+0.5)*imageSize[0]/featureMapSize[0]
			y = (i+0.5)*imageSize[1]/featureMapSize[1]
			for idx, a in enumerate(anchors):
				w, h = a
				ctx = np.clip(x, 0, imageSize[0])
				cty = np.clip(y, 0, imageSize[1])
				A[i, j, idx, :] = ctx, cty, w, h 
	if visualize:
		background = np.zeros((imageSize[1], imageSize[0], 3), dtype = np.uint8)
		choosed = (int(featureMapSize[0]/2), int(featureMapSize[1]/2))
		for i in range(k):
			ctx, cty, w, h = A[choosed[1], choosed[0], i, :]

			background = cv2.rectangle(background, (int(ctx - w/2), int(cty - h/2)), (int(ctx + w/2), int(cty + h/2)), (0, 0, 255), 1)
		cv2.imshow('background', background)
		cv2.waitKey(0)
	return A

def resize_with_padding(img, target_width, target_height, padding = True):
	if padding:
		im = Image.open(img)
		old_size = im.size  # old_size[0] is in (width, height) format

		ratio = min(target_width/old_size[0], target_height/old_size[1])
		new_size = tuple([int(x*ratio) for x in old_size])
		# use thumbnail() or resize() method to resize the input image

		# thumbnail is a in-place operation

		# im.thumbnail(new_size, Image.ANTIALIAS)

		im = im.resize(new_size, Image.ANTIALIAS)
		# create a new image and paste the resized on it

		new_im = Image.new("RGB", (target_width, target_height))
		padded_width = (target_width-new_size[0])//2
		padded_height = (target_height-new_size[1])//2
		new_im.paste(im, (padded_width, padded_height))
		return np.array(new_im), old_size, new_size, (padded_width, padded_height)
	else:
		img = cv2.imread(img)
		old_size = (img.shape[1], img.shape[0])
		rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return cv2.resize(rgb_img, (target_width, target_height)), old_size, (target_width, target_height), [0, 0]

def resize_image(img, min_size = 608, max_size = 1024, dim_reduce = 32):
	org_height, org_width, _ = img.shape
	ratio = org_width/org_height
	if ratio <= 1:	# Width is small than height
		if ratio <= 608/1024:
			img = cv2.resize(img, (int(ratio*1024/dim_reduce)*dim_reduce, 1024))
		else:
			img = cv2.resize(img, (608, int(608/ratio/dim_reduce)*dim_reduce))			
	else:
		if ratio >= 1024/608:
			img = cv2.resize(img, (1024, int(1024/ratio/dim_reduce)*dim_reduce))
		else:
			img = cv2.resize(img, (int(608*ratio/dim_reduce)*dim_reduce, 608))
	return img

def read_and_rescale(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = resize_image(img)
	img = prewhiten(img)
	return img

def read_and_rescale_test(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = resize_image(img)
	#img = prewhiten(img)
	return img

def read_and_rescale_notpre(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = resize_image(img)
	return img

def remove_cross_boundaries_anchors(anchors, imWidth, imHeight):
	copy_anchors = anchors.copy()
	copy_anchors = copy_anchors.reshape((-1, 4))
	copy_anchors = format_anchors(copy_anchors)	# convert to topleft-bottomright format
	cleaned_anchor_indices_1 = np.logical_and(copy_anchors[:, 2] < imWidth, copy_anchors[:, 3] < imHeight)
	cleaned_anchor_indices_2 = np.logical_and(copy_anchors[:, 0] > 0, copy_anchors[:, 1] > 0)
	cleaned_anchors = np.logical_and(cleaned_anchor_indices_1, cleaned_anchor_indices_2)
	cleaned_anchors_indices = np.arange(len(copy_anchors))[cleaned_anchors == True]
	return cleaned_anchors_indices


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def split_dataset(data_img, data_label, data_coor, val):
	# Split current dataset to train & val data
	total_data = len(data_img)
	ind = [i for i in range(total_data)]
	ind = shuffle(ind)
	data_img, data_label, data_coor = shuffle(data_img, data_label, data_coor)
	# data_img = data_img[ind , :, :, :]
	# data_label = data_label[ind, :]
	# data_coor = data_coor[ind, :, :]
	total_validate = int(total_data*val)
	#assert(total_validate > 0) 
	print('Validate size: {}'.format(total_validate))
	return data_img[:total_data - total_validate], data_label[:total_data - total_validate], data_coor[:total_data - total_validate], \
		data_img[total_data - total_validate:], data_label[total_data - total_validate:], data_coor[total_data - total_validate:]

def read_one_data(img_file, label_file):
	img_base = cv2.imread(img_file)
	old_size_height, old_size_width, _ = img_base.shape
	img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
	img = resize_image(img_base)
	new_size_height, new_size_width, _ = img.shape
	img = prewhiten(img)

	# Label
	tree = ET.parse(label_file)
	root = tree.getroot()
	label = []
	coordinate = []
	objs = root.findall('object')
	if len(objs) == 0:
		return 0
	for obj in objs:
		name = obj.find('name').text
		bndbox = obj.find('bndbox')
		xmin = float(int(bndbox.find('xmin').text)*1.0/old_size_width*new_size_width)
		ymin = float(int(bndbox.find('ymin').text)*1.0/old_size_height*new_size_height)
		xmax = float(int(bndbox.find('xmax').text)*1.0/old_size_width*new_size_width)
		ymax = float(int(bndbox.find('ymax').text)*1.0/old_size_height*new_size_height)
		label.append(int(classes_id[name]))
		coordinate.append([xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2, xmax-xmin, ymax-ymin])
	return img, label, coordinate

def get_list_data(img_folder, label_folder, val_percent = 0.2):
	img_list = []
	all_list = glob.glob(label_folder + '/*.xml')
	numb_val = int(val_percent*len(all_list))
	numb_train = len(all_list) - numb_val
	train_label_list = all_list[:numb_train]
	val_label_list = all_list[numb_train:]
	train_img_list = []
	val_img_list = []
	for i in range(len(train_label_list)):
		img_path = img_folder + '/' + train_label_list[i].split('/')[-1].split('.')[0] + '.jpg'
		if not os.path.exists(img_path):
			raise Exception(" Not found image {}".format(img_path))
		train_img_list.append(img_path)
	for i in range(len(val_label_list)):
		img_path = img_folder + '/' + val_label_list[i].split('/')[-1].split('.')[0] + '.jpg'
		if not os.path.exists(img_path):
			raise Exception(" Not found image {}".format(img_path))
		val_img_list.append(img_path)
	print('Train: {}'.format(len(train_img_list)))
	print('Val: {}'.format(len(val_img_list)))
	return train_img_list, train_label_list, val_img_list, val_label_list


def get_data(img_folder, label_folder, min_size = 608, max_size = 1024, visualize = True):
	# Return coordinate in center width, height in pixel
	obj_count = np.zeros((len(classes_id)+1), dtype = np.int32)
	all_labels = []
	all_coordinates = []
	all_imgs = []
	all_sizes = []
	list_file = glob.glob(label_folder + '/*.xml')
	print('Found ' + str(len(list_file)) + ' images with labels')
	print('First file is ' + str(list_file[0]))
	for xml in list_file:
		# Image
		img_path = img_folder + '/' + xml.split('/')[-1].split('.')[0] + '.jpg'
		if not os.path.exists(img_path):
			continue
		print('Current: ' + img_path)
		#img, old_size, new_size, padded = resize_with_padding(img_path, target_width, target_height, padding = False)

		img_base = cv2.imread(img_path)
		old_size_height, old_size_width, _ = img_base.shape
		img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
		img = resize_image(img_base)
		new_size_height, new_size_width, _ = img.shape
		img = prewhiten(img)
		all_imgs.append(img)
		# Label
		tree = ET.parse(xml)
		root = tree.getroot()
		label = []
		coordinate = []
		objs = root.findall('object')
		if len(objs) == 0:
			continue
		for obj in objs:
			name = obj.find('name').text
			bndbox = obj.find('bndbox')
			xmin = float(int(bndbox.find('xmin').text)*1.0/old_size_width*new_size_width)
			ymin = float(int(bndbox.find('ymin').text)*1.0/old_size_height*new_size_height)
			xmax = float(int(bndbox.find('xmax').text)*1.0/old_size_width*new_size_width)
			ymax = float(int(bndbox.find('ymax').text)*1.0/old_size_height*new_size_height)
			label.append(int(classes_id[name]))
			obj_count[classes_id[name]] += 1
			coordinate.append([xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2, xmax-xmin, ymax-ymin])
		all_labels.append(label)
		all_coordinates.append(coordinate)
		all_sizes.append((new_size_width, new_size_height))
	print('Total objects: ' + str(obj_count))

	if visualize:
		for i in range(5):
			tmp = random.randint(0, len(list_file)-1)
			img = all_imgs[tmp]#*255
			img = np.array(img, dtype = np.uint8)
			width, height = all_sizes[tmp]
			#img = cv2.resize(img, (width, height))
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			labels = all_labels[tmp]
			coordinates = all_coordinates[tmp]
			for j in range(len(labels)):
				x, y, w, h = coordinates[j]
				x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
				img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
				#class_text = [k for (k, v) in classes_id.iteritems() if v == labels[j]][0]
				#cv2.putText(img, class_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
			cv2.imshow('Example', img)
			cv2.waitKey(0)
	return all_imgs, all_labels, all_coordinates, all_sizes

def normalize_anchors(A, imageWidth, imageHeight):
	# Normalize anchor from (center_x, center_y, width, height) format
	# to (topleft_x, topleft_y, bottomright_x, bottomright_y) format and Scale in [0, 1]


	# Normalize numpy_A
	numpy_A = A.copy()
	numpy_A = numpy_A.reshape((-1, 4))
	numpy_A[:, 0] = numpy_A[:, 0] - numpy_A[:, 2]/2
	numpy_A[:, 1] = numpy_A[:, 1] - numpy_A[:, 3]/2
	numpy_A[:, 2] = numpy_A[:, 2] + numpy_A[:, 0]
	numpy_A[:, 3] = numpy_A[:, 3] + numpy_A[:, 1]
	numpy_A[numpy_A < 0] = 0
	numpy_A[:, 2] = np.clip(numpy_A[:, 2], 0, imageWidth)
	numpy_A[:, 3] = np.clip(numpy_A[:, 3], 0, imageHeight)

	# Scale in [0, 1]
	numpy_A[:, 0] = numpy_A[:, 0]/ imageWidth
	numpy_A[:, 1] = numpy_A[:, 1]/ imageHeight
	numpy_A[:, 2] = numpy_A[:, 2]/ imageWidth
	numpy_A[:, 3] = numpy_A[:, 3]/ imageHeight
	return numpy_A

def rescale_anchors(A, imageWidth, imageHeight):
	# Scale in [0, 1]
	numpy_A = A.copy()
	numpy_A = numpy_A.reshape((-1, 4))
	numpy_A[:, 0] = numpy_A[:, 0]/ imageWidth
	numpy_A[:, 1] = numpy_A[:, 1]/ imageHeight
	numpy_A[:, 2] = numpy_A[:, 2]/ imageWidth
	numpy_A[:, 3] = numpy_A[:, 3]/ imageHeight
	return numpy_A

def format_anchors(A):
	# Convert from center, width, height to topleft bottomright format
	numpy_A = A.copy()
	numpy_A = numpy_A.reshape((-1, 4))
	numpy_A[:, 0] = numpy_A[:, 0] - numpy_A[:, 2]/2
	numpy_A[:, 1] = numpy_A[:, 1] - numpy_A[:, 3]/2
	numpy_A[:, 2] = numpy_A[:, 2] + numpy_A[:, 0]
	numpy_A[:, 3] = numpy_A[:, 3] + numpy_A[:, 1]
	# Thresh

	return numpy_A

def format_anchors_inv(A):
	# Convert from tl br to center, width, height 
	numpy_A = A.copy()
	numpy_A = numpy_A.reshape((-1, 4))
	numpy_A[:, 0] = (numpy_A[:, 0] + numpy_A[:, 2])/2
	numpy_A[:, 1] = (numpy_A[:, 1] + numpy_A[:, 3])/2
	numpy_A[:, 2] = (numpy_A[:, 2] - numpy_A[:, 0])*2
	numpy_A[:, 3] = (numpy_A[:, 3] - numpy_A[:, 1])*2
	# Thresh

	return numpy_A

def iou_tensorial(numpy_A, numpy_B):

	A = numpy_A.copy()
	B = numpy_B.copy()
	B = B.reshape((-1, 4))
	number_boxes_A = len(A)
	number_boxes_B = len(B)
	AA = np.repeat(A, len(B), 0)
	#print(AA)
	BB = np.tile(B, (len(A), 1))
	tl_x = np.minimum(AA[:, 0:1], BB[:, 0:1])
	tl_y = np.minimum(AA[:, 1:2], BB[:, 1:2])
	br_x = np.maximum(AA[:, 2:3], BB[:, 2:3])
	br_y = np.maximum(AA[:, 3:4], BB[:, 3:4])
	w = br_x - tl_x
	h = br_y - tl_y
	w_i = np.clip(AA[:, 2:3] + BB[:, 2:3] - AA[:, 0:1] - BB[:, 0:1] - w, 0, np.Infinity)
	h_i = np.clip(AA[:, 3:4] + BB[:, 3:4] - AA[:, 1:2] - BB[:, 1:2] - h, 0, np.Infinity)
	I = w_i*h_i
	U = (AA[:, 2:3] - AA[:, 0:1])*(AA[:, 3:4] - AA[:, 1:2]) + (BB[:, 2:3] - BB[:, 0:1])*(BB[:, 3:4] - BB[:, 1:2]) - I
	result = I/U
	#print('IoU shape: ' + str(result.shape))
	#print(result)
	return result.reshape((-1, number_boxes_B))

def nms(boxes, probabilities, clss, overlapThresh = 0.7, visualize = False):
	# Non maximum suppression to clean overlap boxes

	# First: Sort boxes by confidence
	temp = zip(probabilities, boxes, clss)
	sorted_temp = sorted(temp, reverse = True)
	boxes = [x for _, x, _ in sorted_temp]
	probabilities = [x for x, _, _ in sorted_temp]
	clss = [x for _, _, x in sorted_temp]
	boxes = np.array(boxes)

	sorted_boxes_formated = format_anchors(boxes)

	# if there are no boxes, return an empty list
	if len(sorted_boxes_formated) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if sorted_boxes_formated.dtype.kind == "i":
		sorted_boxes_formated = sorted_boxes_formated.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = sorted_boxes_formated[:,0]
	y1 = sorted_boxes_formated[:,1]
	x2 = sorted_boxes_formated[:,2]
	y2 = sorted_boxes_formated[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.arange(len(sorted_boxes_formated))
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	#probabilities = probabilities[pick]
	probabilities = np.take(probabilities, pick)
	boxes = np.take(boxes, pick, axis = 0)
	clss = np.take(clss, pick)

	return boxes, probabilities, clss

if __name__ == '__main__':
	#get_data('images', 'images/train', 224, 224)
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
	print(get_correct_label_and_color(3, classes_id))
