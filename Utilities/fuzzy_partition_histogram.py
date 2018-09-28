# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:31:59 2018

@author: Zhi Zeng
"""

import tensorflow as tf
import cv2

# Prerequisites
image_height = 240
image_width = 320
num_bin = 20+1
bin_width = 1.0/(num_bin-1)
max_sum = 50.0

class FuzzyHistogramModel():

    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.model_scope = tf.variable_scope('model', reuse=tf.AUTO_REUSE)
            self.input_scope = tf.variable_scope('input', reuse=tf.AUTO_REUSE)
            with self.model_scope:
                # Define the model
                self.fuzzy_histogram = tf.get_variable('fuzzy_histogram',
                                                       shape=[image_height, image_width, num_bin],
                                                       initializer=tf.constant_initializer(1.0),
                                                       trainable=False)
                # Intermitent constants
                self.indexing_constant = tf.constant([[[r,c] for c in range(image_width)] for r in range(image_height)],
                                                     name='indexing_constant')
            with self.input_scope:
                # Define the input image
                self.input_image = tf.placeholder(tf.float32, 
                                                  shape=[image_height,image_width], 
                                                  name='image')
                # Define the synthesis result
                self.synthesis_result = tf.placeholder(tf.float32, 
                                                       shape=[image_height,image_width], 
                                                       name='synthesis_result')
                # Define the training flag
                self.training_flag = tf.placeholder(tf.bool, 
                                                    shape=[], 
                                                    name='training_flag')
            
            self.define_histogram_checking_ops()
            self.define_update_weight_calculation_ops()
            self.define_update_histogram_ops()
            self.define_reduce_histogram_ops()
            
            self.saver = tf.train.Saver(max_to_keep=10000000)
            
    # Initialization
    def initialize_sess(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
    
    # Define histogram checking
    def define_histogram_checking_ops(self):
        with self.graph.as_default():
            self.data_position_in_histogram = self.input_image/bin_width
            self.pre_index = tf.to_int32(tf.floor(self.data_position_in_histogram))
            self.pre_weight = self.data_position_in_histogram-tf.to_float(self.pre_index)
            self.next_index = self.pre_index+1
            self.next_weight = tf.to_float(self.next_index)-self.data_position_in_histogram
            self.max_bin_value = tf.reduce_max(self.fuzzy_histogram,axis=-1)
            # the checking result has to be converted to the probability. 
            self.raw_segmentation = tf.divide(tf.add(tf.multiply(tf.gather_nd(self.fuzzy_histogram,
                                                                              tf.concat([self.indexing_constant,tf.expand_dims(self.pre_index, -1)], -1)),
                                                                 self.pre_weight),
                                                     tf.multiply(tf.gather_nd(self.fuzzy_histogram,
                                                                              tf.concat([self.indexing_constant,tf.expand_dims(self.next_index, -1)], -1)),
                                                                 self.next_weight)),
                                              self.max_bin_value,
                                              name='raw_segmentation_calculation')
    
    def histogram_checking(self,gray_float_image):
        '''
        Inputs:
            data_to_check: [height, width], float (from 0.0 to 1.0)
            self.histogram_to_use: [height, width, num_bins], float
        Outputs:
            result: [height, width], float
        '''
        with self.graph.as_default():
            raw_segmentation_in_np_in_float = self.sess.run(self.raw_segmentation, feed_dict={self.input_image: gray_float_image})
        return raw_segmentation_in_np_in_float
        
    # Define fake_synthesis_generate
    def fake_synthesis_generate(self,raw_segmentation_in_np_in_float):
        # The following can be replaced by avr_pooling
        raw_segmentation_in_np_in_float_scaled_to_255 = 255.0*raw_segmentation_in_np_in_float
        raw_segmentation_in_np_in_uint8 = raw_segmentation_in_np_in_float_scaled_to_255.astype('uint8')
        blurred_mask_uint8 = cv2.medianBlur(raw_segmentation_in_np_in_uint8,9)
        blurred_mask_float = blurred_mask_uint8.astype('float')/255
        return blurred_mask_float
        
    # Define update_weight_calculation_ops
    def define_update_weight_calculation_ops(self):
        # a*y^5/(x+b) a = 0.0792; b = 0.1585
        with self.graph.as_default():
            self.update_mask = (1-tf.to_float(self.training_flag))*tf.divide(0.0792*self.synthesis_result**5,self.raw_segmentation+0.1585,name='update_weight_calculation')+tf.to_float(self.training_flag)*self.synthesis_result
    
    # Define update_histogram 
    def define_update_histogram_ops(self):
        with self.graph.as_default():
            self.fuzzy_histogram_add_pre = self.fuzzy_histogram + tf.sparse_tensor_to_dense(tf.SparseTensor(indices=tf.reshape(tf.to_int64(tf.concat([self.indexing_constant, tf.expand_dims(self.pre_index, -1)], -1)),
                                                                                                                               [image_height*image_width,3]), 
                                                                                            values=tf.reshape(tf.multiply(self.pre_weight,self.update_mask),
                                                                                                              [image_height*image_width,]), 
                                                                                            dense_shape=[image_height,image_width,num_bin]))
            self.fuzzy_histogram_add_next = self.fuzzy_histogram_add_pre + tf.sparse_tensor_to_dense(tf.SparseTensor(indices=tf.reshape(tf.to_int64(tf.concat([self.indexing_constant, tf.expand_dims(self.next_index, -1)], -1)),
                                                                                                                     [image_height*image_width,3]), 
                                                                                                     values=tf.reshape(tf.multiply(self.next_weight,self.update_mask),
                                                                                                                       [image_height*image_width,]), 
                                                                                                     dense_shape=[image_height,image_width,num_bin]))

    def update_histogram(self,gray_float_image,synthesis_map,train_flag):
        with self.graph.as_default():
            new_hist = self.sess.run(self.fuzzy_histogram_add_next, feed_dict={self.input_image: gray_float_image, 
                                                                               self.synthesis_result: synthesis_map,
                                                                               self.training_flag: train_flag})
            self.fuzzy_histogram.load(new_hist,self.sess)
    
    # Define reduce_histogram (One should never normalize the histogram)
    def define_reduce_histogram_ops(self):
        with self.graph.as_default():
            self.reducing_factor = 1.0+(self.max_bin_value - max_sum)/max_sum
            self.fuzzy_histogram_reduced = tf.divide(self.fuzzy_histogram,
                                                     tf.tile(tf.expand_dims(self.reducing_factor, -1),
                                                             [1, 1, num_bin]), 
                                                     name='fuzzy_histogram_reduced')

    def reduce_histogram(self):
        with self.graph.as_default():
            new_hist = self.sess.run(self.fuzzy_histogram_reduced)
            self.fuzzy_histogram.load(new_hist,self.sess)
    
    def write_graph(self,log_path):
        with self.graph.as_default():
            tf.summary.FileWriter(log_path, self.sess.graph)
    
    def save_model(self,model_path):
        with self.graph.as_default():
            self.saver.save(self.sess,model_path + '\\model')
    
    def load_model(self,model_path):
        with self.graph.as_default():
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
    
    def __del__(self):
        self.sess.close()