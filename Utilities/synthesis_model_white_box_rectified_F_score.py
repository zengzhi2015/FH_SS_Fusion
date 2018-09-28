# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 20:27:14 2018

@author: Zhi Zeng
"""
import tensorflow as tf
import numpy as np

# Prerequisites
image_height = 240
image_width = 320

class SynthesisModel():

    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.model_scope = tf.variable_scope('model', reuse=tf.AUTO_REUSE)
            self.input_scope = tf.variable_scope('input', reuse=tf.AUTO_REUSE)
            with self.input_scope:
                self.input_feature = tf.placeholder(tf.float32,shape=[None,image_height,image_width,22],name='input_feature')
                self.input_positive_mask = tf.placeholder(tf.float32,shape=[None,image_height,image_width,1],name='positive_mask')
                self.input_negative_mask = tf.placeholder(tf.float32,shape=[None,image_height,image_width,1],name='negative_mask')
            with self.model_scope:
                # Define the model
                self.feature_classical, self.feature_semantic = tf.split(value=self.input_feature,
                                                                         num_or_size_splits=[1,21],
                                                                         axis=-1,
                                                                         num=None,
                                                                         name='feature_split')
                self.feature_classical_stack = tf.concat(values=[self.feature_classical for t in range(21)],
                                                         axis=-1,
                                                         name='concat_feature_classical')
                self.inverse_feature_classical_stack = 1.0-self.feature_classical_stack
                
                self.feature_joint_prob_1 = self.feature_semantic*self.feature_classical_stack
                self.feature_joint_prob_2 = self.feature_semantic*self.inverse_feature_classical_stack
                self.feature_conditonal_prob_1 = tf.get_variable('conditonal_prob_1',
                                                                 shape=[21, 1],
                                                                 initializer=tf.constant_initializer(0.5),
                                                                 trainable=True,
                                                                 constraint=lambda t: tf.clip_by_value(t, 0.0, 1.0))
                self.feature_conditonal_prob_2 = tf.get_variable('conditonal_prob_2',
                                                                 shape=[21, 1],
                                                                 initializer=tf.constant_initializer(0.5),
                                                                 trainable=True,
                                                                 constraint=lambda t: tf.clip_by_value(t, 0.0, 1.0))
                
                self.synthesis_result = tf.tensordot(self.feature_joint_prob_1, self.feature_conditonal_prob_1, axes=[[3], [0]]) + tf.tensordot(self.feature_joint_prob_2, self.feature_conditonal_prob_2, axes=[[3], [0]])
                
                self.variable_summaries(self.synthesis_result,name_scope='synthesis_result')
            
                self.TP = tf.reduce_sum(self.input_positive_mask*(1.0-self.synthesis_result))
                self.TN = tf.reduce_sum(self.input_negative_mask*self.synthesis_result)
                self.FP = tf.reduce_sum(self.input_negative_mask*(1.0-self.synthesis_result))
                self.FN = tf.reduce_sum(self.input_positive_mask*self.synthesis_result)
                
                self.variable_summaries(self.TP,name_scope='TP')
                self.variable_summaries(self.TN,name_scope='TN')
                self.variable_summaries(self.FP,name_scope='FP')
                self.variable_summaries(self.FN,name_scope='FN')
                
                self.Recall = tf.maximum(1e-3,self.TP) / tf.maximum(1e-3,self.TP + self.FN)
                self.Specificity = tf.maximum(1e-3,self.TN) / tf.maximum(1e-3,self.TN + self.FP)
                self.PWC = 100.0 * tf.maximum(1e-3,self.FN + self.FP) / tf.maximum(1e-3,self.TP + self.FN + self.FP + self.TN)
                self.Precision = tf.maximum(1e-3,self.TP) / tf.maximum(1e-3,self.TP + self.FP)
                self.F_Measure = (2 * self.Precision * self.Recall) / (self.Precision + self.Recall)
                
                self.variable_summaries(self.Recall,name_scope='Recall')
                self.variable_summaries(self.Specificity,name_scope='Specificity')
                self.variable_summaries(self.PWC,name_scope='PWC')
                self.variable_summaries(self.Precision,name_scope='Precision')
                self.variable_summaries(self.F_Measure,name_scope='F_Measure')
                
                
                self.loss = 1-self.F_Measure
                self.variable_summaries(self.loss,name_scope='loss')
                self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss, 
                                                                        var_list=[var for var in tf.trainable_variables()]) # original 0.0001
                self.merged_summary = tf.summary.merge_all()
                
                self.saver = tf.train.Saver(max_to_keep=10000000)
                self.train_writer = None
            
    
    def variable_summaries(self,var,name_scope=None):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        if name_scope is None:
            name_scope = var.name.split(':')[0]+'_'+var.name.split(':')[1]
        with tf.name_scope(name_scope):
            if len(var.shape)==0:
                tf.summary.scalar('value', var)
            if len(var.shape)>0:
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
            if len(var.shape)==4 and (var.shape[-1]==1 or var.shape[-1]==3):
                tf.summary.image('image',var)
    
    # Initialization
    def initialize_sess(self,log_path):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            if self.train_writer is not None:
                self.train_writer.close()
            self.train_writer = tf.summary.FileWriter(log_path, self.sess.graph)
    
    # Model training
    def train(self,composite_feature,positive_mask,negative_mask, step=None):
        summary, _, current_loss,synthesis_result = self.sess.run([self.merged_summary,self.optimizer,self.loss,self.synthesis_result],
                                                                  feed_dict={self.input_feature:composite_feature, 
                                                                             self.input_positive_mask: positive_mask,
                                                                             self.input_negative_mask: negative_mask})
        self.train_writer.add_summary(summary, global_step = step)
        
        return current_loss,synthesis_result
        
    # Model estimating
    def estimate(self,composite_feature):
        synthesis_result = self.sess.run([self.synthesis_result],
                                         feed_dict={self.input_feature:composite_feature})
        
        return synthesis_result
    
    def save_model(self,model_path):
        with self.graph.as_default():
            self.saver.save(self.sess,model_path + '\\model')
    
    def load_model(self,model_path):
        with self.graph.as_default():
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
    
    def __del__(self):
        self.sess.close()

# Calculate loss weight
def calculate_double_mask(cv_gray_truth):
    """Calculate the weight for calculating the loss of the model"""
    cv_gray_positive_mask = np.zeros_like(cv_gray_truth.astype('float'))
    cv_gray_negative_mask = np.zeros_like(cv_gray_truth.astype('float'))
    
    index_object = np.where(cv_gray_truth==255)
    index_background = np.where(cv_gray_truth==0)
    
    cv_gray_positive_mask[index_object] = 1.0
    cv_gray_negative_mask[index_background] = 1.0
    
    return np.expand_dims(cv_gray_positive_mask,-1),np.expand_dims(cv_gray_negative_mask,-1)

# Single input feature builder
def single_feature_builder(raw_segmentation,seg_logits_normalized):
    return np.concatenate([np.expand_dims(raw_segmentation,axis=-1),seg_logits_normalized[:image_height,:image_width,:]],axis=-1)