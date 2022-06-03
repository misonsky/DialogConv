#coding=utf-8

import tensorflow as tf
from tensorflow import keras
from models.Encoder import Embedding

def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.cast(tf.sqrt(2.0),dtype=input_tensor.dtype)))
    return input_tensor * cdf

class LocalMatch(keras.Model):
    def __init__(self,config):
        super(LocalMatch, self).__init__()
        self.config = config
        self.conv_channel = keras.layers.Conv2D(filters = config.d_model,
                                         kernel_size=(1,1),
                                         strides=(1,1),
                                         padding="SAME",
                                         use_bias=False,
                                         kernel_initializer=tf.random_normal_initializer())
        self.conv_token = keras.layers.Conv2D(filters = config.max_utterance_len,
                                         kernel_size=(1,1),
                                         strides=(1,1),
                                         padding="SAME",
                                         use_bias=False,
                                         kernel_initializer=tf.random_normal_initializer())
        self.phrase_channel = keras.layers.Conv2D(filters = config.d_model,
                                         kernel_size=(1,3),#（1，3）
                                         strides=(1,1),
                                         padding="SAME",
                                         use_bias=False,
                                         kernel_initializer=tf.random_normal_initializer())
        self.phrase_token = keras.layers.Conv2D(filters = config.max_utterance_len,
                                         kernel_size=(1,1),#（1，1）
                                         strides=(1,1),
                                         padding="SAME",
                                         use_bias=False,
                                         kernel_initializer=tf.random_normal_initializer())
    def call(self,x,traing=False):
        residual = x
        
        x = gelu(x)
        x = self.conv_channel(x)
        x = tf.transpose(x, perm=[0,1,3,2])
          
          
        x = gelu(x)
        x = self.conv_token(x)
        x = tf.transpose(x, perm=[0,1,3,2])
         
        x = residual + x
        residual = x
         
         
        x = gelu(x)
        x = self.phrase_channel(x)  #batch * turn * seq * d
        x = tf.transpose(x, perm=[0,1,3,2]) #batch * turn * d *seq 
         
         
        x = gelu(x)
        x = self.phrase_token(x)
        x = tf.transpose(x,perm=[0,1,3,2])#batch * turn  *seq *d 
         
        x = residual + x
        return x

class ContextMatch(keras.Model):
    def __init__(self,config):
        super(ContextMatch, self).__init__()
        self.conv1_token  = keras.layers.Conv1D(filters = config.max_utterance_len * (config.max_turn+1),
                                                kernel_size=5,
                                                strides=1,
                                                padding="SAME",
                                                use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.conv1_channel  = keras.layers.Conv1D(filters = config.d_model,
                                                kernel_size=1,
                                                strides=1,
                                                use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
    def call(self,x,traing=False):
        residual = x
        #sentence-level
        x_shape= x.shape
        x = tf.reshape(x, shape=[x.shape[0],x.shape[1] * x.shape[2],x.shape[-1]])
         
         
        x = gelu(x)
        x = self.conv1_channel(x)
        x = tf.transpose(x,perm=[0,2,1])
         
         
        x = gelu(x)
        x = self.conv1_token(x) 
        x = tf.transpose(x,perm=[0,2,1])
        x = tf.reshape(x,shape=x_shape)
        x = residual + x
        
        return x

class SpatialConv(keras.Model):
    def __init__(self,config):
        super(SpatialConv, self).__init__()
        self.config = config
        self.conv_channel_1 = keras.layers.Conv2D(filters=config.d_model,
                            kernel_size=(1,3),
                            strides=(1,1),
                            padding="SAME")
        
        self.conv_channel_2 = keras.layers.Conv2D(filters=config.d_model,
                            kernel_size=(3,1),
                            strides=(1,1),
                            padding="SAME")
        
        self.conv_channel_3 = keras.layers.Conv2D(filters=config.d_model,
                            kernel_size=(1,1),
                            strides=(1,1),
                            padding="SAME")

    def call(self,x,training):
        residual = x
        
        x = gelu(x)
        x = self.conv_channel_1(x)
        x = self.conv_channel_2(x)
        x = self.conv_channel_3(x)
        return x + residual
        
class EncoderBlock1(keras.Model):
    def __init__(self,config):
        super(EncoderBlock1, self).__init__()
        self.config = config
        
        self.conv_channel = keras.layers.Conv2D(filters = config.d_model,
                                         kernel_size=(1,1),
                                         strides=(1,1),
                                         padding="SAME",
                                         use_bias=False,
                                         kernel_initializer=tf.random_normal_initializer())
        self.conv_token = keras.layers.Conv2D(filters = config.max_utterance_len,
                                         kernel_size=(1,1),
                                         strides=(1,1),
                                         padding="SAME",
                                         use_bias=False,
                                         kernel_initializer=tf.random_normal_initializer())
        self.phrase_channel = keras.layers.Conv2D(filters = config.d_model,
                                         kernel_size=(1,3),#（1，3）
                                         strides=(1,1),
                                         padding="SAME",
                                         use_bias=False,
                                         kernel_initializer=tf.random_normal_initializer())
        self.phrase_token = keras.layers.Conv2D(filters = config.max_utterance_len,
                                         kernel_size=(1,1),#（1，1）
                                         strides=(1,1),
                                         padding="SAME",
                                         use_bias=False,
                                         kernel_initializer=tf.random_normal_initializer())
        self.conv1_token  = keras.layers.Conv1D(filters = config.max_utterance_len * (config.max_turn+1),
                                                kernel_size=5,
                                                strides=1,
                                                padding="SAME",
                                                use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.conv1_channel  = keras.layers.Conv1D(filters = config.d_model,
                                                kernel_size=1,
                                                strides=1,
                                                use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())

    def call(self,x,training):
        #token-level
        residual = x
        
        x = gelu(x)
        x = self.conv_channel(x)
        x = tf.transpose(x, perm=[0,1,3,2])
           
           
        x = gelu(x)
        x = self.conv_token(x)
        x = tf.transpose(x, perm=[0,1,3,2])
          
        x = residual + x
        residual = x
          
          
        x = gelu(x)
        x = self.phrase_channel(x)  #batch * turn * seq * d
        x = tf.transpose(x, perm=[0,1,3,2]) #batch * turn * d *seq 
          
          
        x = gelu(x)
        x = self.phrase_token(x)
        x = tf.transpose(x,perm=[0,1,3,2])#batch * turn  *seq *d 
          
        x = residual + x
        residual = x
        #sentence-level
        x_shape= x.shape
        x = tf.reshape(x, shape=[x.shape[0],x.shape[1] * x.shape[2],x.shape[-1]])
         
         
        x = gelu(x)
        x = self.conv1_channel(x)
        x = tf.transpose(x,perm=[0,2,1])
         
         
        x = gelu(x)
        x = self.conv1_token(x) 
        x = tf.transpose(x,perm=[0,2,1])
        x = tf.reshape(x,shape=x_shape)
        x = residual + x
        
        return x

class EncoderBlock2(keras.Model):
    def __init__(self,config):
        super(EncoderBlock2, self).__init__()
        self.config = config
        self.conv1_token = keras.layers.Conv1D(filters=config.d_model,
                                               kernel_size=3,
                                               strides=1,
                                               padding="SAME",
                                               use_bias=False,
                                               kernel_initializer=tf.random_normal_initializer())
        self.conv1_turns = keras.layers.Conv1D(filters=config.max_turn+1,
                                               kernel_size=1,
                                               strides=1,
                                               padding="SAME",
                                               use_bias=False,
                                               kernel_initializer=tf.random_normal_initializer())
    def call(self,x,training):
        residual = x
        
        
#         x = gelu(x)
#         x = self.conv1_token(x)
        x = tf.transpose(x,perm=[0,2,1])
        
        
        x = gelu(x)
        x = self.conv1_turns(x)
        x = tf.transpose(x,perm=[0,2,1])
        return x + residual

class SuperiviesdModel(keras.Model):
    def __init__(self,vocab_size,embedding_matrix,config):
        super(SuperiviesdModel, self).__init__()
        self.config = config
        self.embedding = Embedding(vocab_size=vocab_size,
                                   embedding_matrix=embedding_matrix,
                                   config=config)
        self.pool_utter = keras.layers.MaxPool2D(pool_size=(1,config.max_utterance_len),
                                            strides=(1,config.max_utterance_len),
                                            padding="SAME")
        
        self.encoderBlock1 = EncoderBlock1(config)
        self.spatial = SpatialConv(config)
        self.encoderBlock2 = EncoderBlock2(config)
        self.max_pool = keras.layers.MaxPool1D(pool_size=config.max_turn+1,
                                               strides=config.max_turn+1,
                                               padding="SAME")
        
        
#         self.output_prj = keras.layers.Dense(1)
        self.output_prj_1 = keras.layers.Dense(2)
    @tf.function
    def call(self,history,positive,negative=None,training=False):
        his=self.embedding(history)
        res=self.embedding(positive)
        exp_res = tf.expand_dims(res,axis=1)
        x = tf.concat([his,exp_res],axis=1) 
        x= self.encoderBlock1(x)#Local and context encoding
       
        if training:
            x = tf.nn.dropout(x,self.config.dropout)
        x = self.spatial(x)
        x = self.pool_utter(x)
        x = tf.reshape(x,shape=[x.shape[0],x.shape[1],x.shape[-1]])
        #global-level
        x = self.encoderBlock2(x,training)
        
        x = self.max_pool(x)
        x = tf.reshape(x,shape=[x.shape[0],x.shape[-1]])
        
#         logits = self.output_prj(x)
        logits = self.output_prj_1(x)
        y_pre = tf.nn.softmax(logits,axis=-1)
        return logits,y_pre
class BaseEncoder(keras.Model):
    def __init__(self,config):
        super(BaseEncoder, self).__init__()
        self.conv1_channel = keras.layers.Conv1D(filters=config.max_utterance_len,
                                               kernel_size=1,
                                               strides=1,
                                               padding="SAME",
                                               use_bias=False,
                                               kernel_initializer=tf.random_normal_initializer())
        self.conv1_token = keras.layers.Conv1D(filters=config.d_model,
                                               kernel_size=3,
                                               strides=1,
                                               padding="SAME",
                                               use_bias=False,
                                               kernel_initializer=tf.random_normal_initializer())
        self.max_pool = keras.layers.MaxPool1D(pool_size=config.max_utterance_len,
                                               strides=config.max_utterance_len,
                                               padding="SAME")
    def call(self,x):
        """
            parameters:
                x: batch * seq * d
            return:
                batch * d
        """
        x = self.conv1_token(x)
        x = tf.transpose(x, perm=[0,2,1])
        x = self.conv1_channel(x)
        x = tf.transpose(x,perm=[0,2,1])
        x = self.max_pool(x)
        x = tf.reshape(x,shape=[x.shape[0],x.shape[-1]])
        return x

class ContrastiveModel(keras.Model):
    def __init__(self,vocab_size,embedding_matrix,config):
        super(ContrastiveModel, self).__init__()
        self.config = config
        self.embedding = Embedding(vocab_size=vocab_size,
                                   embedding_matrix=embedding_matrix,
                                   config=config)
        self.pool_utter = keras.layers.MaxPool2D(pool_size=(1,config.max_utterance_len),
                                            strides=(1,config.max_utterance_len),
                                            padding="SAME")
        
        self.encoderBlock1 = EncoderBlock1(config)
        self.spatial = SpatialConv(config)
        self.encoderBlock2 = EncoderBlock2(config)
        self.max_pool = keras.layers.MaxPool1D(pool_size=config.max_turn,
                                               strides=config.max_turn,
                                               padding="SAME")
        
        
        self.output_prj = keras.layers.Dense(1)
    def cosine_distance(self,x1, x2):
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=-1))
        x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=-1))
        x1_x2 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
        cosin = x1_x2 / (x1_norm * x2_norm)
        return cosin
        
    def encoder(self,x,training):
        for i in range(len(self.encoderBlock1)):
            x = self.encoderBlock1[i](x,training)
        if training:
            x = tf.nn.dropout(x,self.config.dropout)
        x = self.conv_utter(x)
        x = self.pool_utter(x)
        x = tf.reshape(x,shape=[x.shape[0],x.shape[1],x.shape[-1]])
        #global-level
        for i in range(len(self.encoderBlock2)):
            x = self.encoderBlock2[i](x,training)
        return x
    
    @tf.function
    def call(self,history,positive,negative=None,training=False):
        """
        parameter:
            history:batch * turn * seq
            positive:batch * seq
            negative:batch * num * seq
        """
        his=self.embedding(history)
        res=self.embedding(positive)
        exp_res = tf.expand_dims(res,axis=1)
        x = tf.concat([his,exp_res],axis=1) 
        x = self.spatial(x)
        x= self.encoderBlock1(x)
       
        if training:
            x = tf.nn.dropout(x,self.config.dropout)
#         x = self.conv_utter(x)
        x = self.spatial(x)
        x = self.pool_utter(x)
        x = tf.reshape(x,shape=[x.shape[0],x.shape[1],x.shape[-1]])
        #global-level
        x = self.encoderBlock2(x,training)# batch * t * d
        
        response_batch = x[:,-1,:]#batch * d
        contra_list = list()
        for i in range(x.shape[0]):
            anchor = tf.expand_dims(x[i,0:-1,:],axis=0)
            anchor_context = tf.reshape(self.max_pool(anchor),shape=[anchor.shape[-1]])
            p_response = tf.expand_dims(response_batch[i],axis=0)
            pscore = self.cosine_distance(anchor_context,p_response)
            neg_batch = tf.concat([response_batch[0:i,:],response_batch[i:,:]],axis=0)
            nscores = self.cosine_distance(anchor_context,neg_batch)
            scores = tf.concat([pscore,nscores],axis=0)
            contra_list.append(scores)
        
        contra_stack = tf.stack(contra_list,axis=0)
        contra_logits=tf.reshape(contra_stack,shape=[contra_stack.shape[0],contra_stack.shape[1]])
        contra_logits = contra_logits / self.config.temperature
        labels = tf.zeros(shape=[contra_logits.shape[0]])
        return contra_logits,labels
            
        
        
    
            
            
            
            
            
            
            
            
            
                
            
            
        
            
        
        
        
        
        
        
        
        
