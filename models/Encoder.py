#coding=utf-8
import tensorflow as tf
from tensorflow import keras
from utils.trans_utils import point_wise_feed_forward_network,scaled_dot_product_attention,positional_encoding


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    def split_heads(self, x, batch_size):
        """
            return (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (ba:124 calltch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights

class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    def call(self, x,  training=False,mask=None,):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2
    
class TransformerEncoder(keras.layers.Layer):
    def __init__(self,
                 config):
        super(TransformerEncoder, self).__init__()
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.enc_layers = [EncoderLayer(config.d_model, config.num_heads, config.dff, config.dropout) for _ in range(config.num_layers)]
      
        self.dropout = tf.keras.layers.Dropout(config.dropout)
    def call(self, x, mask,training):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x,None # (batch_size, input_seq_len, d_model)
class Recurrent(keras.layers.Layer):
    def __init__(self,
                 config):
        super(Recurrent, self).__init__()
        self.gru = tf.keras.layers.GRU(config.d_model,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer=keras.initializers.glorot_normal())
    def call(self, x, mask=None, training=False):
        whole_states, final_state = self.gru(x)
        return whole_states, final_state

class Embedding(keras.layers.Layer): 
    def __init__(self,
                 vocab_size,
                 embedding_matrix,
                 config):
        super(Embedding, self).__init__()
        self.encoder_type=config.encoder_type
        self.d_model = config.d_model
        if config.encoder_type == "gru":
            self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                  config.emb_size,
                                  embeddings_initializer=keras.initializers.constant(embedding_matrix),
                                  trainable=False)
        elif config.encoder_type == "transformer":
            self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                  config.emb_size,
                                  embeddings_initializer=keras.initializers.constant(embedding_matrix),
                                  trainable=False)
            self.pos_encoding = positional_encoding(config.maximum_position, config.d_model)
    def call(self,x):
        if self.encoder_type == "gru":
            emb=self.embedding(x)
            return emb
        elif self.encoder_type == "transformer":
            emb = self.embedding(x)
            emb *= tf.math.sqrt(tf.cast(self.d_model, emb.dtype))
            p_emb=[]
            if len(emb.shape)==4:
                emb_unstack = tf.unstack(emb,axis=1)
                for instance_emb in emb_unstack:
                    instance_emb += tf.cast(self.pos_encoding,dtype=emb.dtype)
                    p_emb.append(instance_emb)
                p_emb=tf.stack(p_emb,axis=1)
                return p_emb    
            return emb
        
        
