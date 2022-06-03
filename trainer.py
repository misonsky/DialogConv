#coding=utf-8
import os
import json
import datetime
import math
import numpy as np
import random
import matplotlib as mpl
import pickle as pkl
from evaluation_tool import DoubanMetrics
from evaluation_tool import MutualMetrics
from evaluation_tool import ContraDoubanMetrics,ContraMutualMetrics
from evaluation_tool import TopMetrics
from DatasetInstance import DatasetInstance
from models.Network import SuperiviesdModel,ContrastiveModel
from models.optimization import create_optimizer,GradientAccumulator
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.patheffects as PathEffects
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,rc={"lines.linewidth": 2.5})
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def validation_parameters(config):
    if not config.train and not config.eval:
        raise ValueError("must specify one of them")
    model_dir=os.path.join(config.model_dir,config.corpus)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
def get_metaFiles(config):
    meteFile={"sup":"sup_meta.json","contra":"contra_meta","smn":"smn_meta.json"}
    return meteFile[config.model]

def get_record(fileName,config):
    """
        fileName must be %s_fileName
    """
    Tfrecords={"sup":fileName%("sup"),
               "contra":fileName%("sup"),#fileName%("contra_%d"%(config.negatives_num)),
               }
    return Tfrecords[config.model]

def get_metrics_key(config):
    metrics_dict={"sup":"R10@1","contra":"acc_1"}
    return metrics_dict[config.mode]

def get_TFRecordFunction(dataobj,config):
    TFRecordFunction={"sup":dataobj.SupTFRecoderFeature,
                      "contra":dataobj.SupTFRecoderFeature,#dataobj.ContraTFRecoderFeature,
                      }
    return TFRecordFunction[config.model]

def get_batchFunction(dataobj,config):
    BatchFunction={"sup":dataobj.Sup_batch_data,
                   "contra":dataobj.Sup_batch_data,#dataobj.Contra_batch_data,
                   }
    return BatchFunction[config.model]

def get_model(config,vocab_size):
    models={"sup":SuperiviesdModel,"contra":ContrastiveModel}#,"contra":ContrastiveModel
    emb_file=os.path.join(config.data_dir,config.corpus,config.emb_file)
    with open(emb_file,"rb") as f:
        embedding_matrix=pkl.load(f)
    train_model=models[config.model]
    model_handle = train_model(vocab_size=vocab_size,
                    embedding_matrix=embedding_matrix,
                    config=config)
    return model_handle
 

class Trainer(object):
    def __init__(self,config,model=None,optimizer=None,LRSchedule=None):
        self.lr_scheduler=LRSchedule
        self.config = config
        self.model=model
        self.optimizer=optimizer
        self.gradient_accumulator = GradientAccumulator()
        self.global_step = 0
        self.meta_dict = None
        self.key_metrics = "R10@1" if self.config.model != "contra" else "acc_1"
        self.best_metrics = -1
    def prepare(self):
        meta_file=os.path.join(self.config.data_dir,self.config.corpus,get_metaFiles(self.config))
        train_file=os.path.join(self.config.data_dir,self.config.corpus,self.config.train_files)
        train_record=os.path.join(self.config.data_dir,self.config.corpus,get_record(self.config.train_record,self.config))
        dev_file=os.path.join(self.config.data_dir,self.config.corpus,self.config.dev_files)
        dev_record=os.path.join(self.config.data_dir,self.config.corpus,get_record(self.config.dev_record,self.config))
        test_file=os.path.join(self.config.data_dir,self.config.corpus,self.config.test_files)
        test_record=os.path.join(self.config.data_dir,self.config.corpus,get_record(self.config.test_record,self.config))
        if not os.path.isfile(os.path.join(self.config.data_dir,self.config.corpus,"dataInstance.pkl")):
            datasetInstance=DatasetInstance(train_file)
            with open(os.path.join(self.config.data_dir,self.config.corpus,"dataInstance.pkl"),"wb") as f:
                pkl.dump(datasetInstance,f)
        else:
            with open(os.path.join(self.config.data_dir,self.config.corpus,"dataInstance.pkl"),"rb") as f:
                datasetInstance=pkl.load(f)
        emb_file=os.path.join(self.config.data_dir,self.config.corpus,self.config.emb_file)
        if not os.path.isfile(emb_file):
            datasetInstance.generate_embedding(self.config)
        tfrecord_function = get_TFRecordFunction(datasetInstance,self.config)
        meta_dict=dict()
        num_train = tfrecord_function(train_file, train_record,self.config)
        num_dev = tfrecord_function(dev_file, dev_record,self.config)
        num_test = 0
        if self.config.corpus == "mutual":
            if self.config.model =="sup":
                num_test = tfrecord_function(test_file, test_record,self.config,True)
            else:
                num_test = tfrecord_function(test_file, test_record,self.config)
        else:
            num_test = tfrecord_function(test_file, test_record,self.config)
        meta_dict["num_train"]=num_train
        meta_dict["num_dev"]=num_dev
        meta_dict["num_test"]=num_test
        self.meta_dict = meta_dict
        with open(meta_file,"w",encoding="utf-8") as f:
            json.dump(meta_dict, f)
    def loss_function(self,real, pred,clip_value=10):
        loss_ = loss_object(real, pred)
        return tf.reduce_mean(loss_)

    def run_model(self, features,loss_flag,training=False):
        """
            parameters:
                features:dict()
                return:
                    sup: loss, logits, prediction_label
                    contra: loss, logits, real label
        """
        his=features["history"]
        res=features["response"]
        placeholder=features["placeholder"]
        his=tf.reshape(his,shape=[res.shape[0],self.config.max_turn,self.config.max_utterance_len])
#         if self.config.model == "contra":
#             placeholder = tf.reshape(placeholder,shape=[res.shape[0],self.config.negatives_num,self.config.max_utterance_len])
        
        output1,output2=self.model(his,res,placeholder,training=training)
        batch_loss = 0.0
        if loss_flag:
            if self.config.model == "contra":
                logits = output1
                labels  = output2
#                 batch_loss = tf.keras.losses.mean_squared_error(labels,logits)
                batch_loss=self.loss_function(labels, logits)
            else:
                logits = output1
                batch_loss=self.loss_function(placeholder, logits)
        if self.config.model == "sup":
            return batch_loss, logits,output2
        else:
            return batch_loss,output1,output2
    def training_step(self, features):
        """
        Perform a training step on features and labels.

        Subclass and override to inject some custom behavior.
        """
        per_example_loss, _,_ = self.run_model(features,True,training=True)
#         scaled_loss = per_example_loss / tf.cast(nb_instances_in_global_batch, dtype=per_example_loss.dtype)
        gradients = tf.gradients(per_example_loss, self.model.trainable_variables)
        gradients = [
            g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.model.trainable_variables)
        ]

        if self.config.gradient_accumulation_steps > 1:
            self.gradient_accumulator(gradients)
        self.train_loss.update_state(per_example_loss)
        if self.config.gradient_accumulation_steps == 1:
            return gradients
    
    @tf.function
    def apply_gradients(self, features):
        if self.config.gradient_accumulation_steps == 1:
            gradients = self.training_step(features)

            self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))
        else:
            for _ in tf.range(self.config.gradient_accumulation_steps):
                reduced_features = {
                    k: ft[: self.config.train_batch_size] for k, ft in features.items()
                }
                
                self.training_step(reduced_features)
                features = {
                    k: tf.concat(
                        [ft[self.config.train_batch_size:], reduced_features[k]],
                        axis=0,
                    )
                    for k, ft in features.items()
                }

            gradients = self.gradient_accumulator.gradients
#             gradients = [(tf.clip_by_value(grad, -self.config.max_grad_norm, self.config.max_grad_norm)) for grad in gradients
#             ]

            self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))
            self.gradient_accumulator.reset()
    def get_flops(self,model):
        concrete = tf.function(lambda inputs1,inputs2: model(inputs1,inputs2))
        concrete_func = concrete.get_concrete_function(
            tf.TensorSpec(shape=(1,10,50)),
            tf.TensorSpec(shape=(1,50)))
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
            return flops.total_float_ops
    def train(self):
        meta_file = os.path.join(self.config.data_dir,self.config.corpus,get_metaFiles(self.config))
        with open(meta_file,"r",encoding="utf-8") as f:
            meta_dict=json.load(f)
        self.meta_dict = meta_dict
        train_record=os.path.join(self.config.data_dir,self.config.corpus,self.config.train_record)
        model_dir=os.path.join(self.config.model_dir,self.config.corpus,self.config.model)
        with open(os.path.join(self.config.data_dir,self.config.corpus,"dataInstance.pkl"),"rb") as f:
            datasetInstance=pkl.load(f)
        tf.print("Info load data object...............")
        num_update_steps_per_epoch = self.meta_dict["num_train"] / self.config.train_batch_size
        approx = math.floor if self.config.drop_last else math.ceil
        num_update_steps_per_epoch = approx(num_update_steps_per_epoch)
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        self.steps_per_epoch = num_update_steps_per_epoch
        self.total_batch_size = self.config.train_batch_size * self.config.gradient_accumulation_steps
        t_total = num_update_steps_per_epoch * self.config.num_train_epochs
        batchFunction = get_batchFunction(datasetInstance,self.config)
        train_dataset=batchFunction(
            config=self.config,
            recordFile=get_record(train_record,self.config),
            batch_size=self.total_batch_size,
            is_training=True)
        tf.print("generate batch dataset...................")
        max_word_index=max(datasetInstance.DataDict["text2id"].word_index.values())
        self.gradient_accumulator.reset()
        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        if not self.model:
            self.model = get_model(self.config, vocab_size=max_word_index+1)
        
        
        iterations = self.optimizer.iterations
        checkpoint_prefix = os.path.join(model_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,model=self.model)
        ckpt_manager = tf.train.CheckpointManager(checkpoint,checkpoint_prefix, max_to_keep=5)
        self.ckpt_handle = ckpt_manager
        steps_trained_in_current_epoch=0
        if ckpt_manager.latest_checkpoint and self.config.load_last_ckpt:
            tf.print("Checkpoint file %s found and restoring from checkpoint", ckpt_manager.latest_checkpoint)
            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
            self.global_step =  iterations.numpy()
#                epochs_trained = self.global_step // self.steps_per_epoch
            steps_trained_in_current_epoch = self.global_step % self.steps_per_epoch    
        tf.summary.experimental.set_step(self.global_step)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss') 
        self.train_loss.reset_states()
        start_time = datetime.datetime.now()
        for (batch, features) in enumerate(train_dataset):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            self.apply_gradients(features)
#                self.distributed_training_steps(features)
            self.global_step = iterations.numpy()
#             self.model.summary()
#             forward_pass = tf.function(
#             self.model.call,
#             input_signature=[tf.TensorSpec(shape=(32,10,50)),tf.TensorSpec(shape=(32,50))]
#             )
#             graph_info = profile(forward_pass.get_concrete_function().graph,
#                         options=ProfileOptionBuilder.float_operation())
#             flops = graph_info.total_float_ops // 2
#             print('Flops: {:,}'.format(flops))    
#             print("The FLOPs is:{}".format(self.get_flops(self.model)) ,flush=True )
            if self.global_step % self.config.log_steps == 0:
                tf.print("log info {} to step {} train loss {}".format((self.global_step-self.config.log_steps),self.global_step,self.train_loss.result()))
            if self.global_step % self.config.eval_steps == 0:
                metrics_result = self.evaluate(datasetInstance=datasetInstance)
                format_string="step {} ".format(self.global_step)
                for k,v in metrics_result.items():
                    format_string += str(k) +" {} ".format(v)
                tf.print(format_string)
        end_time = datetime.datetime.now()
        tf.print("Training took: {}".format(str(end_time - start_time))) 

    @tf.function       
    def prediction_step(self,features,loss_flag=False):
        _, logits, placeholder = self.run_model(features,loss_flag,training=False)
#         self.eval_loss.update_state(per_example_loss)
        return logits,placeholder
    def padd_numobj(self,inputs,max_len):
        if inputs.shape[-1] < max_len:
            inputs = inputs.tolist()
            inputs = [vec + [0] * (max_len-len(vec)) for vec in inputs]
            return np.asarray(inputs, dtype=np.float32)
        return inputs
    def prediction_loop(self,dataset,prediction_type="eval"):
        placeholders: np.ndarray = None
        preds_logits: np.ndarray = None
        real_labels: np.ndarray = None
#         self.eval_loss = tf.keras.metrics.Sum(name="eval_loss")
#         self.eval_loss.reset_states()
        loss_flag = prediction_type == "eval"
        for _, features in enumerate(dataset):
            logits,placeholder = self.prediction_step(features,loss_flag)
            if real_labels is None:
                real_labels = features["placeholder"].numpy()
            else:
                real_labels = np.append(real_labels,features["placeholder"].numpy(), axis=0)
            if preds_logits is None:
                preds_logits = logits.numpy()
            else:
                preds_logits = np.append(preds_logits,logits.numpy(), axis=0)
            if placeholders is None:
                placeholders = placeholder.numpy()
            else:
                placeholders = np.append(placeholders, placeholder.numpy(), axis=0)
        eval_metrics=dict()
        if prediction_type =="eval":
            if self.config.model =="contra":
#                 cosine_score = np.mean(placeholders)
                contra_label_list = placeholders.tolist()
                contra_logits_list = preds_logits.tolist()
                eval_metrics = TopMetrics(top_list=[1,2,5,10], label_list=contra_label_list, logits_list=contra_logits_list)
            else:
                if self.config.corpus == "mutual":
                    eval_metrics=MutualMetrics(placeholders,real_labels,count=4)
                else:
                    eval_metrics=DoubanMetrics(placeholders,real_labels,count=10)
            return eval_metrics
        elif prediction_type =="prediction":
            if self.config.model =="sup":
                if self.config.corpus == "mutual":
                    eval_metrics=MutualMetrics(placeholders,real_labels,count=4)
                else:
                    eval_metrics=DoubanMetrics(placeholders,real_labels,count=10)
            else:
                if self.config.corpus == "mutual":
                    eval_metrics=ContraMutualMetrics(preds_logits.tolist(),real_labels.tolist(),count=4)
                else:
                    eval_metrics=ContraDoubanMetrics(preds_logits.tolist(),real_labels.tolist(),count=10)
            format_string=" "
            for k,v in eval_metrics.items():
                format_string += str(k) +" {} ".format(v)
            tf.print(format_string)
            return None    
            
    def predict(self,datasetInstance = None):
        if datasetInstance is None:
            with open(os.path.join(self.config.data_dir,self.config.corpus,"dataInstance.pkl"),"rb") as f:
                datasetInstance=pkl.load(f)
            tf.print("Info load data object...............")
        test_record=os.path.join(self.config.data_dir,self.config.corpus,self.config.test_record)
        batchFunction = get_batchFunction(datasetInstance,self.config)
        test_dataset = batchFunction(config=self.config,
                                    recordFile=get_record(test_record,self.config),
                                    batch_size=self.config.eval_batch_size,
                                    is_training=False)
        max_word_index=max(datasetInstance.DataDict["text2id"].word_index.values())
        self.model = get_model(self.config, vocab_size=max_word_index+1)
        model_dir=os.path.join(self.config.model_dir,self.config.corpus,"sup")
        self.create_optimizer_and_scheduler(num_training_steps=1000)
        checkpoint_prefix = os.path.join(model_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,model=self.model)
        ckpt_manager = tf.train.CheckpointManager(checkpoint,checkpoint_prefix, max_to_keep=5)
        tf.print("Checkpoint file found and restoring from checkpoint", ckpt_manager.latest_checkpoint)
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        self.prediction_loop(test_dataset, prediction_type="prediction")
        
    def evaluate(self,datasetInstance = None,only_evaluation=False):
        if datasetInstance is None:
            with open(os.path.join(self.config.data_dir,self.config.corpus,"dataInstance.pkl"),"rb") as f:
                datasetInstance=pkl.load(f)
            tf.print("Info load data object...............")        
        dev_record=os.path.join(self.config.data_dir,self.config.corpus,self.config.dev_record)    
        batchFunction = get_batchFunction(datasetInstance,self.config)
        dev_dataset = batchFunction(config=self.config,
                                    recordFile=get_record(dev_record,self.config),
                                    batch_size=self.config.eval_batch_size,
                                    is_training=False)
        if only_evaluation:
            max_word_index=max(datasetInstance.DataDict["text2id"].word_index.values())
            self.model = get_model(self.config, vocab_size=max_word_index+1)
            model_dir=os.path.join(self.config.model_dir,self.config.corpus,"sup")
            self.create_optimizer_and_scheduler(num_training_steps=1000)
            checkpoint_prefix = os.path.join(model_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,model=self.model)
            ckpt_manager = tf.train.CheckpointManager(checkpoint,checkpoint_prefix, max_to_keep=5)
            tf.print("Checkpoint file found and restoring from checkpoint", ckpt_manager.latest_checkpoint)
            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        
        metrics_result = self.prediction_loop(dev_dataset, prediction_type="eval")
        if not only_evaluation:
            if metrics_result[self.key_metrics] >= self.best_metrics:
                self.best_metrics = metrics_result[self.key_metrics]
                self.save_model()
        return metrics_result
    def image_pool_show(self,key_name,layer_result):
        v_shape = layer_result.shape
        size_1 = v_shape[1]#turns
        n_features = v_shape[-1]
        value = layer_result[0,:,:]
        display_grid = np.zeros((size_1, n_features))
        for i in range(n_features):
            display_grid[:, i: (i + 1)] = value[:,i: (i + 1)]
        scale = 0.6
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(key_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
    def imgage_conv_show(self,key_name,layer_result,num_turn,seq_lens):
        examples=None
        font1 = {'family':"Times New Roman",'weight':'normal','size':5}
        with open("datasets/mutual/visual.txt","r",encoding="utf-8") as f:
            for line in f:
                examples = line.split("\t")[1:]
        examples=[utteranes.split() for utteranes in examples]
        v_shape = layer_result.shape
        size_1 = len(examples)#turn
        n_features = v_shape[-1]
        value = layer_result[0,:,:,:]
        value = value.numpy()
        for j in range(size_1):
            display_grid = np.zeros((seq_lens[j], n_features))
            for i in range(n_features):
                visual_features = value[j,:seq_lens[j],i: (i + 1)]
                visual_features  *= 64
                visual_features += 128
                display_grid[:, i: (i + 1)] = visual_features
            scale = 0.6
            vnorm = mpl.colors.Normalize(vmin=20, vmax=200)
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(key_name)
            ticks = [i for i in range(len(examples[j]))]
            labels = examples[j]
            plt.yticks(ticks, labels)
            plt.tick_params(labelsize=10)
            plt.grid(False)
            plt.ylabel("",font1)
            plt.imshow(display_grid, aspect='auto', cmap='viridis',norm=vnorm)
            plt.colorbar()
            plt.show()
    def TsneConvVisual(self,layer_result):
        value = layer_result[0,:,:,]
        v_shape = value.shape
        value = value.numpy()
        turns_number = [i for i in range(v_shape[0])]
        tsne = TSNE(n_components=2,init="pca",verbose=1)
        vectors_ = tsne.fit_transform(value)
        plt.figure(figsize=(14,10))
        plt.scatter(vectors_[:,0],vectors_[:,1])
        for i in range(v_shape[0]):
            x = vectors_[i][0]
            y = vectors_[i][1]
            plt.text(x,y,turns_number[i])
        plt.show()
        
    def VisualConv(self,datasetInstance = None):
        if datasetInstance is None:
            with open(os.path.join(self.config.data_dir,self.config.corpus,"dataInstance.pkl"),"rb") as f:
                datasetInstance=pkl.load(f)
            tf.print("Info load data object...............")
        visual_file=os.path.join(self.config.data_dir,self.config.corpus,self.config.visual_files)
        visual_record=os.path.join(self.config.data_dir,self.config.corpus,self.config.visual_record) 
        tfrecord_function = get_TFRecordFunction(datasetInstance,self.config)
        num_dev = tfrecord_function(visual_file, visual_record,self.config)
        num_turn,seq_lens = datasetInstance.get_visual(visual_file)
        batchFunction = get_batchFunction(datasetInstance,self.config)
        dev_dataset = batchFunction(config=self.config,
                                    recordFile=visual_record,
                                    batch_size=1,
                                    is_training=False)
        max_word_index=max(datasetInstance.DataDict["text2id"].word_index.values())
        self.model = get_model(self.config, vocab_size=max_word_index+1)
        model_dir=os.path.join(self.config.model_dir,self.config.corpus,"sup")
        self.create_optimizer_and_scheduler(num_training_steps=1000)
        checkpoint_prefix = os.path.join(model_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,model=self.model)
        ckpt_manager = tf.train.CheckpointManager(checkpoint,checkpoint_prefix, max_to_keep=5)
        tf.print("Checkpoint file found and restoring from checkpoint", ckpt_manager.latest_checkpoint)
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        conv_keys = ['G', 'G1', 'G2', 'G3', 'G4', 'G6', 'G7', 'G8', 'G9', 'G10']
        pool_keys = ['G11', 'G12', 'G13']
        for features in dev_dataset:
            his=features["history"]
            res=features["response"]
            placeholder=features["placeholder"]
            his=tf.reshape(his,shape=[res.shape[0],self.config.max_turn,self.config.max_utterance_len])
            self.model(his,res,placeholder,training=False)
            result = self.model.get_intermediate()
            for tsne_key in pool_keys:
                self.TsneConvVisual(result[tsne_key])
            for conv_key in conv_keys:
                self.imgage_conv_show(conv_key,result[conv_key],num_turn,seq_lens)
            for pool_key in pool_keys:
                self.image_pool_show(pool_key,result[pool_key])
                    
                    
                
            
    def save_model(self):
        save_path = self.ckpt_handle.save()
        tf.print("save the model {}".format(save_path))
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        TFTrainer's init through :obj:`optimizers`, or subclass and override this method.
        """
        if not self.optimizer and not self.lr_scheduler:
            self.optimizer, self.lr_scheduler = create_optimizer(
                self.config.learning_rate,
                num_training_steps,
                self.config.warmup_steps,
                optimizer_type=self.config.optimizerType,
                adam_beta1=self.config.adam_beta1,
                adam_beta2=self.config.adam_beta2,
                adam_epsilon=self.config.adam_epsilon,
                weight_decay_rate=self.config.weight_decay,
            )
    














