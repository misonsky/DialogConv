#coding=utf-8
from dataclasses import dataclass,field
from utils.file_utils import cached_property,tf_required
import tensorflow as tf

@dataclass
class Parameters(object):
    data_dir: str = field(default="datasets",metadata={"help":"dataset path"})
    gpu: str = field(default="0",metadata={"help":"which gpu device to use"})
    corpus: str = field(default="mutual",metadata={"help":"which corpus to use"})
    emb_path: str = field(default="embeddings",metadata={"help":"embedding path"})
    emb_file: str = field(default="emb.pkl",metadata={"help":"embedding files"})
    train_files: str= field(default="train.txt",metadata={"help":"train files"})
    dev_files: str= field(default="dev.txt",metadata={"help":"the dev file evaluating the model"})
    visual_files: str= field(default="visual.txt",metadata={"help":"the visual file evaluating the model"})
    test_files: str= field(default="dev.txt",metadata={"help":"the test file"})
    train_record: str= field(default="%s_train.tfrecord",metadata={"help":"the train tfrecorder file"})
    dev_record: str= field(default="%s_dev.tfrecord",metadata={"help":"the dev tfrecorder file"})
    test_record: str= field(default="%s_test.tfrecord",metadata={"help":"the test tfrecorder file"})
    visual_record: str= field(default="visual_record.tfrecord",metadata={"help":"the test tfrecorder file"})
    model: str= field(default="contra",metadata={"help":"which task choices"})
    encoder_type: str= field(default="gru",metadata={"help":"which encoder choice"})
    model_dir: str=field(default="TrainModel",metadata={"help":"path to save model"})
    optimizerType: str = field(default="adam",metadata={"help":"optimizer model"})
    
    load_last_ckpt: bool = field(default=False,metadata={"help":"whether training the model from the last checkpoint"})
    no_cuda: bool = field(default=False,metadata={"help":"whether use the cuda device"})
    do_prepare: bool = field(default=False,metadata={"help":"prepare the dataset"})
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    lower_case: bool = field(default=False,metadata={"help":"whether lower the token"})
    fp16: bool= field(default=True,metadata={"help":"using the mixed_float16 when traing"})
    drop_last: bool= field(default=False,metadata={"help":"whether drop the last dataset"})
    
    max_turn: int= field(default=10,metadata={"help":"max number turn of conversation"})
    max_utterance_len: int= field(default=50,metadata={"help":"max length of utterance"})
    eval_steps: int= field(default=50,metadata={"help":"number steps eval the model"})
    log_steps: int= field(default=10,metadata={"help":"number steps log info"})
    m_layers: int= field(default=1,metadata={"help":"the number of encoder layers"})
    e_layers: int= field(default=1,metadata={"help":"the number of encoder layers"})
    num_heads: int= field(default=8,metadata={"help":"head number of multi-head attention"})
    dff: int= field(default=128,metadata={"help":"dff size of encoder"})
    maximum_position: int= field(default=50,metadata={"help":"maximum position"})
    gradient_accumulation_steps: int= field(default=1,metadata={"help":"gradient accumulation steps"})
    filter_size: int= field(default=5,metadata={"help":"the filter size of conv"})
    train_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    eval_batch_size: int = field(default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."})
    d_model: int= field(default=512,metadata={"help":"the hidden size of model"})
    emb_size: int=field(default=512,metadata={"help":"the embedding dimension"})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    num_train_epochs: int = field(default=500, metadata={"help": "Total number of training epochs to perform."})
    negatives_num: int = field(default=50,metadata={"help":"the negatives number when training"})
    
    dropout: float= field(default=0.1,metadata={"help":"dropout rate"})
    temperature:float = field(default=0.07,metadata={"help":"dropout rate"})
    learning_rate: float = field(default=0.001, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    temperature: float =field(default=0.07,metadata={"help":"the temperature value when contrastive learning"})    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
