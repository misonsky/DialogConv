#coding=utf-8
import tensorflow as tf
import os
from utils.parameters import Parameters
from utils.hf_argparser import HfArgumentParser
from trainer import Trainer

def run():
    parser = HfArgumentParser(Parameters)
    config = parser.parse_args_into_dataclasses()[-1]
    if config.fp16:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy) 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    trainer_handle = Trainer(config)
    if config.do_prepare:
        trainer_handle.prepare()
    elif config.do_train:
        trainer_handle.train()
    elif config.do_eval:
        eval_metrics = trainer_handle.evaluate(datasetInstance=None,only_evaluation=True)
        format_string=" "
        for k,v in eval_metrics.items():
            format_string += str(k) +" {} ".format(v)
        tf.print(format_string)
    elif config.do_predict:
        trainer_handle.predict(datasetInstance=None)
    else:
        trainer_handle.VisualConv(datasetInstance=None)
if __name__=="__main__":
    run()
