### prepare word vectors
> word embedding dimension:200
> Please employ [GloVe](https://github.com/stanfordnlp/GloVe) tools to train word vectors on the corresponding corpus. Put it in the corresponding sub-file under the embeddings file

### Preprocessing Dataset

> All parameters are configured in the utils/parameters.py file  
>
> 1. please set $corpus = corpus_name$ parameter and set $do_prepare=True$  
2. preprocess dataset

If negative examples are randomly sampled in the pre-training phase, please set $model=contra"$;
Please use the following function in DatasetInstance.py:

```
def get_TFRecordFunction(dataobj,config):
    TFRecordFunction={"sup":dataobj.SupTFRecoderFeature,
                      "contra":dataobj.ContraTFRecoderFeature,
                      }
    return TFRecordFunction[config.model]

def get_batchFunction(dataobj,config):
    BatchFunction={"sup":dataobj.Sup_batch_data,
                   "contra":dataobj.Contra_batch_data,
                   }
    return BatchFunction[config.model]
```
By default, we use other samples in the same batch as negative samples;  

Next please run command as follows:

```
python runner.py
```

### Pretraing Model

1. If negative examples are randomly sampled in the pre-training phase, please set 	$model=contra"$;   By default, we use other samples in the same batch as negative samples; 

2. please set $do_train=True$, $optimizerType=sgd$ and conduct "python runner.py"

### Train Model
1. please set $model=sup$, $optimizerType=adam$ and conduct "python runner.py"

### Visual Results
1. please set $do\_{prepare}=False, do_train=False, do\_eval=False, do\_predict=False$  and "python runner.py  


