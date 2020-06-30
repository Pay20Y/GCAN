# Gaussian Constrained Attention Network for Scene Text Recognition
## Introduction
Implementation of the paper "Gaussian Constrained Attention Network for Scene Text Recognition" (Under Review)
## How to use
### Install
```
pip3 install -r requirements.txt
```
### Train
* <b> Data prepare</b>  
LMDB format is suggested. refer [here](https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py) to generate data in LMDB format.

* <b> Run</b>  
	
	```
	python3 train.py --checkpoints /path/to/save/checkpoints --train_data_dir /path/to/your/train/LMDB/data/dir --test_data_dir /path/to/your/validation/LMDB/data/dir -g "0" --train_batch_size 128 --val_batch_size 128 --aug True --att_loss_type "l1" --att_loss_weight 10.0
	```  
	More hyper-parameters please refer to [config.py](https://github.com/Pay20Y/GCAN/blob/master/config.py)  

### Test

* Download the pretrained model from [BaiduYun](https://pan.baidu.com/s/1hY374pvtDtgeBUPsG7R5ew) (key:w14k)
* Download the benchmark datasets from [BaiduYun](https://pan.baidu.com/s/1Z4aI1_B7Qwg9kVECK0ucrQ) (key: nphk) shared by clovaai in this [repo](https://github.com/clovaai/deep-text-recognition-benchmark)

```
python3 test.py --checkpoints /path/to/the/pretrained/model --test_data_dir /path/to/the/evaluation/benchmark/lmdb/dir -g "0"
```  

## Experiments on benchmarks

|  IIIT5K | IC13  |  IC15  |  SVT | SVTP  |  CUTE  |
|:-------:|:-----:|:------:|:----:|:-----:|:------:|
|  94.4   | 93.3  |  77.1  | 90.1 |  81.2 |  85.6  |