# Data related to the analyses of what the model learns

## Model
* Consists of three layer bi-LSTM with shortcuts (dimensions: 265 + 512 + 1024)
* Feature concatenation of the resulting sentence representations of premise [p] and hypothesis [h]:
	* [p, h, |p-h|, p x h]
	* subtraction and multiplication are done element-wise
* MLP (fed with above features) (dimensions: 1600 + 1600 + 3)
* Trained over 5 iterations over all SNLI train data
	* learnrate: 0.0002 (half decay every 2 iterations)
	* batchsize: 32
	* dropout (MLP outputs): 0.1
* Performace:
	* Train: 87.421%
	* Dev: 85.196%
	* Test: 84.782%

### Learning curve of the Model by amount of data points seen.
[[https://github.com/Max216/ThesisPKinDL/blob/master/analyse_repr_data/0_0002lr-1600hidden-256_512_1024lstm-32batch-43_1-relu-0_1dropout_2017-11-26_22:13_glove.png|alt=Learning curve]]