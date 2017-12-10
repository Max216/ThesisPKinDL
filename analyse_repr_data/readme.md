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

