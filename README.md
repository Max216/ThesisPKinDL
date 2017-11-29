# ThesisPKinDL
Repository for sharing code from the theses to integrate external resources into neural networks

## To run (on python 3)
Install
* numpy
* pytorch
* nltk (for tokenizer), only need the package punkt `python -m nltk.downloader 'punkt'`
* matplotlib

Download
* Gove embeddings (300D 840B)
* SNLI dataset

Steps
* Run `data_tools.py req_embeddings` on the glove embeddings and SNLI data to get the subset of embeddings that occur within SNLI
* Run Vered's script (https://github.com/vered1986/PythonUtils/blob/master/word_embeddings/convert_text_embeddings_to_binary.py) to create a binary file of the word embeddings
* Copy `config-template.py` to a new file `config.py` and update the values to correct paths
