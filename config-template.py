'''

TODO
====

* This file needs to stay within the same folder as <model.py>
* re-name to <config.py>
* fill in absolute path names to SNLI data (.jsonl) 
* run Vered's script (https://github.com/vered1986/PythonUtils/blob/master/word_embeddings/convert_text_embeddings_to_binary.py) 
  to create create a binary file for the word embeddings (enter without file ending)

'''


'''
Path to the pre-trained GloVe embeddings
'''
PATH_WORD_EMBEDDINGS = '/path/to/glove/embeddings/without/file/ending'

'''
Path to the train data
'''
PATH_TRAIN_DATA = '/path/to/snli_1.0_train.jsonl'

'''
Path to the dev data
'''
PATH_DEV_DATA = '/path/to/snli_1.0_dev.jsonl'

'''
Path to the test data
'''
PATH_TEST_DATA = ''