
#ONLY_TEST = True

'''
Path to the pre-trained GloVe embeddings
'''

PATH_WORD_EMBEDDINGS = './models/snli_glove.840B.300d'
#if ONLY_TEST:
#    PATH_WORD_EMBEDDINGS = '/home/max/Dokumente/Masterthesis/data/glove/glove.6B/glove_supertiny'

'''
Path to the train data
'''
#PATH_TRAIN_DATA = '/home/max/Dokumente/Masterthesis/data/snli_1.0/snli_1.0/snli_1.0_train.jsonl'
PATH_TRAIN_DATA = '/home/max/Dokumente/Masterthesis/data/snli_1.0/snli_1.0/snli_train_30.jsonl'

'''
Path to the dev data
'''
#PATH_DEV_DATA = '/home/max/Dokumente/Masterthesis/data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl'
PATH_DEV_DATA = '/home/max/Dokumente/Masterthesis/data/snli_1.0/snli_1.0/snli_dev_15.jsonl'

'''
Path to the test data
'''
PATH_TEST_DATA = ''

'''
Name of the model that is trained and saved on the disk
'''
FILENAME_BEST_MODEL = 'trained_model.py'
