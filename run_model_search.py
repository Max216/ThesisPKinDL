import embeddingholder
import train
import mydataloader
import config
from config import *
print('Start script: model search')
embedding_holder = embeddingholder.EmbeddingHolder(PATH_WORD_EMBEDDINGS)


if ONLY_TEST:
	lrs = [0.0002]
	dimens_hidden=[400]
	dimens_sent_encoder = [[32,64,128]]
	batch_sizes=[5]
	chunk_size = 5#
	validate_after = 30
	epochs=5
else:
	lrs = [0.0002]
	dimens_hidden=[800]
	dimens_sent_encoder = [[64,128,256]]
	batch_sizes=[32]
	chunk_size = 32*400
	validate_after = 500
	epochs=10
        
snli_train = mydataloader.get_dataset_chunks(PATH_TRAIN_DATA, embedding_holder, chunk_size=chunk_size, mark_as='[train]')
snli_dev = mydataloader.get_dataset_chunks(PATH_DEV_DATA, embedding_holder, chunk_size=chunk_size, mark_as='[dev]')


#model, epochs, dev_acc, train_acc = train_model(classifier, snli_train, snli_dev, 
#            embedding_holder.padding(),
#            F.cross_entropy, lr, epochs=50, batch_size=5, validate_after=5)

#print('best after ', epochs, ' acc:', dev_acc, train_acc)

#lrs = [0.00005,0.00002]
#dimens_hidden=[800,1600]
#dimens_sent_encoder = [[64,128,256], [128,256,512]]
#batch_sizes=[16,32]



train.search_best_model(snli_train, snli_dev, embedding_holder, lrs, dimens_hidden, dimens_sent_encoder, batch_sizes, epochs=epochs, validate_after=validate_after)