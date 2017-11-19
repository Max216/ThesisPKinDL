import embeddingholder
import train
import mydataloader
import config
from config import *

embedding_holder = embeddingholder.EmbeddingHolder(PATH_WORD_EMBEDDINGS)

        
snli_train = mydataloader.get_dataset_chunks(PATH_TRAIN_DATA, embedding_holder, chunk_size=5)
snli_dev = mydataloader.get_dataset_chunks(PATH_DEV_DATA, embedding_holder, chunk_size=5)

print(snli_train)


#model, epochs, dev_acc, train_acc = train_model(classifier, snli_train, snli_dev, 
#            embedding_holder.padding(),
#            F.cross_entropy, lr, epochs=50, batch_size=5, validate_after=5)

#print('best after ', epochs, ' acc:', dev_acc, train_acc)

#lrs = [0.00005,0.00002]
#dimens_hidden=[800,1600]
#dimens_sent_encoder = [[64,128,256], [128,256,512]]
#batch_sizes=[16,32]

lrs = [0.00005]
dimens_hidden=[800]
dimens_sent_encoder = [[64,128,256]]
batch_sizes=[5]

train.search_best_model(snli_train, snli_dev, embedding_holder, lrs, dimens_hidden, dimens_sent_encoder, batch_sizes, epochs=5, validate_after=40)