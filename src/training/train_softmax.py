from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  KFold



import numpy as np

import pickle

from detectfaces_mtcnn.Configuration import get_logger
from training.softmax import SoftMax

class TrainFaceRecogModel:

    def __init__(self, args):
        self.args = args
        self.logger = get_logger()
        self.data = pickle.loads(open(args["embeddings"], "rb").read())


    def trainKerasModelForFaceRecognition(self):
        le = LabelEncoder()
        labels = le.fit_transform(self.data['names'])
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)
        one_hot_encoder = OneHotEncoder(categorical_features= [0])
        labels = one_hot_encoder.fit_transform(labels).toarray()

        embeddings = np.array((self.data['embeddings']))

        BATCH_SIZE = 8
        EPOCHS = 5
        input_shape = embeddings.shape[1]

        softmax = SoftMax(input_shape=(input_shape,), num_classes= num_classes)
        model = softmax.build()

        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        history = {'acc':[], 'val_acc':[],'loss':[],'val_loss':[]}

        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val , y_train,y_val = embeddings[train_idx], embeddings[valid_idx],labels[train_idx],labels[valid_idx]
            his = model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val,y_val))
            print(his.history['loss'])

            history['acc'] += his.history['accuracy']
            history['val_acc'] += his.history['val_accuracy']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']

            self.logger.info(his.history['accuracy'])

        model.save(self.args['model'])
        f = open(self.args['le'],'wb')
        f.write(pickle.dumps(le))
        f.close()