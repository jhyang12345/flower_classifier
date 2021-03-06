from model import Classifier
from prepare_dataset import fetch_training_data
from keras.callbacks import ModelCheckpoint

class Trainer:
    def __init__(self):
        self.classifier = Classifier()
        self.model = self.classifier.model
        self.training_data, self.training_output, self.testing_data, self.testing_output = \
            fetch_training_data()
        self.checkpointer = ModelCheckpoint(filepath='flower.weights.best.hdf5', verbose=1,
                               save_best_only=True)
        self.epochs = 20
        self.train()

    def train(self):
        self.model.fit(self.training_data, self.training_output, epochs=self.epochs,
                        validation_data=(self.testing_data, self.testing_output),
                        callbacks=[self.checkpointer], verbose=1, shuffle=True)


if __name__ == '__main__':
    Trainer()
