#save the training classifier
# serialize model to JSON

from sklearn.model_selection import train_test_split


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

classifier = Sequential()
model_json = classifier.to_json()
with open("airanalysisltrained.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("airanalysisltrained.h5")
print("Saved model to disk")

# load json and create model
json_file = open('airanalysisltrained.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("airanalysisltrained.h5")
print("Loaded model from disk")
