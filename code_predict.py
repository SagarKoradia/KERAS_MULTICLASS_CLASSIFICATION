import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# fix random seed for reprofucibility
seed = 7
numpy.random.seed(seed)

# load dataset
fn = r'C:\Users\DELL I5558\Desktop\Python\NSW-ER01.csv'
dataframe = pd.read_csv(fn)
dataset = dataframe.values
X = dataset[:, 0:22].astype(float)
Y = dataset[:, 22]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=22, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

X_train, x_test, Y_train, y_test = train_test_split(X, dummy_y, test_size=0.30, random_state=seed)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(x_test)
print(predictions)
print(encoder.inverse_transform(predictions))
