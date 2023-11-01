# imports
import optuna
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD

CATEGORIES = 10
EPOCHS = 10

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # scale inputs to be in range [0-1]
    X_train /= 255
    X_test /= 255

    Y_train = to_categorical(y_train, CATEGORIES)
    Y_test = to_categorical(y_test, CATEGORIES)

    return X_train, Y_train, X_test, Y_test

def create_model(trial):
    # Suggest hyperparameter ranges
    num_filters = trial.suggest_int('num_filters', 16, 64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    optimizer_selected = trial.suggest_categorical('optimizer', ['rmsprop', 'adam', 'sgd'])

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last', input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    # Compile the model with the suggested optimizer and learning rate
    if optimizer_selected == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_selected == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_selected == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    
    return model

def objective(trial):
    # Create the model using the suggested hyperparameters
    model = create_model(trial)

    # Also experiment with batch sizes
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    
    # Fit the model
    history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=EPOCHS, verbose=0)
    
    # Return the test accuracy (the trial's objective function)
    accuracy = model.evaluate(X_test, Y_test, verbose=0)[1]
    return accuracy

X_train, Y_train, X_test, Y_test = load_data()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=4, n_jobs=4, show_progress_bar=True)
print(study.best_params)
