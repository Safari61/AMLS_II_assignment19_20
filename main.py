import tempfile
import pickle
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split

# load data
image_size = 150
pickle_in1 = open('y_150', 'rb')
y = pickle.load(pickle_in1)
y = np.array(y)
pickle_in2 = open('X_150', 'rb')
X = pickle.load(pickle_in2)
X = np.array(X)
X = X.reshape(-1, image_size, image_size, 1)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04)

# Set up pruning configurations
new_pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.2,
        final_sparsity=0.92,
        begin_step=10000,
        end_step=20000
    )
}

# Build the model
model = Sequential()
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(96, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 1.1
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(96, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 1.2
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(96, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 1.3
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(96, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 1.4
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(96, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 1.5
model.add(MaxPooling2D(pool_size=(5, 5), padding='same'))
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(256, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 2.1
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(256, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 2.2
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(384, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 3
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(384, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 4
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Conv2D(256, (3, 3),
           input_shape=X_train.shape[1:],
           activation='relu',
           padding='same'),
    **new_pruning_params)
)  # Pruned Conv layer 5
model.add(Flatten())
model.add(tfmot.sparsity.keras.prune_low_magnitude(
    Dense(1, activation='sigmoid'),
    **new_pruning_params)
)  # FC 8

# Compiling configurations
myadam = optimizers.Adam(learning_rate=0.00008,
                         beta_1=0.9,
                         beta_2=0.999,
                         amsgrad=False
                         )
model.compile(loss='binary_crossentropy',
              optimizer=myadam,
              metrics=['accuracy']
              )
logdir = tempfile.mkdtemp()
callbacks = [
    # Update the pruning step
    tfmot.sparsity.keras.UpdatePruningStep(),
    # Add summaries to keep track of the sparsity in different layers during training
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)
]

# Train the model
acc = model.fit(X_train,
                y_train,
                batch_size=32,
                epochs=10,
                validation_split=0.16666,
                callbacks=callbacks
                )
acc_train = acc.history['accuracy'][9]
acc_val = acc.history['val_accuracy'][9]

# Test the model
test_result = model.evaluate(X_test, y_test)
acc_test = test_result[1]

# Save the model
final_model = tfmot.sparsity.keras.strip_pruning(model)
final_model.save('pruning')

# ======================================================================================================================
print(f'Training Accuracy: {acc_train}, '
      f'Validation Accuracy: {acc_val}, '
      f'Test Accuracy: {acc_test}.')
