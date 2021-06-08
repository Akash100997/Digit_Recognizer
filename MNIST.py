import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np

# Get the MNIST Dataset from Keras
mnist = keras.datasets.mnist

# Split the data's into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data's 0,255 -> 0,1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the Model
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

print(model.summary())

# Loss and Optimizer
loss = keras.losses.SparseCategoricalCrossentropy()
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

# Compile the model
model.compile(optimizer=optim, loss=loss, metrics=metrics)

# Train the model
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

# Evaluate the model
model.evaluate(x_test, y_test, verbose=2, batch_size=batch_size)

# Prediction
predictions = model.predict(x_test)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

# call argmax for multiple labels
pred06s = predictions[0:6]
print(pred06s.shape)
label05s = np.argmax(pred06s, axis=1)
print(label05s)

# Plot the images and Check if the model has predicted correctly
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_test[i], cmap='gray')
plt.show()
