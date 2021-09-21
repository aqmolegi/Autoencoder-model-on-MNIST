from tensorflow.keras.layers import Conv2D, Flatten, Conv2DTranspose, Reshape, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

def Conv_layers(input, filters, kerenal_size, activation, padding = 'same', encoder_part = False):

    # the encoder part
    if encoder_part:
        x = Conv2D( filters = filters,
                    kernel_size = kerenal_size,
                    activation = activation,
                    padding = padding)(input)
    
    # the decoder part
    else:
        x = Conv2DTranspose(filters = filters,
                            kernel_size = kernel_size,
                            activation = activation,
                            padding = padding)(input)
    return x

latent_Dim = 20
filters = 20
kernel_size = 3
encoder_layers_num = 2
batch_size = 200
epochs = 5

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
input_shape = (image_size,image_size,1)
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])/255
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])/255

# encoder model
inputs = Input(shape = input_shape)
x = inputs
for i in range(encoder_layers_num):
    x = Conv_layers(x, filters, kernel_size, activation='relu', padding='same', encoder_part=True)

# get the shape of x before Flatten
shape = K.int_shape(x)

x = Flatten()(x)
outputs = Dense(latent_Dim)(x)
encoder = Model(inputs, outputs)
encoder.summary()
plot_model(encoder, to_file='MNIST_A_encoder_layers.png', show_shapes = True)

# decoder model
latent_inputs = Input(shape=(latent_Dim,))
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv_layers(x, filters, kernel_size, activation='relu', padding='same')
outputs = Conv_layers(x, 1, kernel_size, activation='sigmoid', padding='same')

decoder = Model(latent_inputs, outputs)
decoder.summary()
plot_model(decoder, to_file='MNIST_A_decoder_layers.png', show_shapes=True)

# autoencoder model
autoencoder = Model(inputs, decoder(encoder(inputs)))
autoencoder.summary()
plot_model(autoencoder, to_file='MNIST_Autoencoder_layers.png', show_shapes=True)

autoencoder.compile(loss='mse', optimizer='adam')

# train the autoencoder
autoencoder.fit(x_train,
                x_train,
                validation_data = (x_test, x_test),
                epochs = epochs,
                batch_size = batch_size)

# predict the autoencoder output
x_decoded = autoencoder.predict(x_test)

# display 10 test input and decoded images
imgs = np.concatenate([x_test[:10], x_decoded[:10]])
imgs = imgs.reshape((2, 10, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('The first row is the input and the second row is the Decoded images.')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('Input_and_decoded_images.png')
plt.show()
