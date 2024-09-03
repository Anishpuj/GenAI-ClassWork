# -*- coding: utf-8 -*-
"""Copy of Variational AutoEncoder CIFAR 10.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WY-x-ZbAM49BXClpG7b3Lj_FBBfrxktM

# Define sampling layer
Custom layer: subclassing the abstract layer class and defining the call method which describes how a tensor is transformed by the layer

Subclass inherits sttributes and methods of parent class
"""

import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt



# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train.shape

import tensorflow as tf
from tensorflow.keras import layers, models, metrics, backend as K
from tensorflow.keras.losses import binary_crossentropy

class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding an item of
  clothing."""
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""# Defining the Encoder"""

encoder_input = layers.Input(shape=(32, 32, 3),name="encoder_input")
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
z_mean = layers.Dense(2, name="z_mean")(x) #mapping the flattened layer to z-mean, 2 is the dimnesion of the latent space
z_log_var = layers.Dense(2, name="z_log_var")(x) #mapping the flattened layer to z-mean, 2 is the dimnesion of the latent space
z = Sampling()([z_mean, z_log_var])
encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

"""# Define loss function"""

class KLLossLayer(layers.Layer):
    """
    Custom layer to calculate KL Divergence loss.
    """
    def call(self, z_mean, z_log_var):
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        return kl_loss

# ... (Your existing code for encoder)

# Instantiate the custom layer
kl_loss_layer = KLLossLayer()

# Calculate KL loss using the layer
kl_loss = kl_loss_layer(z_mean, z_log_var)

"""#Define the decoder"""

decoder_input = layers.Input(shape=(2,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation = 'relu',padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation = 'relu',padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation = 'relu',padding="same")(x)
decoder_output = layers.Conv2D(3, (3, 3), strides = 1, activation="sigmoid",padding="same", name="decoder_output")(x)
decoder = models.Model(decoder_input, decoder_output)
decoder.summary()

"""# Train VAE"""

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        """Call the model on a particular input."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = 500 * tf.reduce_mean(
                binary_crossentropy(data, reconstruction))
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

# Assuming the decoder model is defined elsewhere
# decoder = ...

vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")
vae.fit(
    x_train,
    epochs=5,
    batch_size=100
)

"""#Making Predictions"""

example_images = x_test[:5000]
predictions = vae.predict(example_images)
plt.figure(figsize=(1,1))
plt.imshow(predictions[2][51])

plt.figure(figsize=(0.7,0.7))
plt.imshow(example_images[51])

"""#Visualize sample image"""

x_train[0]
plt.figure(figsize=(0.7,0.7))
plt.imshow(x_train[100])

embeddings = encoder.predict(example_images)
embeddings[0].shape #z_mean

"""#Mapping the low dimensional latent space"""

# embeddings = encoder.predict(example_images) #encodings to low dimensional space of example images
plt.figure(figsize=(8, 8))
color_values=y_test[:5000]
plt.scatter(embeddings[0][:, 0], embeddings[0][:, 1],  c=color_values, cmap='tab10', alpha=0.5, s=3)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# embeddings = encoder.predict(example_images) #encodings to low dimensional space of example images
plt.figure(figsize=(8, 8))
plt.scatter(embeddings[0][:, 0], embeddings[0][:, 1],  c=color_values, cmap='tab10', alpha=0.5, s=3)

color_values=y_test[:5000]

plt.colorbar(ticks=range(10), label='Categories')


offset=[300,300]# distance where to display the actual
for index in [51,210]:
    x0, y0 = embeddings[0][index]# get the 2D embedding of the selected image
    image = example_images[index]  #get the actual image from mnist dataset

    # Create an OffsetImage
    imagebox = OffsetImage(image, zoom=1.5)  # Adjust zoom as needed

    # Create an AnnotationBbox with an offset
    ab = AnnotationBbox(
        imagebox,
        (x0, y0),
        frameon=False,
        xybox=offset,
        xycoords='data',
        boxcoords="offset points",
        pad=0.5,
        arrowprops=dict(arrowstyle="->", color='red')
    )

    # Add AnnotationBbox to the plot
    plt.gca().add_artist(ab)

# Adjust plot limits to ensure there's space for the annotations
# plt.xlim(min(embeddings[:, 0]) - 1, max(embeddings[:, 0]) + 1)
# plt.ylim(min(embeddings[:, 1]) - 1, max(embeddings[:, 1]) + 1)

# Show plot
plt.show()

# image.shape
example_images=x_test[:5000]
plt.figure(figsize=(0.7,0.7))
plt.imshow(example_images[210])

"""#Generating new images"""

import numpy as np
import matplotlib.pyplot as plt

# Assuming you have the decoder model defined and trained
# Here, `latent_dim` is the dimensionality of the latent space. For example, 2.

latent_dim = 2  # This should match the dimensionality used in your VAE model

# Function to generate new images
def generate_new_images(decoder, num_images=10):
    # Sample random points from a standard normal distribution
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))

    # Decode these latent vectors to generate new images
    generated_images = decoder.predict(random_latent_vectors)

    # Plot the generated images
    plt.figure(figsize=(2,2))
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i].reshape(32, 32,3))  # Assuming image size is 32x32x3
        plt.axis('off')
    plt.show()

# Generate and display new images
generate_new_images(decoder, num_images=3)

"""--------------------**MY CUSTOM CODE AND USING DIFFERENT DATASET**--------------------

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, metrics, backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and normalize MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for convolutional layers (add channel dimension)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train.shape  # Should be (60000, 28, 28, 1)
x_test.shape   # Should be (10000, 28, 28, 1)

# Define sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the encoder
encoder_input = layers.Input(shape=(28, 28, 1), name="encoder_input")
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
z_mean = layers.Dense(2, name="z_mean")(x)
z_log_var = layers.Dense(2, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

# Define KL loss layer
class KLLossLayer(layers.Layer):
    def call(self, z_mean, z_log_var):
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        return kl_loss

# Define the decoder
decoder_input = layers.Input(shape=(2,), name="decoder_input")
x = layers.Dense(7 * 7 * 64)(decoder_input)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding="same")(x)
decoder_output = layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same", name="decoder_output")(x)
decoder = models.Model(decoder_input, decoder_output)

# prompt: reduce the KL loss less tha n1

# Define VAE model
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = 0.1 * tf.reduce_mean(
                binary_crossentropy(data, reconstruction))
            kl_loss = 0.01 * tf.reduce_mean(  # Reduced KL loss weight
                tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")
vae.fit(
    x_train,
    epochs=5,
    batch_size=100
)

# Making Predictions
example_images = x_test[:5000]
predictions = vae.predict(example_images)

plt.figure(figsize=(1,1))
plt.imshow(predictions[2][51].reshape(28, 28), cmap='gray')

plt.figure(figsize=(0.7,0.7))
plt.imshow(example_images[51].reshape(28, 28), cmap='gray')

# Generate new images
latent_dim = 2

def generate_new_images(decoder, num_images=10):
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))
    generated_images = decoder.predict(random_latent_vectors)
    plt.figure(figsize=(2,2))
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

generate_new_images(decoder, num_images=3)