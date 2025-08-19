import tensorflow as tf
from tensorflow.keras import layers  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# CONFIGURATIONS
# -------------------------
latent_dim = 100
epochs = 200
batch_size = 128
image_save_interval = 20

os.makedirs('generated_images', exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
(train_images, _), _ = tf.keras.datasets.mnist.load_data()
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
train_images = np.expand_dims(train_images, axis=-1)
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size).repeat()

# -------------------------
# GENERATOR
# -------------------------
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# -------------------------
# DISCRIMINATOR
# -------------------------
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# -------------------------
# LOSS + OPTIMIZERS
# -------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(real_output), real_output) + \
           cross_entropy(tf.zeros_like(fake_output), fake_output)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# -------------------------
# TRAIN STEP
# -------------------------
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# -------------------------
# GENERATE & SAVE IMAGE
# -------------------------
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'generated_images/image_epoch_{epoch:04d}.png')
    plt.close()

# -------------------------
# TRAINING LOOP
# -------------------------
seed = tf.random.normal([16, latent_dim])
steps_per_epoch = len(train_images) // batch_size
dataset_iter = iter(dataset)

for epoch in range(1, epochs + 1):
    for _ in range(steps_per_epoch):
        image_batch = next(dataset_iter)
        g_loss, d_loss = train_step(image_batch)

    if epoch % image_save_interval == 0 or epoch == 1:
        generate_and_save_images(generator, epoch, seed)
        print(f"Epoch {epoch}, Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}")

