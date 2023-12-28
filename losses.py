import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import \
    Conv2D, AveragePooling2D
from skimage import transform
import hyperparameters as hp

def get_gram(style_output):
    style_shape = tf.shape(style_output)
    output = tf.linalg.einsum('bijc,bijd->bcd', style_output, style_output)
    dimensions = style_shape[1] * style_shape[2]
    dimensions = tf.cast(dimensions, tf.float32)
    return output / dimensions


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self, content_image, style_image): #normalize these images to float values
        super(YourModel, self).__init__()
       
        self.content_image = transform.resize(content_image, tf.shape(style_image), anti_aliasing=True, preserve_range=True)
        self.content_image = tf.image.convert_image_dtype(self.content_image, tf.float32)
        
        self.style_image = transform.resize(style_image, tf.shape(style_image), anti_aliasing=True, preserve_range=True)
        self.style_image = tf.image.convert_image_dtype(self.style_image, tf.float32)
        
        image = tf.image.convert_image_dtype(content_image, tf.float32)
        self.x = tf.Variable([image])

        self.content_weight = hp.alpha
        self.style_weight = hp.beta

        self.photo_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
       
        self.num_photo_layers = len(self.photo_layers)
        self.num_style_layers = len(self.style_layers)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate, beta_1=hp.beta_1, epsilon=hp.epsilon)

        self.vgg16 = tf.keras.applications.VGG16(include_top=False, weights='vgg16_imagenet.h5')
        
        self.vgg16.trainable = False

       #  creating the Gram Matrix
        p_output = self.vgg16.get_layer(self.photo_layers[0]).output
        style_output = []
        for layer in self.style_layers:
               style_output.append(self.vgg16.get_layer(layer).output)

        G = [get_gram(x) for x in style_output]

        self.vgg16 = tf.keras.Model([self.vgg16.input], [p_output, G])

       #  figure this out Michael
        img_to_np = lambda img: np.array([img * 255])

        self.content_target = self.vgg16(img_to_np(content_image))[0]
        self.style_target = self.vgg16(img_to_np(style_image))[1]

        # create a map of the layers to their corresponding number of filters if it is a convolutional layer

    def call(self, x):
       # call onto our pretrained network, since we don't have a classifcation head to follow
       x = self.vgg16(x * 255)
       return x

    def loss_fn(self, x):
       x = self.call(x)  
       content_l = self.content_loss(x[0], self.content_target)
       style_l = self.style_loss(x[1], self.style_target)
       return (self.content_weight * content_l) + (self.style_weight * style_l)
        
    def content_loss(self, photo_layers, input_layers):
       return tf.reduce_mean(tf.square(photo_layers - input_layers))

    def style_loss(self, art_layers, input_layers):
       layer_losses = []
       for created, target in zip(art_layers, input_layers):
              reduced = tf.reduce_mean(tf.square(created - target))
              layer_losses.append(reduced)
       return tf.add_n(layer_losses)


    def train_step(self, epoch):
       with tf.GradientTape(watch_accessed_variables=False) as tape:
              tape.watch(self.x)
              # loss = self.loss_fn(self.content_image, self.style_image, self.x)
              loss = self.loss_fn(self.x)
              print('\rEpoch {}: Loss: {:.4f}'.format(epoch, loss), end='')
              gradients = tape.gradient(loss, self.x)
       self.optimizer.apply_gradients([(gradients, self.x)])
       self.x.assign(tf.clip_by_value(self.x, clip_value_min=0.0, clip_value_max=1.0))

