import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import \
    Conv2D, AveragePooling2D
from skimage import transform
import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self, content_image, style_image):
        super(YourModel, self).__init__()

        # --------------------------------------------------------------------------------------------------------------
        # PART 1 : preprocess/init the CONTENT, STYLE, and CREATION IMAGES #
        # --------------------------------------------------------------------------------------------------------------
        # 1) resize the content and style images to be the same size
        self.content_image = transform.resize(content_image, tf.shape(style_image), anti_aliasing=True,
                                              preserve_range=True)
        self.style_image = transform.resize(style_image, tf.shape(style_image), anti_aliasing=True, preserve_range=True)

        # 2) convert the content and style images to float32 tensors for loss functions (from uint8)
        self.content_image = tf.image.convert_image_dtype(self.content_image, tf.float32)
        self.style_image = tf.image.convert_image_dtype(self.style_image, tf.float32)

        # 3) set the image we are creating as a copy of the tensor that represents the content image
        #   (we do this to give the creation image a good starting point)
        image = tf.image.convert_image_dtype(content_image, tf.float32)
        self.x = tf.Variable([image])

        # --------------------------------------------------------------------------------------------------------------
        # PART 2 : load and configure vgg_16 network use (without classification head) #
        # --------------------------------------------------------------------------------------------------------------
        # 1) load the pretrained vgg_16 network
        self.vgg16 = tf.keras.applications.VGG16(include_top=False, weights='vgg16_imagenet.h5')
        self.vgg16.trainable = False

        # 2) define the layers of the vgg_16 network from which we will extract the content and style features
        self.photo_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

        self.num_photo_layers = len(self.photo_layers)
        self.num_style_layers = len(self.style_layers)

        #  3) get the output (filters and biases) for the defined photo and style layers above
        # only using one filter for the photo layer, so oonly that outpur is needed for our model
        p_output = self.vgg16.get_layer(self.photo_layers[0]).output

        # using multiple filters for the style layers, so we to create the Gram Matrix from each style layers' output
        style_output = []
        for layer in self.style_layers:
            style_output.append(self.vgg16.get_layer(layer).output)

        # map each style layer output to its Gram Matrix
        G = [self.__get_gram(x) for x in style_output]

        # 4) create the vgg16 model from the photo and style layers
        self.vgg16 = tf.keras.Model([self.vgg16.input], [p_output, G])

        # --------------------------------------------------------------------------------------------------------------
        # PART 3 : assign our optimizers, loss weights, and loss/style targets #
        # --------------------------------------------------------------------------------------------------------------
        #  1) use the adam optimizer with hyperparameters defined in the hyperparamters.py
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate, beta_1=hp.beta_1, epsilon=hp.epsilon)

        # 2) assign the loss weights from hyperparameters.py
        self.content_weight = hp.alpha
        self.style_weight = hp.beta

        # 3) get the targets that serve as the baseline of the content and style loss calculations
        # covert images to their float -> numpy representations to call on our model for the targets
        img_to_np = lambda img: np.array([img * 255])
        # content target is the first output of the vgg16 model since it is the output of the photo layer
        self.content_target = self.vgg16(img_to_np(content_image))[0]
        # style target is the second output of the vgg16 mode, the Gram Matrix of the style layers
        self.style_target = self.vgg16(img_to_np(style_image))[1]

    # here for convention - this is the forward pass
    def call(self, x):
        # call only onto our created model
        x = self.vgg16(x * 255)
        return x

    def loss_fn(self, x):
        # since our loss depends on the result of the forward pass (call), we call and get the results
        x = self.call(x)

        # helper functions to calculate the content and style loss
        content_l = self.__content_loss(x[0], self.content_target)
        style_l = self.__style_loss(x[1], self.style_target)

        # return the weighted sum of the content and style loss
        return (self.content_weight * content_l) + (self.style_weight * style_l)

    # called for each epoch and updates the model based on the optimizer and loss function
    def train_step(self, epoch):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # watch how the image changes for backpropagation
            tape.watch(self.x)

            # calculate the loss
            loss = self.loss_fn(self.x)
            gradients = tape.gradient(loss, self.x)

            # print the progress of the training and loss
            print('\rEpoch {}: Loss: {:.4f}'.format(epoch, loss), end='')

        # update the optimizer based on the gradients
        self.optimizer.apply_gradients([(gradients, self.x)])
        # update the image we are creating
        self.x.assign(tf.clip_by_value(self.x, clip_value_min=0.0, clip_value_max=1.0))

    # ------------------------------------------------------------------------------------------------------------------
    # (STATIC) HELPER FUNCTIONS THAT IMPLEMENT THE CALCULATIONS FOR THE GRAM MATRIX AND LOSSES FROM THE REFERENCE
    #             PAPER (https://arxiv.org/pdf/1508.06576.pdf)
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __content_loss(photo_layers, input_layers):
        return tf.reduce_mean(tf.square(photo_layers - input_layers))

    @staticmethod
    def __style_loss(art_layers, input_layers):
        # each layer used for style has a loss
        layer_losses = []
        for created, target in zip(art_layers, input_layers):
            reduced = tf.reduce_mean(tf.square(created - target))
            layer_losses.append(reduced)
        # the total style loss is the sum of each style layer loss
        return tf.add_n(layer_losses)

    @staticmethod
    def __get_gram(style_output):
        style_shape = tf.shape(style_output)
        output = tf.linalg.einsum('bijc,bijd->bcd', style_output, style_output)
        dimensions = style_shape[1] * style_shape[2]
        dimensions = tf.cast(dimensions, tf.float32)
        return output / dimensions
