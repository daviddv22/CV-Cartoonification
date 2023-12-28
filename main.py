import os
import sys
import argparse

import tensorflow as tf
from skimage import transform

import hyperparameters as hp
from model import YourModel

from skimage.io import imread, imsave
from matplotlib import pyplot as plt
import numpy as np


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--content',
        required=True,
        help='''Content image filepath''')
    parser.add_argument(
        '--style',
        required=True,
        help='Style image filepath')
    parser.add_argument(
        '--savefile',
        required=True,
        help='Filename to save image')

    return parser.parse_args()


def save_tensor_as_image(tensor, path, img_name):
    # make copy of tensor
    copy = tf.identity(tensor)
    copy = tf.squeeze(copy)
    # convert tensor to back to uint8 image
    copy = tf.image.convert_image_dtype(copy, tf.uint8)

    # save image (make path if it doesn't exist)
    if not os.path.exists(path):
        os.makedirs(path)
    imsave(path + img_name, copy)

    # return copy if used
    return copy


def train(model: YourModel):
    # do as many epochs from hyperparameters.py
    for i in range(hp.num_epochs):
        # save a checkpoint every 100 epochs
        if i % 100 == 0:
            save_tensor_as_image(model.x, 'out/checkpoints/cp-{}/'.format(ARGS.savefile),
                                 '{}_epoch.{}'.format(i, ARGS.savefile.split('.')[-1]))

        # do the training step
        model.train_step(i)


def main():
    """ Main function. """
    # --------------------------------------------------------------------------------------------------------------
    # PART 1 : parse the arguments #
    # --------------------------------------------------------------------------------------------------------------
    if os.path.exists(ARGS.content):
        ARGS.content = os.path.abspath(ARGS.content)
    if os.path.exists(ARGS.style):
        ARGS.style = os.path.abspath(ARGS.style)
    if os.path.exists('out/checkpoints/cp-{}/'.format(ARGS.savefile)):
        # print an error to the console if the checkpoint directory already exists
        print('Error: out/checkpoints/cp-{}/ already exists. Please choose a different name.'.format(ARGS.savefile))
        return
    os.chdir(sys.path[0])

    # --------------------------------------------------------------------------------------------------------------
    # PART 2 : read and process the style and content images #
    # --------------------------------------------------------------------------------------------------------------
    # 1) read content and style images
    content_image = imread(ARGS.content)
    style_image = imread(ARGS.style)
    # 2) make the style image the same size as the content image
    style_image = transform.resize(style_image, content_image.shape, anti_aliasing=True)

    # --------------------------------------------------------------------------------------------------------------
    # PART 3 : make and train our model #
    # --------------------------------------------------------------------------------------------------------------
    # 1) initialize our model class
    my_model = YourModel(content_image=content_image, style_image=style_image)
    # 2) train the model calling the helper
    train(my_model)

    # --------------------------------------------------------------------------------------------------------------
    # PART 4 : save and show result from final epoch #
    # --------------------------------------------------------------------------------------------------------------
    # model.x is the most recent image created by the model
    result_tensor = my_model.x
    # save image in output folder
    final_image = save_tensor_as_image(result_tensor, 'out/results/', ARGS.savefile)

    # show the final result :)
    plt.imshow(final_image)
    plt.show()


ARGS = parse_args()
main()
