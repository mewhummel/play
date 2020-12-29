#This program attempts to train a model to output blank images for all input.
#Given it only has one input example and output, it takes a large learning rate and a huge number of epochs to get even close.
#
#Best results: Learning Rate 1.0, epochs=200000 for loss of 0.009.
#Output values: ~0.05 which isn't close enough to 0.
#
#
#

import tensorflow as tf
import random
import numpy as np

class Blanka():
    def __init__(self, noise_width = 100):
        self.noise_width = noise_width
        self.input_layer = tf.keras.layers.Input(shape=(noise_width))
        self.output_layer = tf.keras.layers.Dense(784, activation='sigmoid')

        self.model = tf.keras.models.Sequential()
        self.model.add(self.input_layer)
        self.model.add(self.output_layer)
        
        self.loss_fn = tf.keras.losses.MSE
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.5,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        
        self.model.compile(optimizer=optimizer,
                           loss = self.loss_fn,
                           metrics=['MeanSquaredError'])

    def get_noise(self, width, length):
        bitmap_n = np.random.normal(0, 1, (length, width))
        return bitmap_n

    def compare_weights(self, before, after):
        kernel_before, bias_before = before[0], before[1]
        kernel_after, bias_after = after[0], after[1]
        kernel_diff = before[0] - after[0]
        bias_diff = before[1] - after[1]

        kernel_mse = (np.square(kernel_diff)).mean(axis=None)
        bias_mse = (np.square(bias_diff)).mean(axis=None)
        
        return kernel_mse, bias_mse

    
    def go(self):
        input_noise = self.get_noise(self.noise_width, 1)
        input_0 = np.zeros((1, self.noise_width))
        input_1 = np.ones((1, self.noise_width))
        predict_data = np.zeros((1, 784))

        #Select the mode: all zeros, all ones, random
        trndata = input_0
        
        init_output = self.model.predict(trndata)
        init_loss = self.loss_fn(predict_data, init_output)
        print ('Loss before training: ', init_loss)
        init_weights = self.output_layer.get_weights()
        
        self.model.fit(trndata, predict_data, epochs=100000, verbose=0)
        
        new_weights = self.output_layer.get_weights()
        weight_MSE = self.compare_weights(init_weights, new_weights)
        new_output = self.model.predict(trndata)
        new_loss = self.loss_fn(predict_data, new_output)
        print ('Loss now: ', new_loss)

        test_input = self.get_noise(self.noise_width, 1)
        test_img = self.model.predict(test_input)
        print (test_img)
        

if __name__ == '__main__':
    generator = Blanka()
    generator.go()
