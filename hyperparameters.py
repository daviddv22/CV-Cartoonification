"""
Number of epochs determines the number of times the entire training dataset is passed through the network.
Adjusted based on network complexity and observed training performance.
"""
num_epochs = 1000

"""
Learning rate sets the step size at the start of training. 
Adjusted based on network complexity and observed training performance.
"""
learning_rate = .002

"""
Beta_1 is the first hyperparameter for the Adam optimizer.
"""
beta_1 = .99

"""
epsilon for the Adam optimizer.
"""
epsilon = 1e-1

"""
A critical parameter for style transfer. The value for this will determine 
how much the generated image is "influenced" by the CONTENT image.
"""
alpha = .05

"""
A critical parameter for style transfer. The value for this will determine 
how much the generated image is "influenced" by the STYLE image.
"""
beta = 5
