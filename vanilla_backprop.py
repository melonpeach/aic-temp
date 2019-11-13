"""
modified github.com/utkuozbulak
"""
import torch

class VanillaBackprop():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # evaluation mode
        self.model.eval()
        # hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # register hook to the first convolution layer
        first_layer = self.model.conv1
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # forward pass
        model_output = self.model(input_image)
        self.model.zero_grad()
        
        # target for backprop
        one_hot_output = torch.zeros([1, model_output.size()[-1]], dtype=torch.float)
        one_hot_output[0][target_class] = 1
        
        # backward pass
        model_output.backward(gradient=one_hot_output)
        
        # convert pytorch variable to numpy array
        # [0] to get rid of the first channel (1, channel, width, height)
        grads = self.gradients.data.numpy()[0]
        
        return grads
