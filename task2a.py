import numpy as np
import utils
import typing
np.random.seed(5)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    

        
    #task 2a, creating and array of 1's, and then appending it to the data #columnwise (axis = 1)
    b = np.array(np.ones(X.shape[0], dtype=float))
    X = np.concatenate( (X, np.array(b)[:,None]), axis=1 )  
    #normalize the images:
    mean = np.mean(X)
    stDev = np.std(X)
    newX = (X-mean)/stDev
    return newX


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    
    m = outputs.shape[0]
    totalLoss = (-(np.sum(targets*np.log(outputs))))
    avgLoss = (totalLoss/m)
    
    return (avgLoss)


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer


        # for 2b, store intermediate values:
        self.outputLayerOutput = 0
        self.hiddenLayerOutput = 0
        #making it general:
        self.hiddenLayerOutputs = [0 for i in range(len(self.neurons_per_layer))]
        
        # Initialize the weights randomly
        self.ws = []
        prev = self.I
        if (self.use_improved_weight_init):
            print("using improved weight init:")
            mu = 0
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                print("Initializing weight to shape:", w_shape)
                sigma = 1/np.sqrt(prev)
                w = np.random.normal(mu, sigma, w_shape)
                self.ws.append(w)
                prev = size
        else:
            print("using standard weight init:")
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                print("Initializing weight to shape:", w_shape)
                w = np.random.uniform(-1, 1, size=w_shape)
                self.ws.append(w)
                prev = size
        
        self.grads = [None for i in range(len(self.ws))]

    #activation function
    def softmax(self, z):
        return np.true_divide(np.exp(z.T), np.sum(np.exp(z), axis=1)).T
    
    def sigmoid(self, z):
        return np.true_divide(1, (1+np.exp(-z)))

    def improvedSigmoid(self, z):
        return 1.7159*np.tanh(2*z/3) 

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        
        # implementation for task 2 and 3
        
        #if (self.use_improved_sigmoid):
        #    a2 = self.improvedSigmoid(X.dot(self.ws[0]))
        #else:
        #    a2 = self.sigmoid(X.dot(self.ws[0]))
        #yHat = self.softmax(a2.dot(self.ws[1])) 
        #self.hiddenLayerOutput = a2
        #self.outputLayerOutput = yHat
        #return yHat
        
        

        #generelized implementation for task 4:
        #iterate over all layers and do tanh, then do softmax on last layer:
        a = X
        for layer in range(len(self.neurons_per_layer)-1):
            a = self.improvedSigmoid(a.dot(self.ws[layer])) 
            self.hiddenLayerOutputs[layer] = a
        #last layer forward we use softmax
        yHat = self.softmax(a.dot(self.ws[-1]))
        self.hiddenLayerOutputs[-1] = yHat
        return yHat
        



    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        """
        
        #implementation for task 2 and 3
        #a2 = self.hiddenLayerOutput
        #yHat = self.outputLayerOutput
        #self.grads[1] = np.dot(a2.T, (outputs-targets))/a2.shape[0] 
        #only thing that changes is the final part we multiply (not dot) at the end, which is the derivative of the #activation function
        #if (self.use_improved_weight_init):
        #    deltaJ = np.dot( (outputs-targets), self.ws[1].T)*(1-np.power(a2, 2)) 
        #else:
        #    deltaJ = np.dot( (outputs-targets), self.ws[1].T)*(a2*(1-a2)) 
        #self.grads[0] = np.dot(X.T, deltaJ)/X.shape[0]
  
        
        #general implementation for task 4:
        deltaK = (outputs-targets)
        deltaJ = deltaK
        #last layer:
        secondLastLayerOutput = self.hiddenLayerOutputs[-2] # -2 since i have yHat in last position in list
        self.grads[-1] = np.dot(secondLastLayerOutput.T, deltaK)/outputs.shape[0] 
        #backprop from second last layer, to first layer
        #huske if improved weight
        for layer in range(len(self.neurons_per_layer)-2, -1, -1):
            a = self.hiddenLayerOutputs[layer]
            deltaJ = np.dot(deltaJ, self.ws[layer+1].T)*(2.0/3.0) * (1.7159 - (1 / 1.7159) * np.square(a))
            if layer > 0:
                prevLayer = self.hiddenLayerOutputs[layer-1]
                self.grads[layer] = np.dot(prevLayer.T, deltaJ)/outputs.shape[0] #delte på prevLayer
            else:
                self.grads[layer] = np.dot(X.T, deltaJ)/outputs.shape[0]
       
        
        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
            f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."


    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    m = Y.shape[0]
    Y = Y.reshape(m,)
    newY = np.zeros((m, num_classes))
    newY[np.arange(m), Y] = 1
    return newY


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True #endra denne, var False
    use_improved_weight_init = True #endra denne, var False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
