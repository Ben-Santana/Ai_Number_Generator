import numpy as np


# forward propogate through the neural network to obtain prediction
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

# train through back propogation
def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):

    #initialize progress bar
    printProgressBar(0, epochs, prefix = "Progress:", suffix = "Complete", length = 50)

    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward propagation
            output = predict(network, x)

            # calculate loss
            error += loss(y, output)

            # backward propagation
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)

        #update progress bar
        printProgressBar(e, epochs, prefix = "Progress:", suffix = "Complete", length = 50)

    #finilize progress bar
    printProgressBar(epochs, epochs, prefix = "Progress:", suffix = "Complete", length = 50)
    print("Error = " + str(error))


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
