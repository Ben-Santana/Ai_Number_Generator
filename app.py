from cmath import cos, sin
import math
import numpy as np
import random
import layer_def as ld
import network as net
from colorama import Fore, Back, Style

import mnist as mn
from keras.datasets import mnist

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
mpl.use('TkAgg')

#loads neural network from JSON file
ld.load(mn.network)

# creates input interface array
outputArray = np.zeros((28, 28))

# prints available commands
commands = [Fore.RED + "Exit: 'quit'", "Train model: 'train' (# of epochs)", "Generate: Type #"]
for cmd in commands:
    print(cmd)



# sets plot style for grid
plt.style.use('_mpl-gallery-nogrid')


# enables interactive mode for realtime updates
plt.ion()

# creates plot and sets size to 7 x 7 inches
fig, ax = plt.subplots()
fig.set_size_inches(7, 7)

# TODO: Make radio buttons fit better on plot screen, and remove ticks from graph
# rax = plt.axes([0.05, 0.7, 0.15, 0.2])
# radio = RadioButtons(rax, ('Dark Mode', 'Light Mode'))




def initMnist(epochs):
    global inputArray

    #loads from JSON save file
    print(Fore.CYAN + "Loading network from JSON file...")
    ld.load(mn.network)
    print(Fore.CYAN + "Network loaded succesfully!")

    #trains network
    mn.trainMnist(mn.network, epochs)

    #saves newly trained model
    ld.save(mn.network)
    print(Fore.GREEN + "Network saved succesfully!")


#inputs users number into neural network, then output and return prediction
def mnistPredict(value):
    global outputArray
    
    # Ensure outputArray is a NumPy array and has the correct shape
    outputArray = outputArray.reshape(784, 1)
    outputArray = outputArray.astype("float32")

    # create array based on input value, (ie: 3, [0, 0, 0, 1, 0, 0 ...])
    inputArray = np.zeros((10, 1))
    inputArray[value] = 1

    #get prediction array, most probable number and certainty from neural network
    prediction = net.predict(mn.network, inputArray)

    outputArray = prediction

    #reset to original shape
    outputArray = outputArray.reshape(28, 28)

    return prediction

#TODO: call set Style when radio button is clicked and change plot style accordingly
# def setStyle(label):
#     pass
#
#radio.on_clicked(setStyle)

appRunning = True
while appRunning:

    predValue = input(Fore.YELLOW + "::: ")

    try :
        if int(predValue) <= 9 and int(predValue) >= 0:
            mnistPredict(int(predValue))

            print(Fore.GREEN + "Updating plot...")

            ax.imshow(outputArray)
            fig.canvas.draw()
            fig.canvas.flush_events()

            print(Fore.GREEN + "Plot updated!" + Fore.RESET)

    except:
        #tries for string
        try:
            if str(predValue) == "quit":
                appRunning = False

            elif str(predValue) == "train":
                initMnist(int(input(Fore.CYAN + "Epochs: ")))

            else:
                print(Fore.CYAN + "Unknown input")

        except:
            print(Fore.RED + "Invalid input")
        