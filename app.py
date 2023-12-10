from cmath import cos, sin
import math
import numpy as np
import pygame
import random
import mnist as mn
import layer_def as ld
import network as net
from colorama import Fore, Back, Style

from keras.datasets import mnist

pygame.init()

#screen settings
screen = pygame.display.set_mode((561, 561))
backgroundColor = [55, 55, 55]
screen.fill(backgroundColor)

appRunning = True
delta_time = 0.0
clock = pygame.time.Clock()

#load neural network from JSON file
ld.load(mn.network)

# create input interface array
outputArray = np.zeros((28, 28))

commands = [Fore.RED + "Exit: 'quit'", "Train model: train (# of epochs)", "Generate: #"]

for cmd in commands:
    print(cmd)


def initMnist(epochs):
    global inputArray

    #load from JSON save file
    print(Fore.CYAN + "Loading network from JSON file...")
    ld.load(mn.network)
    print(Fore.CYAN + "Network loaded succesfully!")

    #train network
    mn.trainMnist(mn.network, epochs)

    #save newly trained model
    ld.save(mn.network)
    print(Fore.GREEN + "Network saved succesfully!")



#draws window to write number
def drawOutputArray():

    #loop through input array
    for x in range(28):
        for y in range(28):
            #set color according to array value
            color = (
                min(255, int(220 * outputArray[x][y] + 35)),
                min(255, int(220 * outputArray[x][y] + 35)),
                min(255, int(220 * outputArray[x][y] + 35))
            )

            #draw corresponding rectangle
            pygame.draw.rect(screen, color, pygame.Rect((y*20), (x*20), 19, 19))


#input users number into neural network, then output and return prediction
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



def update():
    pass


def draw():
    drawOutputArray()
    pass



while appRunning:

    predValue = input(Fore.YELLOW + "::: ")

    try :
        if int(predValue) <= 9 and int(predValue) >= 0:
            mnistPredict(int(predValue))
    except:
        #try for string
        try:
            if str(predValue) == "quit":
                appRunning = False

            elif str(predValue) == "train":
                initMnist(int(input(Fore.CYAN + "Epochs: ")))
            else:
                print(Fore.CYAN + "Unknown input")
        except:
            print(Fore.RED + "Invalid input")
        

    update()
    draw()

    pygame.display.flip()

    delta_time = 0.001 * clock.tick(144)


pygame.quit()