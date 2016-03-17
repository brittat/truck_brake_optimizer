class Truck(object):

   def __init__(self):
      self.weight = 20000
      self.T_max = 750
      self.v_max = 25
      self.v_start = 20
      self.startingGear = 7
      self.startTb = 500
      self.slopeLength = 1000
      self.alphaMax = 10

import truck_functions as tf
import numpy as np
import matplotlib.pyplot as plt

timeStep = 0.1 #Used in numerical integration
maxGens = 3000
populationSize = 6 #Number of chromosomes
noOfGenes = 10
Nh = 3 #Number of neurons in hidden layer of NN
#The below numbers are part of the model and not to be varied:
inputNeurons = 3 
outputNeurons = 2

#Parameters for GA
tournamentSelectionParameter = 0.7
tournamentSize = 2
Pmutation = 0.03

truck = Truck()

avgFitVec = []
valFitVec = []
bestFitInGen = []
bestFitVec = []
population = tf.initializePopulation(populationSize,noOfGenes,Nh,inputNeurons,outputNeurons)
tmpBestFit = 0
bestFitEver = 0
genVec = []
fitPlotVec = []
newPlotVec = []
newXVec = []

for iGen in xrange(maxGens):
    genVec.append(iGen)
    fitness = []
    if iGen %100 == 0:
        print iGen
    for iPop in xrange(populationSize):
        [W1,W2] = tf.decodeChromosome(population[iPop],Nh,noOfGenes)
        [currentTrainFit,storedData] = tf.testChromosome(W1,W2,truck,timeStep,Nh,1)
        fitness.append(currentTrainFit)
        if currentTrainFit > tmpBestFit:
            tmpBestFit = currentTrainFit
            bestChromoInGen = np.copy(population[iPop])
            if currentTrainFit > bestFitEver:
                leaderOfThePack = np.copy(population[iPop])
                bestFitEver = currentTrainFit
                newPlotVec.append(currentTrainFit)
                newXVec.append(iGen)
    
    [W1,W2] = tf.decodeChromosome(bestChromoInGen,Nh,noOfGenes)
    [currentTrainFit,storedData] = tf.testChromosome(W1,W2,truck,timeStep,Nh,1)

    fitPlotVec.append(currentTrainFit)
    bestFitVec.append(tmpBestFit)
    population[1] = bestChromoInGen
    tempPopulation = list(population)

    for i in xrange(0,populationSize,2):
        i1 = tf.tournamentSelect(fitness,tournamentSelectionParameter,tournamentSize)
        i2 = tf.tournamentSelect(fitness,tournamentSelectionParameter,tournamentSize)
        tempPopulation[i] = np.copy(population[i1])
        tempPopulation[i+1] = np.copy(population[i2])
        
        #[firstNewChromosome, secondNewChromosome] = Cross(population(i1,:,:),population(i2,:,:));
        #tempPopulation(i,:,:) = firstNewChromosome;
        #tempPopulation(i+1,:,:) = secondNewChromosome;

    # print "first"
    # [W1,W2] = tf.decodeChromosome(bestChromoInGen,Nh,noOfGenes)
    # [currentTrainFit,storedData] = tf.testChromosome(W1,W2,truck,timeStep,Nh,1)
    # print currentTrainFit
    # print W1

    for iPop in xrange(populationSize):
        tempPopulation[iPop] = tf.mutate(tempPopulation[iPop],Pmutation)
    
    # [W1,W2] = tf.decodeChromosome(bestChromoInGen,Nh,noOfGenes)
    # [currentTrainFit,storedData] = tf.testChromosome(W1,W2,truck,timeStep,Nh,1)
    # print currentTrainFit
    # print W1

    # [currentTrainFit,storedData] = tf.testChromosome(W1,W2,truck,timeStep,Nh,1)
    # print currentTrainFit

    population = tempPopulation
    
    [W1,W2] = tf.decodeChromosome(bestChromoInGen,Nh,noOfGenes)
    [currentTrainFit,storedData] = tf.testChromosome(W1,W2,truck,timeStep,Nh,1)
    #print currentTrainFit
    [currentValFit,storedValData] = tf.testChromosome(W1,W2,truck,timeStep,Nh,2)


    valFitVec.append(currentValFit)
    bestFitVec.append(currentTrainFit)


#The storedData concerns the last slope of the data set

plt.subplot(3, 1, 1)
plt.plot(storedData[0], storedData[1], 'ko-')
plt.xlabel('x position')
plt.ylabel('y position')

plt.subplot(3, 1, 2)
plt.plot(storedData[0], storedData[2], 'r.-')
plt.xlabel('x position')
plt.ylabel('velocity')

plt.subplot(3, 1, 3)
plt.plot(newXVec, newPlotVec, 'r.-')
plt.xlabel('x position')
plt.ylabel('Fitness')

plt.show()





