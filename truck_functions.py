from __future__ import division
import numpy as np
import math


def acceleration(truck,currentTb,activeGear,currentAlpha,pedalPressure):
#M,Tb,Tmax,currentAlpha,activeGear,Cb
#Pp is pedalpressure, one of the outputs from the NN
    Cb = 3000
    g=9.81
    brakingFactors = [7.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.6, 1.4, 1.2, 1] #Given parameters
    F_g = truck.weight*g*math.sin(math.radians(currentAlpha))

    if currentTb < truck.T_max-100:
        F_b = (truck.weight*g*pedalPressure)/20
    else:
        F_b = ((truck.weight*g*pedalPressure)/20)*math.exp(-(currentTb-(truck.T_max-100))/100)

    F_eb = brakingFactors[int(activeGear)-1]*Cb
    return (F_g-F_b-F_eb)/truck.weight


def initializePopulation(populationSize,noOfGenes,hiddenNeurons,inputNeurons,outputNeurons):
    population =[]
    for i in range(populationSize):
        individual = generateBinaryMatrix(noOfGenes,hiddenNeurons,inputNeurons,outputNeurons)
        population.append(individual)
    return population
    
def generateBinaryMatrix( noOfGenes,Nh,Ni,No ):
    noRows = (Ni + 1)*Nh + (Nh+1)*No;
    #The rows are the total number of neurons of the network
    #Total as in all the neurons of all the layers
    noCols = noOfGenes
    return np.around(np.random.rand(noRows,noCols))

def calculateFitness(velocities,distanceTravelled):
    averageVelocity = sum(velocities)/float(len(velocities))
    fitness = math.sqrt(averageVelocity)*distanceTravelled

    return fitness

def testChromosome(W1,W2,truck,timeStep,Nh,dataSet):
    beta = .5
    Tamb = 283
    Ch =40
    tau = 30
    patienceParam = 2
    iDataSet = dataSet
    if dataSet == 1:
        iSlope = 10
    else:
        iSlope = 5

    fitness = []

    for currentSlope in range(iSlope):
        timeSinceLastGearChange = patienceParam
        currentV = truck.v_start
        currentTb = truck.startTb
        currentAlpha = getSlopeAngle(0, (currentSlope+1), iDataSet)
        currentGear = truck.startingGear
        currentPositions = [0, 0]
        currentX = currentPositions[0]
        currentY = currentPositions[1]
    
        storedGears = []
        storedTemp = []
        storedVelocities = []
        storedPositions = []
        storedXPositions = []
        storedYPositions = []
    
        storedAlpha = [currentAlpha]
        storedPedalPressure = [0]
    
        storedPositions.append(currentPositions)
        storedVelocities.append(currentV)
        storedGears.append(currentGear)
        storedTemp.append(currentTb)
        storedXPositions.append(0)
        storedYPositions.append(0)

        while currentPositions[0] < truck.slopeLength:
            
            xi = []
            xi.append(currentV/truck.v_max)
            xi.append(currentAlpha/truck.alphaMax)
            xi.append(currentTb/truck.T_max)
            xi.append(-1)

            trainingV = activationFunction(beta, W1, xi)
            trainingV.append(-1)
            trainingOutput = activationFunction(beta, W2, trainingV)
            pedalPressure = math.fabs(trainingOutput[0])
            deltaGear = np.sign(trainingOutput[1])
            
            if timeSinceLastGearChange < patienceParam:
                timeSinceLastGearChange = timeSinceLastGearChange + timeStep
            else:
                currentGear = int(currentGear + deltaGear)
                timeSinceLastGearChange = 0
                if currentGear == 0:
                    currentGear = 1
                elif currentGear == 11:
                        currentGear = 10

            acc = acceleration(truck,currentTb,currentGear,currentAlpha,pedalPressure)
            newV = currentV + timeStep*acc
            newTb= getBrakeTemp(tau,Ch,pedalPressure,currentTb,Tamb,timeStep)
            
            if newV > truck.v_max or newTb>truck.T_max:
                break
            else:
                currentV = newV
                currentTb = newTb

            currentX = currentPositions[0] + currentV*timeStep*math.cos(math.radians(currentAlpha))
            currentY = currentPositions[1] - currentV*timeStep*math.sin(math.radians(currentAlpha))

            storedXPositions.append(currentX)
            storedYPositions.append(currentY)

            currentPositions[0] = currentX
            currentPositions[1] = currentY
            currentAlpha = getSlopeAngle(currentX, iSlope, iDataSet);

            storedPositions.append([currentX, currentY])
            storedVelocities.append(currentV)
            storedGears.append(currentGear)
            storedTemp.append(currentTb)
            storedAlpha.append(currentAlpha)
            storedPedalPressure.append(pedalPressure)


        fitness.append(calculateFitness(storedVelocities,currentX))

    storedData = []
    storedData.append(storedXPositions)
    storedData.append(storedYPositions)
    storedData.append(storedVelocities)
    storedData.append(storedGears)
    storedData.append(storedTemp)
    storedData.append(storedAlpha)
    storedData.append(storedPedalPressure)

    fitnessOut = sum(fitness)

    return fitnessOut,storedData

def decodeChromosome(chromosome,Nh,nGenes):

    w1 = np.zeros((4,Nh))
    w2 = np.zeros(((Nh+1),2))
    variableRange = 2

    for iRow1 in range(4):
        for iCol1 in range(Nh):
            for j in range(nGenes):
                w1[iRow1,iCol1] = w1[iRow1,iCol1] + chromosome[(iRow1+iCol1),j]*2**(-j-1)
            w1[iRow1,iCol1] = -variableRange + 2*variableRange*w1[iRow1,iCol1]/(1-2**(-nGenes))

    for iRow2 in range(Nh+1):
        for iCol2 in range(2):
            for j in range(nGenes):
                w2[iRow2,iCol2] = w2[iRow2,iCol2] + chromosome[(iRow2+iCol2),j]*2**(-j-1)
            w2[iRow2,iCol2] = -variableRange + 2*variableRange*w2[iRow2,iCol2]/(1-2**(-nGenes))

    return (w1,w2)

def tournamentSelect(fitness,tournamentSelectionParameter, tournamentSize):
    populationSize = len(fitness)
    iSelected = []

    for iRand in range(tournamentSize):
        iSelected.append(int(np.random.rand(1)*(populationSize-1)))


    while len(iSelected) != 1:

        i =int(np.random.rand(1)*(len(iSelected)-1))

        r = np.random.rand(1)

        if(r<tournamentSelectionParameter):
            if fitness[int(iSelected[i])] > fitness[int(iSelected[i+1])]:
                del iSelected[i+1]
            else:
                del iSelected[i]
        else:
            if fitness[int(iSelected[i])] > fitness[int(iSelected[i+1])]:
                del iSelected[i]
            else:
                del iSelected[i+1]

    return iSelected[0]

def mutate(chromosome,mutationProbability):
    mutatedChromosome = chromosome
    [nRows,nCols] = chromosome.shape

    for iRows in range(nRows):
        for iCols in range(nCols):
            r = np.random.rand(1)
            
            if(r<mutationProbability):
                
                swapGeneIndex1 = int(np.around(np.random.rand(1)*(nRows-1)))
                swapGeneIndex2 = int(np.around(np.random.rand(1)*(nCols-1)))
                
                while swapGeneIndex1 == iRows and swapGeneIndex2 == iCols:
                    
                    swapGeneIndex1 = int(np.around(np.random.rand(1)*(nRows-1)))
                    swapGeneIndex2 = int(np.around(np.random.rand(1)*(nCols-1)))

                tmpGene = mutatedChromosome[swapGeneIndex1,swapGeneIndex2]
                mutatedChromosome[swapGeneIndex1,swapGeneIndex2] = mutatedChromosome[iRows,iCols]
                mutatedChromosome[iRows,iCols] = tmpGene

    return mutatedChromosome
            
def activationFunction( beta,w,xi ):
    [nTmp,nOut] = w.shape
    argument = []
    g = []
    for iCount2 in range(nOut):
        tmp = 0

        for iCount in range(nTmp):
            tmp = tmp + xi[iCount]*w[iCount,iCount2]

        g.append(math.tanh(beta*tmp))

    return g

def getBrakeTemp(tau,Ch,Pp,Tb,Tamb,timeStep):

    DeltaTb = Tb-Tamb;

    if Pp < 0.01:
        changeDeltaTb = - DeltaTb/tau
    else:
        changeDeltaTb = Ch*Pp

    newDeltaTb = DeltaTb + changeDeltaTb*timeStep

    return Tamb + newDeltaTb

def getSlopeAngle(x, iSlope, iDataSet):

    if (iDataSet == 1):
        if (iSlope == 1):
            alpha = 2 + math.sin(x/100) + math.cos(math.sqrt(2)*x/50)
        elif (iSlope== 2):
            alpha = 2 + 8*(x/1000)**3;
        elif (iSlope== 3):
            alpha = 3 + 2*(math.sin(3* math.sin(x/30)/50) + math.cos(math.sqrt(2)*x/100))
        elif (iSlope== 4):
            alpha = 4 + 2*((math.sin(3 * math.cos(4 * math.sin(x)/30)/50))) + 2*(math.cos(2 + math.sqrt(2)*x/100))
        elif (iSlope== 5):
            alpha = 2 + 2*(math.cos( 2* 3 * math.sin(x/30)/50)) + 3 * (math.sin(3 * math.sin(x/80)/20))
        elif (iSlope== 6):
            alpha = 5 + 2*math.sin(x/45) + math.cos(math.sqrt(2)*x/100)
        elif (iSlope== 7):
            alpha = 4 + 4*(math.sin(3 * math.sin(x/20)/50))
        elif (iSlope== 8):
            alpha = 4 + 1*(math.sin(3 * math.cos(4 * math.sin(x/20)/10)/20))
        elif (iSlope== 9):
            alpha = 2 + 2*(math.cos( 3 * math.sin(x/70)/50) + 3)
        elif (iSlope== 10):
            alpha = 5 + 2*math.cos(3 * (math.sin(3*x/20)))

    elif (iDataSet == 2):
        if (iSlope == 1):
            alpha = 2 + 4*(x/1000)**3
        elif (iSlope== 2):
            alpha = 6 + (math.sin(x/50)) + (math.cos(math.sqrt(6)*x/70))
        elif (iSlope== 3):
            alpha = 3 + 2*(math.cos(3 * x/120)) + (math.cos(math.sqrt(2)*x/110))
        elif (iSlope== 4):
            alpha = 4 + 2*(math.sin(3* math.cos(4 * math.sin(x)/30)/110)) + 2*(math.cos(2 + math.sqrt(2)*x/100))
        elif (iSlope == 5): 
            alpha = 5 + (math.sin(x/50)) + (math.cos(math.sqrt(5)*x/50))

    return alpha





    # if (iDataSet == 1):
    #   if (iSlope == 1):
    #       alpha = 5 
    #   elif (iSlope== 2):
    #       alpha = 5 
    #   elif (iSlope== 3):
    #       alpha = 5 
    #   elif (iSlope== 4):
    #       alpha = 5
    #   elif (iSlope== 5):
    #       alpha = 2 + 2*(math.cos( 2* 3 * math.sin(x/30)/50)) + 3 * (math.sin(3 * math.sin(x/80)/20))
    #   elif (iSlope== 6):
    #       alpha = 5 + 2*(math.sin(x/45)) + math.cos(math.sqrt(2)*x/100)
    #   elif (iSlope== 7):
    #       alpha = 4 + 4*(math.sin(3 * math.sin(x/20)/50))
    #   elif (iSlope== 8):
    #       alpha = 5
    #   elif (iSlope== 9):
    #       alpha = 5 
    #   elif (iSlope== 10):
    #       alpha = 5

    # elif (iDataSet == 2):
    #   if (iSlope == 1):
    #       alpha = 2 + 4*(x/1000)**3
    #   elif (iSlope== 2):
    #       alpha = 6 + (math.sin(x/50)) + (math.cos(math.sqrt(6)*x/70))
    #   elif (iSlope== 3):
    #       alpha = 3 + 2*(math.cos(3 * x/120)) + (math.cos(math.sqrt(2)*x/110))
    #   elif (iSlope== 4):
    #       alpha = 4 + 2*(math.sin(3* math.cos(4 * math.sin(x)/30)/110)) + 2*(math.cos(2 + math.sqrt(2)*x/100))
    #   elif (iSlope == 5): 
    #       alpha = 5 + (math.sin(x/50)) + (math.cos(math.sqrt(5)*x/50))

    # return alpha



















