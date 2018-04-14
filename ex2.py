import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Create a normal distribution with the given parameters.
def create_normal_distribution(mean, std):
    normal_distribution = np.random.normal(mean, std, 1000)  # Create the normal distribution.
    np.random.shuffle(normal_distribution)  # Shuffle the data.
    return normal_distribution


# Create a pair of (point, tag) for each list.
def create_tag_list(points_list, tag):
    for point in points_list:
        data.append([point, tag])


# The softMax function.
def soft_max(w):
    #e = np.exp(w)
    e = np.exp(w - np.max(w))
    dist = e / e.sum()
    return dist


# Predict the probability tag for this example.
def predict(example_to_predict):
    # Create the example + bias vector.
    example_vector = np.array([1, example_to_predict[0]])
    example_vector.shape = (2, 1)
    # Predict.
    soft_max_input = np.matmul(weightMatrix, example_vector)
    return np.array(soft_max(soft_max_input))


# Create a binary vector for the update calculation.
def vector_create(number):
    vector = []
    for i in range(1, 4):
        if i == number:
            vector.append(1)
        else:
            vector.append(0)
    return vector


# Create the distributions.
normalDistribution1 = create_normal_distribution(2, 1)
normalDistribution2 = create_normal_distribution(4, 1)
normalDistribution3 = create_normal_distribution(6, 1)

# Create the data set.
data = []
create_tag_list(normalDistribution1, 1)
create_tag_list(normalDistribution2, 2)
create_tag_list(normalDistribution3, 3)
# Shuffle the data.
np.random.shuffle(data)

# Create the weight matrix.
weightMatrix = np.random.random([3, 2])
learningRate = 0.1

# The practice.
for i in range(0, 100): # TODO - CHANGE TO 100?
    for example in data:
        # Predict.
        prediction = predict(example)
        #print 'AFTER PREDICT: weightMatrix =\n', weightMatrix
        # Loss calculation.
        subtractionVector = np.array(vector_create(example[1]))
        subtractionVector.shape = (3, 1)
        #print 'prediction - subtractionVector = ', prediction, ' - ', subtractionVector
        update = prediction - subtractionVector
        #print 'update = ', update
        # Update.
        exampleVector = np.array([1, example[0]])
        exampleVector.shape = (1, 2)
        #print 'exampleVector = ', exampleVector
        updateMatrix = np.matmul(update, exampleVector)
        #print 'BEFORE LR: updateMatrix =\n', updateMatrix
        #print 'AFTER LR: updateMatrix =\n', learningRate * updateMatrix
        weightMatrix = weightMatrix - (learningRate * updateMatrix)
        #print 'AFTER UPDATE: weightMatrix =\n', weightMatrix

# The test.

predictionSet = []
trueProb = []
numberList = []
#numberList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for number1 in range(0, 100):
#for number1 in numberList:
    number2 = np.random.uniform(0, 10)
    prediction = predict([number2, 0])
    if number1 % 10 == 0:
        print 'number: ', number2, ', prediction: ', prediction[0]
    predictionSet.extend(prediction[0])
    numberList.append(number2)
    #trueProb.append(norm.pdf(number1, 2, 1))

#plt.plot(trueProb)
print numberList
print "predictionSet", predictionSet
plt.plot(numberList, predictionSet, 'ro')
plt.axis([0, 10, 0, 1])
plt.ylabel('some numbers')
plt.show()