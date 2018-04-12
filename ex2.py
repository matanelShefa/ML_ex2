import numpy as np
import matplotlib.pyplot as plt


# Create a normal distribution with the given parameters.
def create_normal_distribution(mean, std):
    normal_distribution = np.random.normal(mean, std, 100)  # Create the normal distribution.
    np.random.shuffle(normal_distribution)  # Shuffle the data.
    return normal_distribution


# Create a pair of (point, tag) for each list.
def create_tag_list(points_list, tag):
    for point in points_list:
        data.append([point, tag])


# The softMax function.
def soft_max(w, t):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


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
create_tag_list(normalDistribution1, 2)
create_tag_list(normalDistribution1, 3)
# Shuffle the data.
np.random.shuffle(data)

# Create the weight matrix.
weightMatrix = np.random.random([3, 2])
learningRate = 0.1

# The algorithm.
for i in range(0, 100):
    for example in data:
        # Create the example + bias vector.
        exampleVector = np.array([1, example[0]])
        exampleVector.shape = (2, 1)
        softMaxInput = np.matmul(weightMatrix, exampleVector)
        prediction = np.array(soft_max(softMaxInput, 1))
        subtractionVector = np.array(vector_create(example[1]))
        subtractionVector.shape = (3, 1)
        update = prediction - subtractionVector
        exampleVector.shape = (1, 2)
        updateMatrix = np.matmul(update, exampleVector)
        weightMatrix = weightMatrix - (learningRate * updateMatrix)

plt.plot(data)
plt.axis([0, 10, 0, 20])
plt.ylabel('some numbers')
plt.show()
