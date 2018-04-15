import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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
def soft_max(w):
    e = np.exp(w - np.max(w))
    dist = e / e.sum()
    return dist


# Predict the tag for this example.
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


# Calculate the 'real' probability.
def true_prob(number_to_prob):
    prob1 = norm(2, 1).pdf(number_to_prob)
    prob2 = norm(4, 1).pdf(number_to_prob)
    prob3 = norm(6, 1).pdf(number_to_prob)
    return prob1 / (prob1 + prob2 + prob3)


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
# Set learning rate
learningRate = 0.05

# The practice.
for i in range(0, 100):
    for example in data:
        # Predict.
        prediction = predict(example)

        # Loss calculation.
        subtractionVector = np.array(vector_create(example[1]))
        subtractionVector.shape = (3, 1)
        update = prediction - subtractionVector

        # Update.
        exampleVector = np.array([1, example[0]])
        exampleVector.shape = (1, 2)
        updateMatrix = np.matmul(update, exampleVector)
        weightMatrix = weightMatrix - (learningRate * updateMatrix)

# The test.
predictionSet = []
trueProb = []
numberList = []
for number1 in range(0, 300):
    number2 = np.random.uniform(0, 10)
    prediction = predict([number2, 0])
    predictionSet.extend(prediction[0])
    numberList.append(number2)
    trueProb.append(true_prob(number2))

# Draw the graph.
fig, ax = plt.subplots()
plt.plot(numberList, predictionSet, 'ro', label='Net prediction')
plt.plot(numberList, trueProb, 'bo', label='True probability')

# Set the legend
legend = ax.legend(loc='upper right', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
for label in legend.get_texts():
    label.set_fontsize('large')  # the legend text size
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width

# Set the graph.
plt.axis([0, 10, 0, 1])
plt.ylabel('p(y = 1 | x)')
plt.xlabel('x value')

# Show the graph.
plt.show()