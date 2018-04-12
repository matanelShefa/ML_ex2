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


normalDistribution1 = create_normal_distribution(2, 1)
normalDistribution2 = create_normal_distribution(4, 1)
normalDistribution3 = create_normal_distribution(6, 1)

data = []
create_tag_list(normalDistribution1, 1)
create_tag_list(normalDistribution1, 2)
create_tag_list(normalDistribution1, 3)

# Shuffle the data.
np.random.shuffle(data)

print data


##############################################################################
## count, bins, ignored = plt.hist(normalDistribution6, 30, normed=True)    ##
## plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *                          ##
## np.exp( - (bins - mu)**2 / (2 * sigma**2) ),                             ##
## linewidth=2, color='r')                                                  ##
## plt.show()                                                               ##
##############################################################################
