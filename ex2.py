import numpy as np
import matplotlib.pyplot as plt


# Create a normal distribution with the given parameters.
def create_normal_distribution(mean, std):
    normal_distribution = np.random.normal(mean, std, 100)  # Create the normal distribution.
    np.random.shuffle(normal_distribution)  # Shuffle the data.
    return normal_distribution


def create_tag_list(points_list, tag):
    tag_list = []
    for point in points_list:
        tag_list.append([point, tag])
    return tag_list


normalDistribution1 = create_normal_distribution(2, 1)
normalDistribution2 = create_normal_distribution(4, 1)
normalDistribution3 = create_normal_distribution(6, 1)

data = []
data.append(create_tag_list(normalDistribution1, 1))
data.append(create_tag_list(normalDistribution1, 2))
data.append(create_tag_list(normalDistribution1, 3))

print data
##############################################################################
## count, bins, ignored = plt.hist(normalDistribution6, 30, normed=True)    ##
## plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *                          ##
## np.exp( - (bins - mu)**2 / (2 * sigma**2) ),                             ##
## linewidth=2, color='r')                                                  ##
## plt.show()                                                               ##
##############################################################################
