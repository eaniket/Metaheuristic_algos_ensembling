import numpy as np

from scipy import optimize
from deap.benchmarks import schwefel

from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass


@add_metaclass(ABCMeta)

class ObjectiveFunction(object):

    def __init__(self, name, dim, minf, maxf):
        self.name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf

    def sample(self):
        return np.random.uniform(low=self.minf, high=self.maxf, size=self.dim)

    def custom_sample(self):
        return np.repeat(self.minf, repeats=self.dim) \
               + np.random.uniform(low=0, high=1, size=self.dim) *\
               np.repeat(self.maxf - self.minf, repeats=self.dim)

    @abstractmethod
    def evaluate(self, x):
        pass

class PartitionalClusteringObjectiveFunction(ObjectiveFunction):

    def __init__(self, dim, n_clusters, data):
        super(PartitionalClusteringObjectiveFunction, self)\
            .__init__('PartitionalClusteringObjectiveFunction', dim, 0.0, 1.0)
        self.n_clusters = n_clusters
        self.centroids = {}
        self.data = data

    def decode(self, x):
        centroids = x
        # .reshape(self.n_clusters, self.dim)
        self.centroids = dict(enumerate(centroids))

    @abstractmethod
    def evaluate(self, x):
        pass
     
class SumOfSquaredErrors(PartitionalClusteringObjectiveFunction):

    def __init__(self, dim, n_clusters, data):
        super(SumOfSquaredErrors, self).__init__(dim, n_clusters, data)
        self.name = 'SumOfSquaredErrors'

    def evaluate(self, x):
        # print("X:********", end=", ")
        # print(x)
        self.decode(x)

        clusters = {key: [] for key in self.centroids.keys()}
        for instance in self.data:
            distances = [np.linalg.norm(self.centroids[idx] - instance)
                         for idx in self.centroids]
            clusters[np.argmin(distances)].append(instance)

        sum_of_squared_errors = 0.0
        for idx in self.centroids:
            distances = [np.linalg.norm(self.centroids[idx] - instance)
                         for instance in clusters[idx]]
            sum_of_squared_errors += sum(np.power(distances, 2))
        return sum_of_squared_errors
class Sphere(ObjectiveFunction):

    def __init__(self, dim):
        super(Sphere, self).__init__('Sphere', dim, -100.0, 100.0)

    def evaluate(self, x):
        return sum(np.power(x, 2))


class Rosenbrock(ObjectiveFunction):

    def __init__(self, dim):
        super(Rosenbrock, self).__init__('Rosenbrock', dim, -30.0, 30.0)

    def evaluate(self, x):
        return optimize.rosen(x)


class Rastrigin(ObjectiveFunction):

    def __init__(self, dim):
        super(Rastrigin, self).__init__('Rastrigin', dim, -5.12, 5.12)

    def evaluate(self, x):
        return 10 * len(x)\
               + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * np.array(x)))


class Schwefel(ObjectiveFunction):

    def __init__(self, dim):
        super(Schwefel, self).__init__('Schwefel', dim, -500.0, 500.0)

    def evaluate(self, x):
        return schwefel(x)[0]