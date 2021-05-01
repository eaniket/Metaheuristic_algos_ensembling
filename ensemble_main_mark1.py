from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from simulated_annealing.optimize import SimulatedAnneal
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from genetic_algorithm import genetic_algorithm
from tqdm import tqdm
import pso_simple

from copy import deepcopy
# from abc import ABC
from objective_function import SumOfSquaredErrors
from artificial_bee import ArtificialBee
from employee_bee import EmployeeBee
from onlooker_bee import OnLookerBee
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from metaheuristics.bat import BatAlgorithm
from metaheuristics.bees import BeesAlgorithm
from metaheuristics.firefly import FireflyAlgorithm
from sklearn.decomposition import PCA

def simulated_annealing():
	print("Simmulated Annealing called")
	# Load the Iris data set
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	# Split the data into test and train sets                         
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
	# This is the hyperparameter space we'll be searching over
	svc_params = {'C':np.logspace(-8, 10, 19, base=2),
	              'fit_intercept':[True, False]
	             }

	# Using a linear SVM classifier             
	clf = svm.LinearSVC(max_iter = 1000, dual = False)
	# Initialize Simulated Annealing and fit
	sa = SimulatedAnneal(clf, svc_params, T=10.0, T_min=0.001, alpha=0.75,
	                         verbose=True, max_iter=1000, n_trans=5, max_runtime=300,
	                         cv=3, scoring='f1_macro', refit=True)
	sa.fit(X_train, y_train)

	# Print the best score and the best params
	# print("Score : ")
	# print(sa.best_score_, sa.best_params_)
	# Use the best estimator to predict classes
	optimized_clf = sa.best_estimator_
	y_test_pred = optimized_clf.predict(X_test)
	print(y_test_pred)
	print("Classification report : ")
	# Print a report of precision, recall, f1_score
	print(classification_report(y_test, y_test_pred))


	print("Sklearn metrics")
	result = {}
	result["Simmulated Annealing"] = accuracy_score(y_test, y_test_pred)
	print(accuracy_score(y_test, y_test_pred))

	return result
	# for i in range(len(y_test)):
	# 	print(str(y_test[i]) +" "+ str(y_test_pred[i]))


def normalize_dataset(dataset):
	# Normalize the dataset to [0, 1]
	min_arr = np.amin(dataset, axis=0)
	return (dataset - min_arr) / (np.amax(dataset, axis=0) - min_arr)


def evaluate_new_fuzzy_system(w1, w2, w3, w4, data, target):

	universe = np.linspace(0, 1, 100)

	x = []
	for w in [w1, w2, w3, w4]:
		x.append({'s': fuzz.trimf(universe, [0.0, 0.0, w]),
		          'm': fuzz.trimf(universe, [0.0, w, 1.0]),
			      'l': fuzz.trimf(universe, [w, 1.0, 1.0])})

	x_memb = []
	for i in range(4):
		x_memb.append({})
		for t in ['s', 'm', 'l']:
			x_memb[i][t] = fuzz.interp_membership(universe, x[i][t], data[:, i])

	is_setosa = np.fmin(np.fmax(x_memb[2]['s'], x_memb[2]['m']), x_memb[3]['s'])
	is_versicolor = np.fmax(np.fmin(np.fmin(np.fmin(np.fmax(x_memb[0]['s'], x_memb[0]['l']), np.fmax(x_memb[1]['m'], x_memb[1]['l'])), np.fmax(x_memb[2]['m'], x_memb[2]['l'])),x_memb[3]['m']), np.fmin(x_memb[0]['m'], np.fmin(np.fmin(np.fmax(x_memb[1]['s'], x_memb[1]['m']),x_memb[2]['s']), x_memb[3]['l'])))
	is_virginica = np.fmin(np.fmin(np.fmax(x_memb[1]['s'], x_memb[1]['m']), x_memb[2]['l']), x_memb[3]['l'])

	result = np.argmax([is_setosa, is_versicolor, is_virginica], axis=0)

	return (result == target).mean()


def pso_ga():
	
	iris = datasets.load_iris()
	normalized_iris = normalize_dataset(iris.data)
	n_features = normalized_iris.shape[1]

	fitness = lambda w: 1.0 - evaluate_new_fuzzy_system(w[0], w[1], w[2], w[3], normalized_iris, iris.target)

	# Test Fuzzy
	# w = [0.07, 0.34, 0.48, 0.26] # 95%
	# w = [0, 0.21664307088134033, 0.445098590128248, 0.2350617110613577] # 96.6%
	# print(1.0 - fitness(w))

	record = {'GA': [], 'PSO': []}

	for _ in tqdm(range(30)):

		# GA
		best, fbest = genetic_algorithm(fitness_func=fitness, dim=n_features, n_individuals=10, epochs=30, verbose=False)
		record['GA'].append(1.0 - fbest)

		# PSO
		initial=[0.5, 0.5, 0.5, 0.5]             
		bounds=[(0, 1), (0, 1), (0, 1), (0, 1)] 
		best, fbest = pso_simple.minimize(fitness, initial, bounds, num_particles=10, maxiter=30, verbose=False)
		record['PSO'].append(1.0 - fbest)


	#Statistcs about the runs
	# print('GA:')
	# print(np.amax(record['GA']), np.amin(record['GA']))
	# print(np.mean(record['GA']), np.std(record['GA']))

	# print('PSO:')
	# print(np.amax(record['PSO']), np.amin(record['PSO']))
	# print(np.mean(record['PSO']), np.std(record['PSO']))

	# print(record)
	# fig, ax = plt.subplots(figsize=(5, 4))

	# ax.boxplot(list(record.values()), vert=True, patch_artist=True, labels=list(record.keys())) 

	# ax.set_xlabel('Algorithm')
	# ax.set_ylabel('Accuracy')

	# plt.tight_layout()
	# plt.show()
	print(record)
	result = {}
	result['GA'] = np.mean(record['GA'])
	result['PSO'] = np.mean(record['PSO'])
	return result





class ABC(object):

    def __init__(self, obj_function, colony_size=30, n_iter=5000, max_trials=100):
        self.colony_size = colony_size
        self.obj_function = obj_function

        self.n_iter = n_iter
        self.max_trials = max_trials

        self.optimal_solution = None
        self.optimality_tracking = []

    def __reset_algorithm(self):
        self.optimal_solution = None
        self.optimality_tracking = []

    def __update_optimality_tracking(self):
        self.optimality_tracking.append(self.optimal_solution.fitness)

    def __update_optimal_solution(self):
        n_optimal_solution = \
            min(self.onlokeer_bees + self.employee_bees,
                key=lambda bee: bee.fitness)
        if not self.optimal_solution:
            self.optimal_solution = deepcopy(n_optimal_solution)
        else:
            if n_optimal_solution.fitness < self.optimal_solution.fitness:
                self.optimal_solution = deepcopy(n_optimal_solution)

    def __initialize_employees(self):
        self.employee_bees = []
        #print(str(self.obj_function)+"********")
        for itr in range(self.colony_size // 2):
            self.employee_bees.append(EmployeeBee(self.obj_function))

    def __initialize_onlookers(self):
        self.onlokeer_bees = []
        for itr in range(self.colony_size // 2):
            self.onlokeer_bees.append(OnLookerBee(self.obj_function))

    def __employee_bees_phase(self):
        map(lambda bee: bee.explore(self.max_trials), self.employee_bees)

    def __calculate_probabilities(self):
        sum_fitness = sum(map(lambda bee: bee.get_fitness(), self.employee_bees))
        map(lambda bee: bee.compute_prob(sum_fitness), self.employee_bees)

    def __select_best_food_sources(self):
        self.best_food_sources =\
            filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1),
                   self.employee_bees)
        while not self.best_food_sources:
            self.best_food_sources = \
                filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1),
                       self.employee_bees)

    def __onlooker_bees_phase(self):
        map(lambda bee: bee.onlook(self.best_food_sources, self.max_trials),
            self.onlokeer_bees)

    def __scout_bees_phase(self):
        map(lambda bee: bee.reset_bee(self.max_trials),
            self.onlokeer_bees + self.employee_bees)

    def optimize(self):
        self.__reset_algorithm()
        self.__initialize_employees()
        self.__initialize_onlookers()
        for itr in range(self.n_iter):
            self.__employee_bees_phase()
            self.__update_optimal_solution()

            self.__calculate_probabilities()
            self.__select_best_food_sources()

            self.__onlooker_bees_phase()
            self.__scout_bees_phase()

            self.__update_optimal_solution()
            self.__update_optimality_tracking()
            # print("iter: {} = cost: {}"
            #       .format(itr, "%04.03e" % self.optimal_solution.fitness))

def decode_centroids(centroids, n_clusters, data):
    return centroids.reshape(n_clusters, data.shape[1])
  

def assign_centroid(centroids, point):
	# print("centroids")
	# print(centroids)
	# print("data")
	# print(point)
	distances = [np.linalg.norm(point - centroids[idx]) for idx in centroids]
	return np.argmin(distances)

def abc():
	data = MinMaxScaler().fit_transform(load_iris()['data'][:, [1,3]])
	# plt.figure(figsize=(9,8))
	# plt.scatter(data[:,0], data[:,1], s=50, edgecolor='w', alpha=0.5)
	# plt.title('Original Data')
	# plt.show()

	colors = ['r', 'g', 'y']
	target = load_iris()['target']

	#plt.figure(figsize=(9,8))
	#print(target)
	# for instance, tgt in zip(data, target):
	#     plt.scatter(instance[0], instance[1], s=50,
	#                 edgecolor='w', alpha=0.5, color=colors[tgt])
	# plt.title('Original Groups')
	# plt.show()

	objective_function = SumOfSquaredErrors(dim=6, n_clusters=3, data=data)
	optimizer = ABC(obj_function=objective_function, colony_size=30,
	                n_iter=300, max_trials=100)
	optimizer.optimize()

	centroids = dict(enumerate(decode_centroids(optimizer.optimal_solution.pos,
                                            n_clusters=3, data=data)))

	custom_tgt = []
	for instance in data:
	    custom_tgt.append(assign_centroid(centroids, instance))
	print("Custom Target: ")
	print(len(custom_tgt))
	print(custom_tgt)
	y_data = []
	for i in target:
		y_data.append(i)

	result = {}
	result["ABC"] = accuracy_score(y_data, custom_tgt)
	return result
	# colors = ['r', 'g', 'y']
	# plt.figure(figsize=(9,8))
	# for instance, tgt in zip(data, custom_tgt):
	#     plt.scatter(instance[0], instance[1], s=50, edgecolor='w',
	#                 alpha=0.5, color=colors[tgt])

	# for centroid in centroids:
	#     plt.scatter(centroids[centroid][0], centroids[centroid][1],
	#                 color='k', marker='x', lw=5, s=500)
	# plt.title('Partitioned Data found by ABC')
	# plt.show()


def create_cluster_loss(X, k):
    def cluster_loss(x):
        centers = np.split(x, k)
        dists = np.zeros((len(centers), len(X)))
        for i in range(len(centers)):
            dists[i] = np.sqrt(np.sum(np.square(X-centers[i]), axis=1))
        return np.sum(np.min(dists, axis=0))
    
    return cluster_loss


def plot_clustered_data(X, centers):
    dists = np.zeros((len(centers), len(X)))
    for i in range(len(centers)):
        dists[i] = np.sqrt(np.sum(np.square(X-centers[i]), axis=1))
    respons = np.argmin(dists, axis=0)
    if X.shape[1] > 2:
        pca = PCA(2).fit_transform(np.concatenate([X, centers], axis=0))
        
    X = pca[:-len(centers)]
    centers = pca[-len(centers):]
    centroids = dict(enumerate(centers))
    final_res = []
    # print(len(X))
    # print(X)
    # print(len(centroids))
    # print(centroids)
    for point in X:
    	distances = [np.linalg.norm(point - centroids[idx]) for idx in centroids]
    	final_res.append(np.argmin(distances))

    print("Printing from algo")
    print(final_res)



def bee_bat_firefly():
	iris_data = load_iris()['data']
	iris_labels = load_iris()['target']

	# plt.title('ground truth iris data set')
	iris_data_pca = PCA(2).fit_transform(iris_data)
	# plt.scatter(iris_data_pca.T[0], iris_data_pca.T[1], c=iris_labels)
	# plt.show()

	true_centers = np.array([np.mean(iris_data[np.where(iris_labels == label)], axis=(0)) for label in set(iris_labels)])
	# plot_clustered_data(iris_data, true_centers)

	objective = 'min'
	n = iris_data.shape[0]
	k = 3
	d_iris = iris_data.shape[1] * k # we concatenate all k cluster centers to one vector, i.e. k times 4 dimensions in iris data set
	range_min  = -5.0
	range_max = 5.0
	T = 200

	iris_loss = create_cluster_loss(iris_data, k=k)
	iris_loss(true_centers)

	bees = BeesAlgorithm(d=d_iris, n=n, range_min=range_min, range_max=range_max,
	                     nb=50, ne=20, nrb=5, nre=10, shrink_factor=0.8, stgn_lim=5)

	bat = BatAlgorithm(d=d_iris, n=n, range_min=range_min, range_max=range_max,
	                   a=0.5, r_max=0.5, alpha=0.9, gamma=0.9, f_min=0.0, f_max=3.0)

	firefly = FireflyAlgorithm(d=d_iris, n=n, range_min=range_min, range_max=range_max,
	                           alpha=1.0, beta_max=1.0, gamma=0.5)



	#####bees algo
	print("Bee Algorithm")
	solution_iris, latency_iris = bees.search(objective, iris_loss, T)
	solution_iris_x, solution_iris_y = solution_iris
	# print(solution_iris_x)
	# print(solution_iris_y)
	# print(solution_iris)
	# print(latency_iris)
	#bees.plot_history()

	centers_iris = np.split(solution_iris_x, k)
	# print(centers_iris)
	plot_clustered_data(iris_data, centers_iris)


	######bat algo
	print("Bat Algorithm")
	solution_iris, latency_iris = bat.search(objective, iris_loss, T)
	solution_iris_x, solution_iris_y = solution_iris
	# print(solution_iris)
	# print(latency_iris)
	#bat.plot_history()

	centers_iris = np.split(solution_iris_x, k)
	plot_clustered_data(iris_data, centers_iris)


	#####firefly
	print("Firefly Algorithm")
	solution_iris, latency_iris = firefly.search(objective, iris_loss, T)
	solution_iris_x, solution_iris_y = solution_iris
	# print(solution_iris)
	# print(latency_iris)
	#firefly.plot_history()

	centers_iris = np.split(solution_iris_x, k)
	plot_clustered_data(iris_data, centers_iris)



def main():
	final_result = []
	
	final_result.append(simulated_annealing())
	
	temp = pso_ga()
	for i in temp:
		temp_dict = {}
		temp_dict[i] = temp[i]
		#print(temp_dict)
		final_result.append(temp_dict)

	final_result.append(abc())

	acc = 0.0
	model_name = ""
	for i in range(len(final_result)):
		print(final_result[i])
		for key in final_result[i]:
			# print(str(key) +": "+ str(final_result[i][key]))
			if final_result[i][key]>acc:
				acc = float(final_result[i][key])
				model_name = str(key)


	final_result.append(bee_bat_firefly())

	print("Maximum accuracy is: " + str(acc) + " with model: " + model_name)

if __name__ == '__main__':
	main()