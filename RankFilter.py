#!/usr/bin/python
# encoding: utf-8

import math
from FeatureVector import FeatureVector
from NN import NN

class PreCalc:
	def __init__(self, feature_vectors):
		probabilities = {}
		for vector in feature_vectors:
			if vector.rel not in probabilities:
				probabilities[vector.rel] = 0
			probabilities[vector.rel] += 1
		self.probabilities = {k : (float(v) / len(feature_vectors)) for (k, v) in probabilities.items()}

		# distances = [[]] * len(feature_vectors)
		# for i in xrange(0, len(feature_vectors)):
		# 	distances[i] = [0] * i
		# 	for j in xrange(0, i):
		# 		distances[i][j] = euclidean_dist(feature_vectors[i].features, feature_vectors[j].features)
		# self.distances = distances


class RankFilter:
	def __init__(self, feature_vectors):
		self.feature_vectors = feature_vectors
		self.pre_calc = PreCalc(feature_vectors)
		self.NNTree = NN(feature_vectors)
		self.features_count = self.NNTree.tree.features_count

	def filter(self, k):
		weights = [0] * self.features_count

		for i, vector in enumerate(self.feature_vectors):
			print(i)
			nearest = {pr[0]:self.NNTree.nearest_by_rel(vector, k, pr[0]) for pr in self.pre_calc.probabilities.items()}
			print(str(i) + "-th nearest found.")
			for i in xrange(0, len(weights)):
				weights[i] += self.gather_misses(vector, nearest, i)
				weights[i] -= self.gather_nearest(vector, nearest, i, vector.rel)
		return weights

	def gather_nearest(self, vector, nearest, feature_id, rel):
		nn = nearest[rel]
		s = 0
		for hit in nn:
			s += self.abs_dist(vector.get_feature(feature_id), hit.feature_vector.get_feature(feature_id))
		return float(s) / len(nn)

	def gather_misses(self, vector, nearest, feature_id):
		s = 0
		for kv in nearest.items():
			rel = kv[0]
			miss = kv[1]
			if rel == vector.rel:
				continue
			n_dist = self.gather_nearest(vector, nearest, feature_id, rel)
			rel_diff = self.rel_dist(vector.rel, rel)
			p_dist = self.pre_calc.probabilities[rel] / (1 - self.pre_calc.probabilities[vector.rel])
			s += n_dist * rel_diff * p_dist
		return s

	def rel_dist(self, x, y):
		return math.log(1 + math.fabs(x - y))

	def abs_dist(self, x, y):
		return math.fabs(x - y)