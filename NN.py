#!/usr/bin/python
# encoding: utf-8

from FeatureVector import FeatureVector
from kdtree import KDTree

class NN:
	def __init__(self, data):
		self.data = data
		print("Kd-tree will be constructed...")
		self.tree = KDTree.construct_from_data(data)
		print("Kd-tree construction done!")
		self.rel_levels = set([vector.rel for vector in data])

	def nearest(self, point, count):
		return self.tree.query(feature_vector=point, t=count)

	def nearest_by_rel(self, point, count, rel):
		all_rel_count = 16 * count * len(self.rel_levels)
		while True:
			nn = self.nearest(point, all_rel_count)
			rel_nn = [nbour for nbour in nn if nbour.feature_vector.rel == rel]
			print("Nearest found: " + str(len(rel_nn)) + " from " + str(count))
			if (len(nn) > len(self.data) or len(rel_nn) >= count):
				break
			all_rel_count *= 2
		return rel_nn[0:count]