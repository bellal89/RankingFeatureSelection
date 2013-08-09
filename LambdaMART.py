def parse(line):
	parts = line.split('#')[0].split(' ')
	rel = int(parts[0])
	qid = int(parts[1].split(':')[1])
	return FeatureVector(rel, qid, parts[2:])

class FeatureVector:
	def __init__(self, rel, qid, features):
		self.rel = rel
		self.qid = qid
		self.features = features



features_file = open("BuhFeatureVectors.txt", 'r')
feature_lines = features_file.readlines()

feature_vectors = []
for line in feature_lines:
	feature_vectors.append(parse(line))


weight_vector = [0]*len(feature_vectors[0].features)

