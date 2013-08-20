import os
import math
from FeatureVector import FeatureVector
from RankFilter import RankFilter
import matplotlib.pyplot as plt

features_file_name = "RankLib/SolrTrain.txt"

def parse(line):
	parts = line.split('#')[0].strip().split(' ')
	rel = int(parts[0])
	qid = int(parts[1].split(':')[1])
	return FeatureVector(rel, qid, parts[2:])

def clear_folder(folder):
	for file_name in os.listdir(folder):
		full_file_name = os.path.join(folder, file_name)
		try:
			os.remove(full_file_name) 
		except:
			print("Cannot delete file " + full_file_name)

def gen_plot(ranker):
	clear_folder("RankLib/Results")
	x = []
	y = []
	y2 = []
	y3 = []
	for i in xrange(1, len(sorted_weights) + 1):
		score = get_test_ndcg(i)
		print(i)
		print(score)
		x.append(i)
		y.append(score)

	# Create the plot
	plt.plot(x, y, '-')
	plt.title('Relief-like algorithm')
	plt.xlabel('# of best features')
	plt.ylabel('NDCG@10')
	# plt.legend(str(ranker))
	# Save the figure in a separate file
	plt.savefig("RankLib/Relief_ranker_.png")

def get_test_ndcg(features_count):
	cmd = "java -jar RankLib/RankLib.jar -train " + features_file_name + " -feature RankLib/SelectedFeatureSets/" + str(features_count) + " -ranker 6 -metric2t NDCG@10 -tts 0.65 > RankLib/Results/" + str(features_count)
	os.system(cmd)
	f = open("RankLib/Results/" + str(features_count), 'r')
	ndcg = float(f.readlines()[-1].split(':')[1].strip())
	f.close()
	return ndcg

if os.path.exists(features_file_name + '.Weights'):
	weights = [float(w.strip()) for w in open(features_file_name + '.Weights', 'r').readlines() if w.strip() != '']
else:
	f_f = open(features_file_name, 'r')
	feature_lines = f_f.readlines()
	f_f.close()

	feature_vectors = []
	for line in feature_lines:
		feature_vectors.append(parse(line))
	rank = RankFilter(feature_vectors)
	
	weights = rank.filter(3)
	
	fw_f = open(features_file_name + '.Weights', 'w')
	fw_f.write("\n".join([str(w) for w in weights]))
	fw_f.close()

# print(weights)
sorted_weights = sorted([(i + 1, w) for i, w in enumerate(weights)], key=lambda t: t[1], reverse=True)

clear_folder("RankLib/SelectedFeatureSets")
for i in xrange(1, len(sorted_weights) + 1):
	f = open("RankLib/SelectedFeatureSets/"+str(i), 'w')
	for w in sorted_weights[0:i]:
		f.write(str(w[0]) + '\n')
	f.close()

ranker_list = [6]
for ranker in ranker_list:
	gen_plot(ranker)
