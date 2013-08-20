#!/usr/bin/python
# encoding: utf-8

class FeatureVector:

    def __init__(self, rel, qid, feature_strs):
        self.rel = rel
        self.qid = qid
        self.max_feature_id = 0
        self.features = {}
        for feature in feature_strs:
            f_parts = feature.split(':')
            feature_id = int(f_parts[0])
            self.features[feature_id] = float(f_parts[1])
            if(feature_id > self.max_feature_id):
                self.max_feature_id = feature_id

    def get_feature(self, feature_id):
        if (feature_id in self.features):
            return self.features[feature_id]
        return 0.0
