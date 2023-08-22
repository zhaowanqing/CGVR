import time
import numpy as np
from conceptnet5 import api
from conceptnet5.api import *

# 40 relations
ALL_RELATIONS = [
    "/r/Antonym",
    "/r/AtLocation",
    "/r/CapableOf",
    "/r/Causes",
    "/r/CausesDesire",
    "/r/CreatedBy",
    "/r/DefinedAs",
    "/r/DerivedFrom",
    "/r/Desires",
    "/r/DistinctFrom",
    "/r/Entails",
    "/r/EtymologicallyDerivedFrom",
    "/r/EtymologicallyRelatedTo",
    "/r/ExternalURL",
    "/r/FormOf",
    "/r/HasA",
    "/r/HasContext",
    "/r/HasFirstSubevent",
    "/r/HasLastSubevent",
    "/r/HasPrerequisite",
    "/r/HasProperty",
    "/r/HasSubevent",
    "/r/InstanceOf",
    "/r/IsA",
    "/r/LocatedNear",
    "/r/MadeOf",
    "/r/MannerOf",
    "/r/MotivatedByGoal",
    "/r/NotCapableOf",
    "/r/NotDesires",
    "/r/NotHasProperty",
    "/r/NotUsedFor",
    "/r/ObstructedBy",
    "/r/PartOf",
    "/r/ReceivesAction",
    "/r/RelatedTo",
    "/r/SimilarTo",
    "/r/SymbolOf",
    "/r/Synonym",
    "/r/UsedFor"
    # "/r/dbpedia/capital",
    # "/r/dbpedia/field",
    # "/r/dbpedia/genre",
    # "/r/dbpedia/genus",
    # "/r/dbpedia/influencedBy",
    # "/r/dbpedia/knownFor",
    # "/r/dbpedia/language",
    # "/r/dbpedia/leader",
    # "/r/dbpedia/occupation",
    # "/r/dbpedia/product",
]


def get_Rel(node1, node2):
    res = FINDER.query({'node': '/c/en/' + node1, 'other': '/c/en/' + node2})
    rel = res[0]['rel']['@id']
    return rel


def get_rel_index():
    label_path = "data/nuswide/tags.txt"
    rel_matrix = -1 * np.ones((3139, 3139))  # for nuswide dataset
    tagsList = []
    f1 = open(label_path, encoding='utf-8')
    lines = f1.readlines()
    for line in lines:
        line = line.strip().strip('\n')
        tagsList.append(line)
    f1.close()
    print("total words length： ", len(tagsList))
    for i, w1 in enumerate(tagsList):
        for j, w2 in enumerate(tagsList):
            res = FINDER.query({'node': '/c/en/' + w1, 'other': '/c/en/' + w2})
            if(len(res) > 0):
                rel = res[0]['rel']['@id']
                if rel in ALL_RELATIONS and rel_matrix[i][j] == -1:
                    rel_matrix[i][j] = ALL_RELATIONS.index(rel)
                    rel_matrix[j][i] = ALL_RELATIONS.index(rel)
    np.save("nuswide_rel.npy", rel_matrix)


def get_rel_weight():
    label_path = "data/nuswide/tags.txt"
    rel_weight_matrix = np.zeros((3139, 3139))  # for nuswide dataset
    tagsList = []
    f1 = open(label_path, encoding='utf-8')
    lines = f1.readlines()
    for line in lines:
        line = line.strip().strip('\n')
        tagsList.append(line)
    f1.close()
    print("total words length： ", len(tagsList))
    for i, w1 in enumerate(tagsList):
        for j, w2 in enumerate(tagsList):
            res = FINDER.query({'node': '/c/en/' + w1, 'other': '/c/en/' + w2})
            if(len(res) > 0):
                rel_w = res[0]['weight']
                rel_weight_matrix[i][j] = rel_w
                rel_weight_matrix[j][i] = rel_w
    np.save("nuswide_rel_weight.npy", rel_weight_matrix)


def getScore(str1, str2):
    prefix = "/c/en/"
    node1 = prefix + str1
    node2 = prefix + str2
    score = query_relatedness(node1, node2)['value']
    return score


def get_relateness(tagListPath):
    WeightScore = np.eye(3139).astype(np.float32)  # for nuswide dataset
    tagsList = []
    f1 = open(tagListPath, encoding='utf-8')
    lines = f1.readlines()
    for line in lines:
        line = line.strip().strip('\n')
        tagsList.append(line)
    f1.close()
    print("total words length： ", len(tagsList))
    start = time.time()
    for i, word in enumerate(tagsList):
        for rel_w in tagsList[i:]:
            index = tagsList.index(rel_w)
            s = getScore(word, rel_w)
            WeightScore[i][index] = s
            WeightScore[index][i] = s
        if i+1 % 100 == 0:
            end = time.time()
            print(">>>>>>Computing {}-th word, Time:{:.2f}<<<<<<".format(i+1, end-start))
            start = time.time()
    np.save('dataset/concepnet/nuswide_relateness.npy', WeightScore)


def findRealtnessEdge(word):
    res = api.lookup_paginated("/c/en/" + word)
    edges = res["edges"]
    relateNode = []
    for i, edge in enumerate(edges):
        end = edge["end"]
        # start = edge["start"]
        if "language" in end.keys() and "@type" in end.keys() and end['language'] == "en" and end["@type"] == "Node":
            node = end["term"][6:]
            if node not in relateNode:
                relateNode.append(node)
    return relateNode


def findRealtnessAllEdge(word):
    res = lookup_grouped_by_feature("/c/en/" + word)
    # for key in res.keys():
    #     print(key)
        # @id
        # label
        # language
        # term
        # @type
        # features
        # @context
        # version
    features = res["features"]
    relatness = []
    for i, feature in enumerate(features):
        edges = feature["edges"]
        for j, edge in enumerate(edges):
            if "language" in edge["start"].keys() and "language" in edge["end"].keys() and edge["start"]["language"]\
                    == "en" and edge["end"]["language"] == "en":
                # print("{}-th, start: {}, end: {}, weight:{}".format(j, edge["start"]["term"][6:],
                #                                                     edge["end"]["term"][6:], edge["weight"],
                #                                                     ))
                if edge["start"]["term"][6:] not in relatness:
                    relatness.append(edge["start"]["term"][6:])
                if edge["end"]["term"][6:] not in relatness:
                    relatness.append(edge["end"]["term"][6:])
    return relatness