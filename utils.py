import numpy as np
from tqdm import tqdm
import logging


def computeSimilarityMatrix(queryLabels, baseLabels):
    groundTruthSimilarityMatrix = np.zeros((queryLabels.shape[0], baseLabels.shape[0]))
    for i in tqdm(range(queryLabels.shape[0]), "Compute SimilarityMatrix of gt"):
        curQue = queryLabels[i][:]
        assert sum(curQue) != 0
        threshold = 1
        sim = np.sum(np.logical_and(curQue, baseLabels), axis=-1)
        groundTruthSimilarityMatrix[i][np.where(sim >= threshold)[0]] = 1
    np.save("data/dataname/GroundTruthSimilarityMatrix.npy", groundTruthSimilarityMatrix)


def calcHammingRank(queryHashes, baseHashes, space='Hamming'):
    hammingDist = np.zeros((queryHashes.shape[0], baseHashes.shape[0]))
    hammingRank = np.zeros((queryHashes.shape[0], baseHashes.shape[0]))
    # queryHashes and baseHashes are Tensor on cuda
    queryHashes = queryHashes.cpu().numpy()
    baseHashes = baseHashes.cpu().numpy()
    for i in range(queryHashes.shape[0]):
        hammingDist[i] = np.reshape(np.sum(np.abs(queryHashes[i] - baseHashes), axis=1), (baseHashes.shape[0], ))
        hammingRank[i] = np.argsort(hammingDist[i])
    return hammingDist, hammingRank


def calcMAP(groundTruthSimilarityMatrix, hammingRank, top):
    [Q, N] = hammingRank.shape
    pos = np.arange(1, 1 + N)
    MAP = 0
    numSucc = 0
    for i in range(Q):
        ngb = groundTruthSimilarityMatrix[i, np.asarray(hammingRank[i, :], dtype='int32')]
        ngb = ngb[0:N]
        nRel = np.sum(ngb[:top])
        if nRel > 0:
            prec = np.divide(np.cumsum(ngb), pos)
            rec = np.array(np.cumsum(ngb) / float(np.sum(groundTruthSimilarityMatrix[i])), dtype='float32')
            if i == 0:
                precisions = prec
                recalls = rec
            else:
                precisions = precisions + prec
                recalls = recalls + rec
            prec = prec[0:top]
            ngb = ngb[0:top]
            ap = np.mean(prec[np.asarray(ngb, dtype='bool')])  # 求AP
            MAP = MAP + ap
            numSucc = numSucc + 1
    precisions = precisions / numSucc
    recalls = recalls / numSucc
    MAP = float(MAP)/numSucc
    return MAP, precisions, recalls


def getMAP(queryHashes, baseHashes, groundTruthSimilarityMatrix, space='Hamming', top=5000):
    hammingDist, hammingRank = calcHammingRank(queryHashes, baseHashes, space)
    MAP, precisions, recalls = calcMAP(groundTruthSimilarityMatrix, hammingRank, top)
    return MAP, precisions, recalls


def write_log(args):
    filepath = 'checkpoints/{}/{}/log/{}_{}_{}bit_{}_{}.txt'\
        .format(args.net_name, args.dataname, args.net_name, args.backbone, str(args.nbits), str(args.lr), args.dataname)
    logging.basicConfig(
        filename=filepath,
        level=logging.INFO,
    )
    return logging