##################################################################################
# Cross-modal Guided Visual Representation Learning for Social Image Retrieval   #
#                                                                                #
#                                                                                #
# Description: This .py is used for training and test the CGVR                   #
#                                                                                #
# Note: This code is used for ICCV2023 review purposes                           #
##################################################################################

from utils import getMAP
import torch
from model.models import build_CGVR_net
from torch import optim
from tqdm import tqdm
from torch.optim import lr_scheduler
import time
import os
import logging
import numpy as np
from dataset.dataloader import get_train_loaders, get_base_loaders, get_query_loaders, deserialize_vocab
import random
import argparse
from losses import ContrastiveLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='nuswide', help='nuswide or mirflickr25k')
    parser.add_argument('--img_size', default=224, type=int, help='image size. default(224)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--Epochs', default=60, type=int, help="Number of epoch.")
    parser.add_argument('--lr', default=0.0001, help="Learning rate.")
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--pretrained', default=True)
    parser.add_argument('--word_dim', default=300, type=int, help="dim of word")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--writelog', type=bool, default=False)
    
    # CGVR

    parser.add_argument('--d_model', default=2048, type=int,
                        help="num_channels of the resnet feature (512 for resnet18 and resnet34, 2048 for resnet50 and resnet101)")
    parser.add_argument('--CGVR_backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--enc_layers', default=0, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--kg_layers', default=2, type=int,
                        help="Number of kg_attention layers in the transformer")

    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--CGVR_nbits', default=64, type=int, help="CGVR output length")
    parser.add_argument('--use_KG', default=True)
    parser.add_argument('--margin_scaler', default=0.15 , type=float)
    parser.add_argument('--ignoring_rate', default=0.01, type=float) #0.01 DEFAULE


    args = parser.parse_args()
    setup_seed(41)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    logging = write_log(args)
    print(device)
    logging.info("Backbone: {}".format(args.CGVR_backbone))
    logging.info("Start training on {}...".format(args.dataname))
    logging.info("batch_size: {}".format(args.batch_size))
    logging.info("learning_rate: {}".format(args.lr))
    logging.info("nbits: {}".format(args.CGVR_nbits))

    data_path = "data/{}".format(args.dataname)
    query_caps = 'query/query.txt'
    base_caps = 'base/base.txt'
    train_caps = 'train/train.txt'

    vocab_path = os.path.join('dataset/vocab/{}_vocab.json'.format(args.dataname))
    vocab = deserialize_vocab(vocab_path)
    rel_matrix = np.load('dataset/concepnet/{}_rel50.npy'.format(args.dataname))
    rel_matrix = torch.from_numpy(rel_matrix)
    save_path = 'checkpoints/CGVR/{}/crn_resnet101_{}bits_{}_{}.pth'.format(args.dataname, args.CGVR_nbits, args.lr, args.dataname)
    groundTruthSimilarityMatrix = np.load("data/{}/GroundTruthSimilarityMatrix.npy".format(args.dataname))
    word_vectors = np.load("dataset/wordembedding/{}ConceptNetEmbedding.npy".format(args.dataname))
    word_vectors = torch.from_numpy(word_vectors).to(torch.double)

    train_loader = get_train_loaders(base_path=data_path, data_path=train_caps, vocab=vocab, batch_size=args.batch_size,
                                     workers=args.workers)
    query_loader = get_query_loaders(base_path=data_path, data_path=query_caps, vocab=vocab, batch_size=args.batch_size,
                                     workers=args.workers)
    base_loader = get_base_loaders(base_path=data_path, data_path=base_caps, vocab=vocab, batch_size=args.batch_size,
                                   workers=args.workers)
    print(len(train_loader), len(query_loader), len(base_loader))

    CGVR_net = build_CGVR_net(word_vectors, rel_matrix, args)
    CGVR_net = CGVR_net.to(device)
    backbone_params = CGVR_net.backbone.parameters()
    backbone_params_list = list(map(id, CGVR_net.backbone.parameters()))
    customers_params = filter(lambda p: id(p) not in backbone_params_list, CGVR_net.parameters())
    optimizer = optim.SGD([{'params': customers_params},
                           {'params': backbone_params, 'lr': args.lr * 0.1}],
                          lr=args.lr, weight_decay=0.01, momentum=0.9)
    lr_sche = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, last_epoch=-1)

    certerion = ContrastiveLoss(args)
    best_mAP = 0
    

    if args.resume:
        print('loading latest model parameters...')
        logging.info('loading latest model parameters...')
        model_state_dict = torch.load(save_path, map_location=device)
        CGVR_net.load_state_dict(model_state_dict)
        
    # training CGVR
    for epoch in range(args.Epochs):
        print('----------------------------Training----------------------------')
        print('Epoch: {}, lr: {:.6f}'.format(epoch + 1, optimizer.param_groups[0]["lr"]))
        logging.info('----------------------------Training----------------------------')
        logging.info('Epoch: {}, lr: {:.6f}'.format(epoch + 1, optimizer.param_groups[0]["lr"]))
        avg_loss = train(train_loader, CGVR_net, certerion, optimizer, epoch, device, query_loader, base_loader, groundTruthSimilarityMatrix, save_path)
        lr_sche.step()
        print("AVGLoss: {:.4f}".format(avg_loss))
        logging.info("AVGLoss: {:.4f}".format(avg_loss))
        if (epoch + 1) % 1 == 0:
            print('----------------------------Verifying----------------------------')
            logging.info('----------------------------Verifying----------------------------')
            MAP, precisions, recalls = validate(query_loader, base_loader, groundTruthSimilarityMatrix, CGVR_net, device)
            if best_mAP < MAP:
                best_mAP = MAP
                torch.save(CGVR_net.state_dict(), save_path)
                print('Teacher Model Changed!')
            else:
                print('Teacher Model Not Changed!')

            print('Teacher Model : cur_mAP = {:.4f}, best_mAP = {:.4f} \n'.format(MAP, best_mAP))
    '''    
    print('----------------------------Verifying----------------------------')
    logging.info('----------------------------Verifying----------------------------')
    MAP, precisions, recalls = validate(query_loader, base_loader, groundTruthSimilarityMatrix, CRN_net, device)
    torch.save(CGVR_net.state_dict(), save_path)

    print('CGVR Model : mAP = {:.4f} \n'.format(MAP))
    logging.info('CRN Model : mAP = {:.4f}\n'.format(MAP))   
    '''    
    logging.info("End training on {}...".format(args.dataname))
    logging.info("nbits: {}".format(args.CRN_nbits))


def getHashes(outputs):
    return (torch.sgn(outputs - 0.5) + 1) * 0.5


def train(train_loader, CGVR_net, criterion, optimizer, epoch, device, query_loader, base_loader, groundTruthSimilarityMatrix, save_path):
    loss_sum = 0
    batch_start = time.time()
    CGVR_net.train()
    best_mAP = 0
    for i, (imgs, targets, ids, lengths) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs_h1, outputs_h2, caps_embed, s_v, sim_mat, s_v2 = CGVR_net(imgs, targets, lengths)
        loss = criterion(outputs_h1, outputs_h2, caps_embed, sim_mat, epoch, s_v, s_v2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if (i + 1) % 10 == 0:
            batch_end = time.time()
            print('batch_{}, loss: {:.4f}, Time: {:.2f}'.format(i + 1, loss.item(), batch_end - batch_start))
            logging.info('batch_{}, loss: {:.4f}, Time: {:.2f}'.format(i + 1, loss.item(), batch_end - batch_start))
            batch_start = time.time()
            
    avg_loss = loss_sum / len(train_loader)
    return avg_loss


def validate(query_loader, base_loader, groundTruthSimilarityMatrix, CGVR_net, device):
    queryHashes, baseHashes = [], []
    CGVR_net.eval()
    with torch.no_grad():
        start = time.time()
        for i, (imgs, targets, ids, lengths) in enumerate(tqdm(query_loader, "queryHashBuilding")):
            imgs = imgs.to(device)
            outputs_h1 = CGVR_net(imgs)
            queryHashes.append(getHashes(outputs_h1))
        queryHashes = torch.cat(queryHashes, dim=0)
        end = time.time()
        print("Q_Hash's shape: {}, Time:{:.2f}".format(queryHashes.shape, end - start))
        logging.info("Q_Hash's shape: {}, Time:{:.2f}".format(queryHashes.shape, end - start))
        for i, (imgs, targets, ids, lengths) in enumerate(tqdm(base_loader, "baseHashBuilding")):
            imgs = imgs.to(device)
            outputs_h1 = CGVR_net(imgs)
            baseHashes.append(getHashes(outputs_h1))
        baseHashes = torch.cat(baseHashes, dim=0)
        end = time.time()
        print("B_Hash's shape: {}, Time:{:.2f}".format(baseHashes.shape, end - start))
        logging.info("B_Hash's shape: {}, Time:{:.2f}".format(baseHashes.shape, end - start))
        MAP, precisions, recalls = getMAP(queryHashes, baseHashes, groundTruthSimilarityMatrix, top=5000)
    return MAP, precisions, recalls


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def write_log(args):
    filepath = 'checkpoints/CGVR/{}/log/CGVR_resnet_{}bits_{}_{}.txt'\
        .format(args.dataname, str(args.CGVR_nbits), str(args.lr), args.dataname)
    logging.basicConfig(
        filename=filepath,
        level=logging.INFO,
    )
    return logging


if __name__ == '__main__':
    main()