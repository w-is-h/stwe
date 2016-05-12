#!/usr/bin/env python3
import argparse
from gensim import models, matutils
from io_utils import DocumentTimePair
import sys

#python train_word2vec.py --data= ../data/final.txt --save-path=../output/models/word2vec.dat 

parser = argparse.ArgumentParser(description='Compute vectors using gensim word2vec.')
parser.add_argument('--min-frequency', help='Minimum word frequency', type=int, default=5)
parser.add_argument('--data', help='Input data file', required=True)
parser.add_argument('--emb-dim', help='Dim of vector representations', type=int, default=100)
parser.add_argument('--save-path', help='Output file for the trained model', required=True)
parser.add_argument('--window-size', help='Max distance between current and predicted word', type=int, default=5)
parser.add_argument('--sample', help='Subsampling', type=float, default=5e-5)
parser.add_argument('--nworkers', help='Number of threads to use', type=int, default=1)
parser.add_argument('--nneg', help='Number of negative samples to use', type=int, default=5)


args = parser.parse_args()
if args is None:
    parser.print_help()
    sys.exit(0)

print("Started word2vec training")
dtp = DocumentTimePair(args.data)
word2vec = models.Word2Vec(sentences=dtp, workers=args.nworkers, negative=args.nneg, hs=0, min_count=args.min_frequency, window=args.window_size, size=args.emb_dim, sample=args.sample)
print("Finished training, saving the model")

#Save the model
word2vec.save(args.save_path)
print("Model saved.")
