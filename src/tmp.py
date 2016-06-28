from gensim.models import Word2Vec
import logging
from io_utils import word2vecIterator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

it = word2vecIterator("/local/kraljevic/twiter/processed/all.dat")

model = Word2Vec(it, sample=5e-5, size=100, negative=3, window=5, min_count=30, workers=8)

model.save("../output/models/word2vec.mdl")

