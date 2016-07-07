from twe import *
import os

opts = Options()
opts.save_path = "../output/model/nips/same_div_00001_01"

try:
    os.makedirs(opts.save_path)
except:
    pass

opts.learning_rate=0.0001
opts.tau = 0.1

opts.train_data = "../data/train.txt"
opts.test_data = "../data/test.txt"

opts.start_time = 536454000 
opts._start_time = 536454000
opts.end_time = 1041375600
opts._end_time = 1041375600
opts.time_transform = 50000000

opts.nclst = 300
opts.clst_window = 30

opts.nepochs = 500

opts.batch_size = 200
opts.epoch_size = 100000

opts.window_size = 20
opts.max_pairs_from_sample = 100
opts.max_same_target = 10
opts.min_count = 30

main(opts)
