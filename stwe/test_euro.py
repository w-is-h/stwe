from twe import *
import os

opts = Options()
opts.save_path = "../output/model/euro/first"

try:
    os.makedirs(opts.save_path)
except:
    pass

opts.learning_rate=0.0005
opts.tau = 3

opts.train_data = open("../data/euro_train.dat", 'r').readlines()
opts.test_data = open("../data/euro_test.dat", 'r').readlines()

opts.start_time = 1464731999
opts._start_time = 1464731999
opts.end_time = 1470005990
opts._end_time = 1470005990
opts.time_transform = 5000000

opts.nclst = 300
opts.clst_window = 50

opts.nepochs = 600

opts.batch_size = 200
opts.epoch_size = 100000

opts.window_size = 10
opts.max_pairs_from_sample = 30
opts.max_same_target = 5
opts.min_count = 40

main(opts)
