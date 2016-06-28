from twe import *

opts = Options()
opts.save_path = "../output/model/sou"
opts.train_data = "../data/sou/train.dat"
opts.test_data = "../data/sou/test.dat"

opts.start_time = -5679590961
opts._start_time = -5679590961
opts.end_time = 1138662000
opts._end_time = 1138662000
opts.time_transform = 681825296

opts.nclst = 200
opts.clst_window = 30

opts.nepochs = 200

opts.batch_size = 100
opts.epoch_size = 100000

opts.window_size = 20
opts.max_pairs_from_sample = 100
opts.max_same_target = 10

main(opts)
