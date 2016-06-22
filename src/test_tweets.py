from twe import *

opts = Options()
opts.save_path="../output/model/tweets/"
opts.train_data = "/local/kraljevic/twiter/processed/1M_train.dat"
opts.test_data = "/local/kraljevic/twiter/processed/1M_test.dat"

opts.tau = 0

opts.start_time = 1244732370
opts._start_time = 1244732370
opts.end_time = 1262300364
opts._end_time = 1262300364
opts.time_transform = 1756799

opts.nclst = 500
opts.clst_window = 50

opts.nepochs = 500

opts.batch_size = 500
opts.epoch_size = 200000
#opts.epoch_size = 2000

opts.window_size = 5
opts.max_pairs_from_sample = 20
opts.max_same_target = 4
opts.tags_clst_files = ['../output/clustering/tags_clst_85.dat']
opts.min_count = 100

main(opts)
print("___ALL_DONE___")
