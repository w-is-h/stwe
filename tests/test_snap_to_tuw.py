from stwe.tweets_preprocessing import Preprocessing

in_folder = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/all_snap/"
out_folder = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/all_tuw/"

p = Preprocessing()

p.snap_to_tuw(in_folder, out_folder, offset=100, preprocessing=True)
