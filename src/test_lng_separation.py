from tweets_preprocessing import Preprocessing

in_tweets = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/test_tuw/tweets.txt"
in_ht = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/test_tuw/hashtags.txt"
in_mnt = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/test_tuw/mentions.txt"

out_folder = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/test_lng/"

p = Preprocessing()

p.separate_languages(in_tweets, in_ht, in_mnt, out_folder)
