from stwe.tweets_preprocessing import Preprocessing

in_tweets = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/all_tuw/tweets.txt"
in_ht = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/all_tuw/hashtags.txt"
in_mnt = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/all_tuw/mentions.txt"

out_folder = "/home/wish/Private/other/stwe/data/stnd_tweets/raw/all_lng/"

p = Preprocessing()

p.separate_languages(in_tweets=in_tweets, output_folder=out_folder, 
                     in_ht=in_ht, in_mt=in_mnt)
