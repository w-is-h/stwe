from nltk.corpus import stopwords
from langdetect import detect
import re
from utils import get_logger
import os
from exceptions import TweetLengthValidationError
import time
from datetime import datetime
from params import *

log = get_logger(name='tweets_preprocessing')

class Preprocessing(object):
    def __init__(self, doc_min_len=6):
        self.stopwords = stopwords.words('english')
        self.doc_min_len = doc_min_len
        # Anything that is not a letter, number or _'-\s.
        self.re_nonan = re.compile(r"[^a-z0-9_'.\-? ]")
        # Match .\-?'
        self.re_pun = re.compile(r"[.\-?']")
        # Remove repetition of letters, if more than three in a row
        self.re_rep_let = re.compile(r'([a-z])\1{2,}')
        # @something - mentions in twitter
        self.re_atx = re.compile(r"@\w+")
        # Hashtags
        self.re_hashtag = re.compile(r"#\w+")
        # Space
        self.re_wspace = re.compile(r"[ ]+")
        # Web addresses 
        self.re_web = re.compile(r"(?:http:|www)+[\w\-/\.]*")
        # Numbers, anywhere
        self.re_num = re.compile(r"(?:[^a-z]+\d[^a-z]+)")

    def pp_doc(self, doc):
        return [x.strip() for x in doc.split()
                if x not in self.stopwords and len(x) > self.doc_min_len]

    def preprocess_tweet_general(self, tweet):
        """ This will clean the tweet of spaces, numbers and similar garbage, plus
        it will will split a tweet into three parts.

        return: (tweet, mnts, tags)
            tweet - the cleaned text of a tweet
            mnts - @mentions in a tweet
            tags - #tags in a tweet
        """
        tweet = tweet.lower()
        tweet = re.sub(self.re_web, " ", tweet)
        mnts = re.findall(self.re_atx, tweet)
        mnts = [x.replace("@", "").strip() for x in mnts]
        tweet = re.sub(self.re_atx, " ", tweet)
        tweet = re.sub(self.re_num, " ", tweet)
        tags = re.findall(self.re_hashtag, tweet)
        tags = [x.replace("#", "").strip() for x in tags]
        tweet = re.sub(self.re_hashtag, " ", tweet)
        tweet = re.sub(self.re_nonan, " ", tweet)
        tweet = re.sub(self.re_pun, '', tweet)
        tweet = re.sub(self.re_rep_let, r'\1', tweet)
        tweet = re.sub(self.re_wspace, " ", tweet).strip()

        return (tweet, mnts, tags)

    def str_to_time(self, tweet_time):
        """ Converts a SNAP tweet time format, e.g. 2009-06-11 16:56:47 to python time.

            tweet_time: A string representing time in SNAP
            return: A time object
        """
        t = time.mktime(datetime.strptime(
            (" ".join(tweet_time.split()[1:])).strip(), "%Y-%m-%d %H:%M:%S").timetuple())

        return t

    def process_one_snap_line(self, line, tweet, preprocessing):
        """ Converts one line of tweets in SNAP format
        into the appropriate entry in the 'tweet' dictionary.

        line: one line of text in SNAP format
        tweet: dictionary containing data for one tweet

        except: SkipTweet - Exception thrown if the tweet is to short and should
                            be skipped
        return: boolean - Is one tweet completly read, meaning we have
                time, user and words
        """

        if line[0] == 'T':
            #Means beginning of a new tweet
            t = self.str_to_time(line)
            tweet['time'] = t
        elif line[0] == 'U':
            user = line.split("/")[-1]
            tweet['user'] = user
        elif line[0] == 'W':
            # We use line[2:] to skip the W at the begining in SNAP format
            if preprocessing:
                (words, mentions, hashtags) = self.preprocess_tweet_general(line[2:].strip()) 
                tweet['mentions'] = mentions
                tweet['hashtags'] = hashtags
            else:
                words = line[2:].strip()

            # Skip tweets shorter than the limits
            if len(words.split(" ")) > MIN_WORDS and len(words) > MIN_CHARS:
                tweet['words'] = words
            else:
                raise TweetLengthValidationError("Tweet is too short: {}".format(line))

            # This means the tweet dictionary is now complete, as we have the 
            #time, user and words. We return True.
            return True

        # Return False as the tweet dictionary is still not complete.
        return False

    def _skip_and_position(self, data, offset):
        """ Skips and correctly positions the cursor to the start of a tweet
        in a open input file, on tweets data in SNAP format.

        data: an open file
        offset: how much to skip
        """

        for i in range(offset):
            # Skip first 'offset' lines, usually just garbage
            next(data)

        while True:
            # Find next blank line
            tmp = next(data)
            if tmp == "\n":
                break


    def _write_tweet_to_file(self, tweet, output_files):
        """ Writes a tweet to one/three files

        tweet: A dictionary containing data for one tweet
        output_files: A dictionary with all the necessary output files (one or three)
        """

        if len(output_files) == 3:
            # This means we have output files for mentions and hashtags, or in fact
            #preprocessing was done
            output_files['mentions'].write("{}\n".format("\t".join(tweet['mentions'])))
            output_files['hashtags'].write("{}\n".format("\t".join(tweet['hashtags'])))

        # In all cases write the text
        output_files['tweets'].write("{}\t{}\t{}\n".format(
                                     tweet['time'],
                                     tweet['user'].strip(),
                                     tweet['words'].strip()))


    def snap_to_tuw(self, input_folder, output_folder, ext='txt', offset=4000,
            preprocessing=False):
        """ Converters tweets from SNAP format to TUW, meaning <time>\t<user>\t<words>.
        This will also remove tweets shorter than the limits in params.py . It will also
        combine all files in input_folder into one(three) output file.

        input_folder: where all the data is located
        output_folder: where the data will be written
        ext: extensions of the files to be read in the input_folder
        offset: Number of lines to skip from each file
        preprocessing: Will the input text be preprocessed or not, depending on this
            the output is
                one file
                    tweets.txt - all tweets without cleaning
                three files
                    tweets.txt   - all tweets with cleaning plus
                    mentions.txt - all @mentions
                    hashtags.txt - all #tags
        """
        # Get the input files
        files = os.listdir(input_folder)
        files = [x for x in files if x.endswith(ext)]
        files.sort()

        # Open output files
        output_files = {}
        output_files['tweets'] = open(os.path.join(output_folder, 'tweets.txt'), 'w')
        if preprocessing:
            output_files['mentions'] = open(os.path.join(output_folder, 'mentions.txt'), 'w')
            output_files['hashtags'] = open(os.path.join(output_folder, 'hashtags.txt'), 'w')

        cnt = 0
        c_cnt = 0
        dont_log_at = 0

        tweet = {}
        for f in files:
            log.info("Started processing of file: {}".format(f))

            with open(os.path.join(input_folder, f), 'r') as data:
                # Skip some lines and position the cursor to the beginning of a tweet
                try:
                    self._skip_and_position(data, offset)
                except:
                    log.info("Data in {}, shorter than offset".format(f))

                for line in data:
                    if cnt % CONV_LOG_EVERY == 0 and cnt != dont_log_at:
                        log.info(" {:,}/{:,} - Tweets processed vs Tweets choosen".format(
                            cnt, c_cnt))
                        dont_log_at = cnt

                    try:
                        to_add = self.process_one_snap_line(line, tweet, preprocessing)
                        if to_add:
                            self._write_tweet_to_file(tweet, output_files)
                            cnt += 1
                            c_cnt += 1
                        else:
                            # Do nothing, just read the next line
                            pass
                    except TweetLengthValidationError as e:
                        # This just means a tweet was skipped because 
                        #it was too short
                        cnt += 1


    def separate_languages(self, in_tweets, output_folder, in_ht=None, in_mt=None):
        """ Separate the tweets in TUW format by language

        in_tweets: file containing tweets in TUW format
        output_folder: where the new data separated by language will be 
            written
        in_ht: file containing hashtags
        in_mt: file containing mentions
        """
        output_folders = []
        output_files = {}

        cnt = 0
        if in_ht is not None and in_mt is not None:
            with open(in_tweets) as tweets, open(in_ht) as hashtags, open(in_mt) as mentions:
                for tweet, hashtag, mention in zip(tweets, hashtags, mentions):
                    # Split the tweet to get the time, user, words
                    parts = tweet.split("\t")
                    if cnt % CONV_LOG_EVERY == 0:
                        log.info(" {:,} Tweets processed".format(cnt))
                    try:
                        lng = detect(parts[2])

                        if lng not in output_folders:
                            # Create the output_folder if it doesn't exist
                            lng_path = os.path.join(output_folder, lng)
                            os.makedirs(lng_path, exist_ok=True)

                            # Create the necessary files in the new output folder
                            output_files[lng + '_tweets'] = open(os.path.join(lng_path,
                                'tweets.txt'), 'w')
                            output_files[lng + '_hashtags'] = open(os.path.join(lng_path,
                                'hashtags.txt'), 'w')
                            output_files[lng + '_mentions'] = open(os.path.join(lng_path,
                                'mentions.txt'), 'w')
                            output_folders.append(lng)

                        # Write everything into the appropirate folder
                        output_files[lng + '_tweets'].write("{}\n".format(tweet.strip()))
                        output_files[lng + '_hashtags'].write("{}\n".format(hashtag.strip()))
                        output_files[lng + '_mentions'].write("{}\n".format(mention.strip()))
                    except Exception as e:
                        pass
                        #print(e)
                    cnt += 1
        else:
            raise NotImplementedError("This function is not implemented, please set the 
                                       in_ht and in_mt files")
