import argparse
import numpy as np
from io_utils import sort_hashtags
import pickle
import tags_distr

parser = argparse.ArgumentParser(description='Sort and group tweets based on tags.')
parser.add_argument('--data', help='hashtags file', required=True)
parser.add_argument('--sorted', help='File where to write sorted_hashtags', required=True)
parser.add_argument('--output', help='File where to write dict containg the clusters', required=True)
parser.add_argument('--min-freq', help='Minimum tag frequency', type=int, default=1000)
parser.add_argument('--max-tags', help='The maximum amount of tags to be choosen', type=int, default=500)

class TagsClustering:
    def __init__(self, min_freq):
        self.min_freq = min_freq
    
    def get_tags(self, in_file, max_tags, offset=0):
        tags = []
        with open(in_file) as data:
            ofcnt = 0
            for line in data:
                if ofcnt < offset:
                    ofcnt += 1
                    continue

                parts = line.split("\t")
                if parts[0] == '__NONE__':
                    continue

                if int(parts[1]) < self.min_freq or len(tags) == max_tags:
                    break
                tags.append(parts[0])
        return tags


    def get_clusters(self, in_file, tags_file, max_tags=500):
        dict = {}
        #tags = self.get_tags(tags_file, max_tags)
        (tags, _) = tags_distr.get_tags_gini()
        #tags = np.array(self.get_tags(tags_file, 10000))
        #get random 100 tags
        #tags = tags[np.random.randint(0, len(tags), 100)]

        for one in tags:
            dict[one] = []

        with open(in_file) as data:
            tid = 0
            for tags in data:
                tmp = tags.split()
                for tag in tmp:
                    if tag != "__NONE__" and tag in dict:
                        dict[tag].append(tid)
                tid += 1

        return dict
                    


args = parser.parse_args()
if args is None:
    parser.print_help()
    sys.exit(0)

#First sort hashtags
sort_hashtags(args.data, args.sorted)
#Now we can do the clustering
tc = TagsClustering(args.min_freq)

dict = tc.get_clusters(args.data, args.sorted, args.max_tags)
print(dict.keys())
print("\nNumber of max-freq tags: {}".format(len(dict.keys())))
#Save the dict with clusters into a file
out = open(args.output, 'wb')
pickle.dump(dict, out, pickle.HIGHEST_PROTOCOL)
out.close()

