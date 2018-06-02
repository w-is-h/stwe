import argparse
from kmeans_word2vec import KmeansWord2Vec
from numpy import save
from vmeasure import tags_kmeans_w2v, calc_vmeasure

parser = argparse.ArgumentParser(description='Cluster tweets using kmeans and word2vec.')
parser.add_argument('--tweets', help='Processed tweets file', required=True)
parser.add_argument('--repr', help='File with vector representations', required=True)
parser.add_argument('--nclst', help='Number of clustesr', required=True, type=int)
parser.add_argument('--niter', help='Number of iterations - or minibatches', type=int, default=100)
parser.add_argument('--random-range', help='Range for random sampling', nargs=2, type=int, required=True)
parser.add_argument('--batch-size', help='Batch size for writing', type=int, default=10000)
parser.add_argument('--output', help='File where to write clusters', required=True)
parser.add_argument('--output-means', help='File where to write mean vectors', required=True)
parser.add_argument('--tags-clusters-file', help="Dict with tags clusters", required=True)
parser.add_argument('--b-dict-file', help="File for twc b", default=None)
parser.add_argument('--means-file', help="file for twc clst", default=None)

args = parser.parse_args()
if args is None:
    parser.print_help()
    sys.exit(0)
#Init Kmeans
kmeans = KmeansWord2Vec(args.nclst, args.repr)

#Calculate means
print("Kmeans calculation started\n")
kmeans.calc_kmeans(args.tweets, args.niter, args.random_range)

#Predict clusters for all tweetes
print("Kmeans predction started\n")
kmeans.predict(args.tweets, args.output, args.batch_size)
print("Saving started\n")
save(args.output_means, kmeans.kmeans.cluster_centers_)

#Calc vmeasure for kmeans
matrix = tags_kmeans_w2v(args.tags_clusters_file, args.output, args.nclst)
print(calc_vmeasure(matrix, 1))

print("DONE")

#Write kmeans to file
f = open(args.output_means, 'w')
for one in kmeans.kmeans.cluster_centers_:
    f.write("{}\n".format(one))
f.close()
"""
if args.b_dict_file and args.means_file:
    kmeans = KmeansWord2Vec(args.nclst, args.repr)
    kmeans.kmeans_words(args.b_dict_file, args.means_file)
"""
