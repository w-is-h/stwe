from io_utils import read_clst_file
import numpy as np
import pickle


def tags_twc(tags_clusters_file, twc_clusters, nclst_twc):
    f_tags = open(tags_clusters_file, 'rb')
    tags_c = pickle.load(f_tags)
    f_tags.close()

    matrix = np.zeros((len(tags_c.keys()), nclst_twc))
 
    c_id = 0
    for key in tags_c.keys():
        tids = tags_c[key]
        for tid in tids:
            # Increase number of occurences in the matrix for 
            #combination of tags cluster and twc cluster
            if twc_clusters[tid] >= 0:
                matrix[c_id, twc_clusters[tid]] += 1
        c_id += 1
    return matrix

def twc_kmeans_w2v(w2v_file, n_clst_kmeans, twc_clusters, n_clst_twc):
    w2v_c = read_clst_file(w2v_file)
    w2v_dict= {}

    for one in range(len(w2v_c)):
        if w2v_c[one] not in w2v_dict:
            w2v_dict[w2v_c[one]] = [one]
        else:
            w2v_dict[w2v_c[one]].append(one)

    matrix = np.zeros((len(w2v_dict.keys()), n_clst_twc))
    
    c_id = 0
    for key in w2v_dict.keys():
        tids = w2v_dict[key]
        for tid in tids:
            # Increase number of occurences in the matrix for 
            #combination of tags cluster and kmeans cluster
            matrix[c_id, twc_clusters[tid]] += 1
        c_id += 1
    return matrix




def tags_kmeans_w2v(tags_clusters_file, w2v_file, n_clst_kmeans):
    f_tags = open(tags_clusters_file, 'rb')
    tags_c = pickle.load(f_tags)
    f_tags.close()
    w2v_c = read_clst_file(w2v_file)
    
    matrix = np.zeros((len(tags_c.keys()), n_clst_kmeans))
    
    c_id = 0
    for key in tags_c.keys():
        tids = tags_c[key]
        for tid in tids:
            # Increase number of occurences in the matrix for 
            #combination of tags cluster and kmeans cluster
            matrix[c_id, w2v_c[tid]] += 1
        c_id += 1
    return matrix


def calc_vmeasure(matrix, beta):
    N = np.sum(matrix)

    h = calc_h(matrix, N)
    c = calc_c(matrix, N)

    return (h, c, ((1 + beta) * h * c) / ((beta * h) + c))

def calc_h(matrix, N):
    Hc = 0
    for i in range(matrix.shape[0]):
        tmp = np.sum(matrix[i]) / N
        if tmp != 0:
            Hc += tmp * np.log(tmp)

    Hck = 0
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            if matrix[j, i] != 0 and np.sum(matrix[:, i]) != 0:
                Hck += matrix[j, i] / N * np.log(matrix[j, i] / np.sum(matrix[:, i]))

    if Hc == 0:
        return 1
    else:
        return (1 - (Hck / Hc))

def calc_c(matrix, N):
    Hk = 0
    for i in range(matrix.shape[1]):
        tmp = np.sum(matrix[:, i]) / N
        if tmp != 0:
            Hk += tmp * np.log(tmp)

    Hkc = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0 and np.sum(matrix[i]) != 0:
                Hkc += matrix[i, j] / N * np.log(matrix[i, j] / np.sum(matrix[i]))

    if Hk == 0:
        return 1
    else:
        return (1 - (Hkc / Hk))


