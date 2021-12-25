import os
import shlex
import argparse
import scipy.spatial
from tqdm import tqdm
import _pickle as cPickle  # for python3: read in python2 pickled files
import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np

test_path = 'data/icdar17_local_features/test/'
train_path = 'data/icdar17_local_features/train/'


def parseArgs(parser):
    parser.add_argument('--labels_test', default='data/icdar17_local_features/icdar17_labels_test.txt',
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', default='data/icdar17_local_features/icdar17_labels_train.txt',
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test', default=test_path,
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train', default=train_path,
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    return parser


def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()

    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)
    return all_files, labels


def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors


def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # Done: minibarch K means to find the cluster centers
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000).fit(descriptors)

    mus = kmeans.cluster_centers_
    return mus


def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    # Done: compute distance matrix
    distance_matrix = scipy.spatial.distance_matrix(descriptors, clusters)

    # create hard assignment
    min_dist_idx = np.argmin(distance_matrix, axis=1)
    assignment = np.zeros((len(descriptors), len(clusters)))
    assignment[np.arange(min_dist_idx.shape[0]), min_dist_idx] = 1
    # Done: set to 1 the closest and 0 the others

    return assignment


def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):  # For each test file (image descriptors)
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        a = assignments(desc, mus)  # DONE: Vectors of hard assignments (1 for closest center, 0 otherwise)

        N, D = desc.shape  # T: number of samples x, D: number of features x sample
        f_enc = np.zeros((N, D*K), dtype=np.float32)

        # For GMP:
        diff = np.zeros((1, K * D))  # Final encodings

        for k in range(mus.shape[0]):  # For each cluster
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the
            # difference to the cluster center than computing the differences
            # first and then select

            # get all descriptors assigned to a cluster center
            assigned_descriptors_ind = np.nonzero(a[:, k])[0]

            if len(assigned_descriptors_ind) == 0: continue

            # substract cluster center from assigned descriptors
            f_enc[assigned_descriptors_ind, k*D: k*D + D] = desc[assigned_descriptors_ind] - mus[k]

            if gmp:
                X_ridge = f_enc[assigned_descriptors_ind,  k*D: k*D + D]  # (N, D)
                y_ridge = np.ones((len(assigned_descriptors_ind), 1))
                clf = Ridge(alpha=gamma, fit_intercept=False, max_iter=500, solver="sparse_cg")
                clf.fit(X_ridge, y_ridge)
                diff[0, k*D: k*D + D] = clf.coef_

        if not gmp:
            # GENERALIZED SUM POOLING: Sum all points into one vector
            diff = np.sum(f_enc, axis=0).reshape(1, K * D)

        # power normalization
        if powernorm:
            # DONE: Normalize.
            alpha = 0.5  # Square root
            diff = np.sign(diff) * np.abs(diff) ** alpha

        # DONE: l2 normalization
        diff = normalize(diff, norm='l2')

        # Stack all encodings together
        if len(encodings) == 0:
            encodings = diff
        else:
            encodings = np.vstack((encodings, diff))

    return encodings


def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """
    # Setup the new encodings: parameters of a trained SVM for each test case
    M = encs_train.shape[0]
    N = encs_test.shape[0]
    D = encs_train.shape[1]
    new_encs = np.zeros((N, D))

    # Setup labels for SVD: First (test samp) is positive, the rest (all training samp) are negatives
    y = np.ones(M + 1)
    x = np.append((encs_test[0].reshape(1, -1)), encs_train, axis=0)
    y[1:] = -1

    # Function for each E-SVM step (each test sample).
    def loop(*i):
        x[0,:] = encs_test[i].reshape(1, -1)
        lin_clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=C, class_weight="balanced")
        lin_clf.fit(x, y)
        weight_vector = lin_clf.coef_  # NEW ENCONDINGS: are the trained parameters normalized
        weight_norm = normalize(weight_vector, norm='l2')
        return weight_norm

    new_encs = list(map(loop, range(len(encs_test))))
    new_encs = np.concatenate(new_encs, axis=0)

    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        desc = cPickle.load(ff) #, encoding='latin1')
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # descriptors
    # mask out distance with itself
    TK = encs.shape[0]
    dists = np.zeros((TK, TK))
    for i in range(TK):
        for j in range(TK):
            dists[i][j] = 1 - np.dot(encs[i].reshape((1, -1)), encs[j].reshape((-1, 1)))
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists


def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)  # Compute pairwise distances between normalized encodings
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


def main(pownorm, gmp, gamma=1):
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42)  # fix random seed

    print("#########################################")
    print("Experiment: POWNORM:", pownorm, " GMP:", gmp)
    print("#########################################")

    #### Parameters for experiments
    pownorm = pownorm  #
    args.gmp = gmp  #
    args.gamma = gamma  # Only for gmp (has to be with ridge regression)

    args.overwrite = True  # To redo the encodings (with and without powernorm)

    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix,
                                         args.labels_train)
    print('#train: {}'.format(len(files_train)))
    if not os.path.exists('mus.pkl.gz'):
        # DONE: load a set of random descriptors (x - features) from each image. The descriptors are originally
        # generated by a neural network (automatic feature extraction).
        descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)
        print('> loaded {} descriptors:'.format(len(descriptors)))

        print('> compute dictionary')
        # DONE: compute dictionary, which is represented by 100 cluster centers of the descriptors.
        mus = dictionary(descriptors, n_clusters=100)  # Simply a k-means
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))
    fname = 'enc_test_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # DONE: get VLAD encoding (also exercise c - powernorm)
        enc_test = vlad(files_test, mus, powernorm=pownorm, gmp=args.gmp, gamma=args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)

    # cross-evaluate test encodings: compute pairwise distances between all normalized encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)  # First result: WITHOUT EXEMPLAR CLASSIFIERS

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        enc_train = vlad(files_train, mus, powernorm=pownorm, gmp=args.gmp, gamma=args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation')
    enc_test = esvm(enc_test, enc_train)
    # eval
    evaluate(enc_test, labels_test)  # Second result: WITH EXEMPLAR CLASSIFIERS
    print('> evaluate')


if __name__ == '__main__':
    # experiments
    # 1. DONE With GSP without Powernorm:
    main(pownorm=False, gmp=False)

    # 2. DONE With GSP With Powernorm:
    main(pownorm=True, gmp=False)

    # 3. With GMP without Powernorm:
    main(pownorm=False, gmp=True)

    # 4. With GMP with Powernorm:
    main(pownorm=True, gmp=True)
