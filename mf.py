import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import math
import time
import sys

def load_matrix(filename, num_users, num_items):
    t0 = time.time()
    counts = np.zeros((num_users, num_items))
    total = 0.0
    num_zeros = num_users * num_items
    for i, line in enumerate(open(filename, 'r')):
        user, item, count = line.strip().split('\t')
        user = int(user)
        item = int(item)
        count = float(count)
        if user >= num_users:
            continue
        if item >= num_items:
            continue
        if count != 0:
            counts[user, item] = count
            total += count
            num_zeros -= 1
        if i % 100000 == 0:
            print >> sys.stderr,  'loaded %i counts...' % i
    alpha = num_zeros / total
    print >> sys.stderr,  'alpha %.2f' % alpha
    counts *= alpha
    counts = sparse.csr_matrix(counts)
    t1 = time.time()
    print >> sys.stderr, 'Finished loading matrix in %f seconds' % (t1 - t0)
    return counts


class ImplicitMF():

    def __init__(self, counts,
                 num_factors=40,
                 num_iterations=30,
                 conv_loss_value=0.0,
                 reg_param=0.8):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.conv_loss_value = conv_loss_value
        self.reg_param = reg_param
        self.scores = None

    def train_model(self):
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))

        for i in xrange(self.num_iterations):
            t0 = time.time()
            print >> sys.stderr, 'Solving for user vectors...'
            self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
            print >> sys.stderr, 'Solving for item vectors...'
            self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))
            t1 = time.time()
            print >> sys.stderr, 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)

            # very expensive to compute the loss function. So default conv_loss_value set to 0 to skip this
            if self.conv_loss_value == 0.0:
                continue
            if self.loss_func() < self.conv_loss_value:
                break

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

        t = time.time()
        for i in xrange(num_solve):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = sparse.diags(counts_i, [0])
            pu = counts_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
            if i % 1000 == 0:
                print >> sys.stderr, 'Solved %i vecs in %d seconds' % (i, time.time() - t)
                t = time.time()

        return solve_vecs

    def loss_func(self):
        t0 = time.time()
        self.scores = self.user_vectors.dot((self.item_vectors).T)
        pui = self.counts
        pui[np.where(pui != 0)] = 1.0
        loss_val = 0.0
        for u in range(0, self.num_users):
            for i in range(0, self.num_items):
                loss_val += self.counts[u, i] * math.pow(pui[u, i] - self.scores[u, i], 2)
                for f in range(0, self.num_factors):
                    loss_val += self.reg_param * math.pow(self.user_vectors[u, f], 2)
                    loss_val += self.reg_param * math.pow(self.item_vectors[i, f], 2)
        t1 = time.time()
        print >> sys.stderr, 'finished compute value of loss function in %f seconds' % (t1 - t0)
        print >> sys.stderr, 'Value of loss function is %f' % (loss_val)
        return loss_val


    def recommend(self, user, K):
        # keep scores
        if self.scores == None:
            self.scores = sparse.csr_matrix(self.user_vectors).dot(sparse.csr_matrix(self.item_vectors).T)

        # don't recommend items already consumed by the user
        recommendable = np.zeros(self.scores.shape[1])
        for i in range(0, self.counts.shape[1]):
            if self.counts[user, i] == 0:
                recommendable[i] = self.scores[user, i]

        # sort and get the indices of top K items
        if len(recommendable) >= K:
            return recommendable.argsort()[-K:]
        else:
            return range(0, len(recommendable))
