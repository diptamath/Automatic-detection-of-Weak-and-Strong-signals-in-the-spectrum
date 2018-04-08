import numpy as np

# Note: X and mu are assumed to be column vector
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = 1.0/(np.math.pow((2*np.pi), float(size)/2) * np.math.pow(det, 1.0/2))
        x_mu = np.matrix(x - mu)
        inv_ = np.linalg.inv(sigma)
        result = np.math.pow(np.math.e, -0.5 * (x_mu.T * inv_ * x_mu))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
        return -1

# N = Number of data points
# M = Dimension of data points
# K = Number of clusters
def initEM(dataSet,K):
    # The weight matrix is an NxK matrix. I am initializing it by assigning the N points in K clusters
    # This assignment is arbitrary. So I am doing it based on the indices of the points. This process assigns
    # same number of points for each cluster
    (N, M) = np.shape(dataSet)
    W = np.zeros([N, K])
    nPerK = N/K
    for k in range(K):
        W[np.floor(k*nPerK):np.floor((k+1)*nPerK), k] = 1
    # Then MU, SIGMA and ALPHA are calculated by applying an M-step
    Alpha,Mu,Sigma = Mstep(dataSet,W)
    return W, Alpha, Mu, Sigma

def Mstep(dataSet,W):
    (N, M) = np.shape(dataSet)
    K = np.size(W,1)
    # Each column of MU represents the mean of a cluster. 
    # So, for K clusters, there will be K columns of MU
    # Each column,
    # mu_k = (1/N_k)*sum_{1}^{N}{w_{ik}*x_i} 
    N_k = np.sum(W,0)
    Alpha = N_k/np.sum(N_k)
    Mu = dataSet.T.dot(W).dot(np.diag(np.reciprocal(N_k)))
    # SIGMA is a 3-dimensional matrix of size MxMxK. 
    # It contains K covariances for each cluster
    Sigma = np.zeros([M,M,K])
    for k in range(K):
        datMeanSub = dataSet.T - Mu[0:,k][None].T.dot(np.ones([1,N]))
        Sigma[:,:,k] = (datMeanSub.dot(np.diag(W[0:,k])).dot(datMeanSub.T))/N_k[k]
    return Alpha,Mu,Sigma

def Estep(dataSet,Alpha,Mu,Sigma):
    # We will calculate the membership weight matrix W here. W is an
    # NxK matrix where (i,j)th element represents the probability of
    # ith data point to be a member of jth cluster given the parameters
    # Alpha, Mu and Sigma
    N = np.size(dataSet,0)
    K = np.size(Alpha)
    W = np.zeros([N,K])
    for k in range(K):
        for i in range(N):
            W[i,k] = Alpha[k]*norm_pdf_multivariate(dataSet[i,:][None].T, \
                     Mu[:,k][None].T,Sigma[:,:,k])
    # Normalize W row-wise because each row represents a pdf. In other words,
    # probability of a point to be any one of the K clusters is equal to 1.
    W = W*np.reciprocal(np.sum(W,1)[None].T)
    return W
    
def logLike(dataSet,Alpha,Mu,Sigma):
    K = len(Alpha)
    N,M = np.shape(dataSet)
    # P is an NxK matrix where (i,j)th element represents the likelihood of 
    # the ith datapoint to be in jth Cluster (i.e. when z_k = 1)
    P = np.zeros([N,K])
    for k in range(K):
        for i in range(N):
            P[i,k] = norm_pdf_multivariate(dataSet[i,:][None].T,Mu[0:,k][None].T,Sigma[:,:,k])
    return np.sum(np.log(P.dot(Alpha)))

