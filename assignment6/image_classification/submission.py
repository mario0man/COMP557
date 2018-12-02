import collections
import numpy as np

############################################################
# Problem 4.1

def runKMeans(k,patches,maxIter):
	"""
	Runs K-means to learn k centroids, for maxIter iterations.
	
	Args:
	  k - number of centroids.
	  patches - 2D numpy array of size patchSize x numPatches
	  maxIter - number of iterations to run K-means for

	Returns:
	  centroids - 2D numpy array of size patchSize x k
	"""
	# This line starts you out with randomly initialized centroids in a matrix 
	# with patchSize rows and k columns. Each column is a centroid.
	centroids = np.random.randn(patches.shape[0],k)

	numPatches = patches.shape[1]

	# for i in range(maxIter):
	#     # BEGIN_YOUR_CODE (around 19 lines of code expected)
	#     raise "Not yet implemented"
	#     # END_YOUR_CODE

	# return centroids
	centroids = np.transpose(centroids)
	patches = np.transpose(patches)

	for i in range(maxIter):
		new_centroids = np.zeros((k, centroids.shape[1]))
		new_centroids_count = np.zeros(k)
		# for all columns in patches, subtract all columns in centroids
		# creates a 3D array 
		# [[patches[0]-centroids[0], patches[0]-centroids[1], ...], 
		#  [patches[1]-centroids[0], patches[1]-centroids[1], ...]]
		differences = patches[:,None] - centroids[None,:]
		# for patch i, centroid_indices[i] is the index of the centroid closest to patch i
		centroid_indices = np.argmin([np.sum(np.square(differences), axis=2)], axis=2).flatten()
		for j in range(numPatches):
			new_centroids[centroid_indices[j]] = new_centroids[centroid_indices[j]] + patches[j]
			new_centroids_count[centroid_indices[j]] += 1.0

		centroids = new_centroids / new_centroids_count[:, np.newaxis] 
	return np.transpose(centroids)

############################################################
# Problem 4.2

def extractFeatures(patches,centroids):
	"""
	Given patches for an image and a set of centroids, extracts and return
	the features for that image.
	
	Args:
	  patches - 2D numpy array of size patchSize x numPatches
	  centroids - 2D numpy array of size patchSize x k
	  
	Returns:
	  features - 2D numpy array with new feature values for each patch
				 of the image in rows, size is numPatches x k
	"""
	k = centroids.shape[1]
	numPatches = patches.shape[1]
	features = np.empty((numPatches,k))

	# # BEGIN_YOUR_CODE (around 9 lines of code expected)
	# raise "Not yet implemented"
	# # END_YOUR_CODE
	# return features
	centroids = np.transpose(centroids)
	patches = np.transpose(patches)

	differences = patches[:,None] - centroids[None,:]
	
	dists = np.sqrt(np.sum(np.square(differences), axis=2))
	# BEGIN_YOUR_CODE (around 9 lines of code expected)
	for i in range(numPatches):
		avg_dist = np.average(dists[i])
		activations = avg_dist - dists[i]

		for j, activation in enumerate(activations):
			if activation < 0.0:
				activations[j] = 0.0
		features[i] = activations

	# END_YOUR_CODE
	return features


############################################################
# Problem 4.3.1

import math
def logisticGradient(theta,featureVector,y):
	"""
	Calculates and returns gradient of the logistic loss function with
	respect to parameter vector theta.

	Args:
	  theta - 1D numpy array of parameters
	  featureVector - 1D numpy array of features for training example
	  y - label in {0,1} for training example

	Returns:
	  1D numpy array of gradient of logistic loss w.r.t. to theta
	"""
	# BEGIN_YOUR_CODE (around 2 lines of code expected)
	# raise "Not yet implemented."
	y = -1.0 if y == 0 else 1.0
	u = np.exp(-1.0 * (np.dot(theta, featureVector)) * y)
	return 1.0 * (-1.0 * featureVector * y * u) / (1 + u)
	# END_YOUR_CODE

############################################################
# Problem 4.3.2
	
def hingeLossGradient(theta,featureVector,y):
	"""
	Calculates and returns gradient of hinge loss function with
	respect to parameter vector theta.

	Args:
	  theta - 1D numpy array of parameters
	  featureVector - 1D numpy array of features for training example
	  y - label in {0,1} for training example

	Returns:
	  1D numpy array of gradient of hinge loss w.r.t. to theta
	"""
	# BEGIN_YOUR_CODE (around 6 lines of code expected)
	# raise "Not yet implemented."
	y = -1.0 if y == 0 else 1.0
	loss = 1 - np.dot(theta, featureVector) * y
	if loss >= 0:
		return -1 * featureVector * y
	else:
		return np.zeros(len(featureVector))
	# END_YOUR_CODE

