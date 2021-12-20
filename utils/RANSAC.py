import numpy as np
import matplotlib.pyplot as plt


class RANSAC:
    def __init__(self, in_cloud):
        """
        Application of the RANSAC algorithm in a 3D regression setting. Linear regression is computed using
        a simple least squares solution.
        :param in_cloud: input point cloud
        :type in_cloud: np.ndarray
        """
        # Rewrite for regression
        self.in_cloud_X = np.ones((in_cloud.shape[0], 3))
        self.in_cloud_X[:,1:] = in_cloud[:, 0:2]
        self.in_cloud_y = in_cloud[:, 2:]

        # Remove invalid points
        valid_mask = self.in_cloud_y[:,0] != 0
        self.in_cloud_X_valid = self.in_cloud_X[valid_mask,:]
        self.in_cloud_y_valid = self.in_cloud_y[valid_mask,:]

        self.total_points = self.in_cloud_X.shape[0]
        self.total_points_valid = self.in_cloud_X_valid.shape[0]

        self.params = None

    def fit(self, max_iter=100, n_samples=1000, threshold=0.1):
        """
        Finds the equation of a plane with the most points.
        :param max_iter: maximum number of iterations
        :type max_iter: int
        :param n_samples: number of samples to fit in each RANSAC iteration
        :type n_samples: int
        :param threshold: threshold which determines if the point is an outlier
        :type threshold: float
        :return: parameters of the final plane
        :rtype: np.ndarray
        """
        max_inliers = 0
        best_model = None

        i = 0
        while i < max_iter:
            samples_X, samples_y = self.get_samples(n_samples)
            model = self.linear_regression(samples_X, samples_y)
            n_inliers = self.evaluate_fit(model, threshold)
            if n_inliers > max_inliers:
                best_model = model
                max_inliers = n_inliers
            i += 1

        return best_model

    def get_samples(self, n_samples):
        """
        Sample points from the whole pointcloud
        :param n_samples: number of samples to drawn
        :type n_samples: int
        :return: matrix with the samples
        :rtype:np.ndarray
        """
        idx = np.random.randint(self.total_points_valid, size=n_samples)
        samples_X = self.in_cloud_X_valid[idx,:]
        samples_Y = self.in_cloud_y_valid[idx,:]
        return samples_X, samples_Y

    def linear_regression(self, X, y):
        """
        Performs a simple 3D linear regression: w* = argmin(y-Xw)^2
        :param X: traning samples
        :type X: np.ndarray
        :param y: training value
        :type y: np.ndarray
        :return: model parameters
        :rtype: np.ndarray
        """
        model = np.linalg.inv(X.T@X)@X.T@y
        return model

    def evaluate_fit(self, model, threshold):
        """
        Evaluates the regressed model: counts the number of inliers
        :param model: model parameters
        :type model: np.ndarray
        :param threshold: threshold that separates inliers from outliers
        :type threshold: float
        :return: number of inliers
        :rtype: int
        """
        distance = np.abs(self.in_cloud_y_valid - self.in_cloud_X_valid@model)
        n_inliers = (distance <= threshold).astype(int)
        return np.sum(n_inliers)

    def get_mask(self, model, threshold=0.1, show=False, new_cloud=None):
        """
        Returns a 1d vector of the points belonging to the floor.
        :param model: regression parameters
        :type model: np.ndarray
        :param threshold: distance threshold to plane
        :type threshold: float
        :return: mask in 1d of the floor points
        :rtype: np.ndarray
        """
        if new_cloud is not None:
            cloud_X = np.ones((new_cloud.shape[0], 3))
            cloud_X[:, 1:] = new_cloud[:, 0:2]
            cloud_y = new_cloud[:, 2:]
        else:
            cloud_X = self.in_cloud_X
            cloud_y = self.in_cloud_y

        distance = np.abs(cloud_y - cloud_X @ model)
        inliers_mask = (distance <= threshold).astype(bool)

        # Show mask
        if show:
            outliers_mask = np.invert(inliers_mask)
            inlier_points_X = cloud_X[inliers_mask[:, 0], :]
            inlier_points_y = cloud_y[inliers_mask]
            outlier_points_X = cloud_X[outliers_mask[:, 0], :]
            outlier_points_y = cloud_y[outliers_mask]

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(inlier_points_X[::6, 1], inlier_points_X[::6, 2], inlier_points_y[::6], label="inliers")
            ax.scatter(outlier_points_X[::6, 1], outlier_points_X[::6, 2], outlier_points_y[::6], label="outliers")
            plt.legend()
            plt.show()

        return inliers_mask
