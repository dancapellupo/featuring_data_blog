
import numpy as np
from sklearn import cluster, metrics
from kneed import KneeLocator
from gap_statistic import OptimalK

from prediction_strength import get_prediction_strength


def kmeans_find_num_clusters(X, method='elbow', n_clust_min=2, n_clust_max=20, inc=1):

    if method in ['elbow', 'silhouette', 'pred_strength']:
        # For the silhouette coefficient method, mininum number of clusters must be 2:
        if method == 'silhouette':
            n_clust_min = max(n_clust_min, 2)

        # Initialize lists for different parameters:
        results_list = []

        # Create train and test sets for the prediction strength:
        if method == 'pred_strength':
            np.random.seed(42)
            msk = np.random.rand(X.shape[0]) < 0.8
            X_train, X_test = X[msk, :], X[~msk, :]

        for jj in range(n_clust_min, n_clust_max+1, inc):
            # Run k-Means:
            model = cluster.KMeans(n_clusters=jj, random_state=42, verbose=0)
            model.fit(X)

            if method == 'elbow':
                # Save the inertia statistic from the clustering algorithm:
                results_list.append(model.inertia_)

            elif method == 'silhouette':
                # Calculate and save the Silhouette score for the current clustering:
                silh_coef = metrics.silhouette_score(X, model.labels_, metric='euclidean')
                results_list.append(silh_coef)

            elif method == 'pred_strength':
                # Calculate prediction strength:
                model_train = cluster.KMeans(n_clusters=jj, random_state=42).fit(X_train)
                model_test = cluster.KMeans(n_clusters=jj, random_state=42).fit(X_test)
                pred_str = get_prediction_strength(jj, model_train.cluster_centers_, X_test, model_test.labels_)
                results_list.append(pred_str)

        if method == 'elbow':
            # Use elbow of inertia curve as initial guess for optimal cluster number:
            num_clusters = np.arange(n_clust_min, n_clust_max + 1, inc)
            # sec_derivative = np.zeros(len(results_list))
            # for ii in range(1, len(results_list) - 1):
            #     sec_derivative[ii] = results_list[ii+1] + results_list[ii-1] - 2 * results_list[ii]
            # best_clust_num = num_clusters[1 + np.argmax(sec_derivative[1:-1])]
            # print('Best cluster number (by inertia - OLD): {}'.format(best_clust_num))

            kneedle = KneeLocator(num_clusters, results_list, S=1.0, curve="convex", direction="decreasing")
            # print('Knee / Elbow:', round(kneedle.knee, 2), round(kneedle.elbow, 2))
            best_clust_num = int(round(kneedle.elbow))
            # print('Best cluster number (by inertia): {}'.format(best_clust_num))
        
        elif method == 'silhouette':
            best_clust_num = np.nanargmax(np.array(results_list)) + n_clust_min

        elif method == 'pred_strength':
            xx = np.where(np.array(results_list) > 0.8)[0]
            best_clust_num = xx[-1] + n_clust_min

    elif method == 'gap_stat':
        optimalK = OptimalK()
        best_clust_num = optimalK(X, cluster_array=np.arange(n_clust_min, n_clust_max+1, inc))
        results_list = optimalK.gap_df["gap_value"].to_list()

    print('Best cluster number (by {}): {}'.format(method, best_clust_num))

    return results_list, best_clust_num
