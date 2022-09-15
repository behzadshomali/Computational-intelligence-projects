# Computational Intelligence course Projects

This repository contains the projects done in the context of the "Introduction of Computational Intelligence" course held at the Ferdowsi University of Mashhad(FUM). During the semester, we were assigned three projects as follows:

1. [Clustering](#proj1-clustering)


## Proj1: Clustering
We were supposed to work with [ORL dataset](!https://www.kaggle.com/datasets/tavarez/the-orl-database-for-training-and-testing). We should easily flatten the images and then fit the clustering algorithms with them. We were asked to work with `DBSCAN`, `K-means`, and `Agglomerative` algorithms. I experimented with three different data normalizations before feeding them to the algorithms, as follows:

* **None:** no normalization has been applied to the data
* **StandardScaler:** normalize data to have *mean* of 0 and *std* of 1
* **Range [0,1]:** normalize data pixels to have value in range of [0,1]. 

*It is worth noting that, due to using a very basic representation of our data (flattening pixels), this last method didn't make any significant change in comparison to applying no normalization!*

Furthermore, to evaluate the performance of the algorithms we had to implement the `Rand Index` metric.

<p align="center">
<img src="./figures/algos_results_table.png" width=50%>
</p>

Additionally, we had to choose one of the above algorithms and propose a solution to relatively enhance its performance. I chose DBSCAN. To have a promising clustering using DBSCAN we have to carefully choose two hyperparameters: *MinPts* and *epsilon*. 

During my experience of finding optimal parameters for this assignment, I faced many challenges to find the right *epsilon* and also figured out its high importance (especially in comparison with *MinPts*). Choosing the optimal *epsilon* has a direct relationship with the outcome. However, finding the optimal *epsilon* can be overwhelming and tricky. To remedy this, I came up with an automated approach that is able to find the optimal *epsilon* for cases we access the ground truth of our data. The process of finding *epsilon* is as follows:

1. Compute distance (in this case euclidean distance) between each pair of different classes. Now we have a list of distances per each class (say class A) indicating the distances of samples of class A to all other samples from other classes

2. Compute the average distances per class. Now we have a scalar for each class. E.g., the number associated with class A indicates the average distance between class A and other classes' samples

3. Finally, *epsilon* is computed as the mean of all obtained averages from the previous step. In the end, the value of *epsilon* is divided by a constant number (=2)

<p align="center">
<img src="./figures/proposed_method_comparison.png" width=50%>
</p>

*!!! Please note that you can also find brief documentation (written in Persian) for this assignment, in the corresponding directory of this project!*
