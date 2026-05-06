from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
import numpy as np


def calculate_distances(image, training_images):
    distances = euclidean_distances([image], training_images)
    return distances.flatten()


def do_knn(image, training_images, training_image_labels, k):
    distances = calculate_distances(image, training_images)
    sorted_distance_indices = np.argsort(distances)
    k_most_similar_indices = sorted_distance_indices[:k]
    labels = [training_image_labels.iloc[index] for index in k_most_similar_indices]
    label_vote_counts = Counter(labels)
    top_votes = label_vote_counts.most_common(2)
    if len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1]:
        return do_knn(image, training_images, training_image_labels, k + 1)
    return top_votes[0][0]
