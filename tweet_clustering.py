import re
import math
import string
import numpy as np


class KMeans:
    # Class constructor
    def __init__(self):
        self.k = 0
        self.centroids = {}
        self.previous_centroids = {}
        self.clusters = {}
        self.tweets = {}
        self.original_tweets = {}
        self.random_seed = None

    # Calculate the jaccard distance between tweets.
    @staticmethod
    def calculate_jaccard_distance(centroids, tweet_set):
        return 1 - len(centroids.intersection(tweet_set)) / len(centroids.union(tweet_set))

    # Preprocess tweets
    def __preprocess_tweets(self, tweets):
        tweets = tweets.dropna()  # Drop nulls
        tweets = tweets.iloc[:, 0].str.split('|', expand=True)

        for index in range(len(tweets)):
            tweet_id = str(tweets.iloc[index, 0])  # Store tweet id as key in dictionary
            tweet = str(tweets.iloc[index, 2])
            self.original_tweets[tweet_id] = tweet  # Store original tweet
            tweet = tweet.replace('#', '')  # Remove #'s
            tweet = ' '.join(filter(lambda x: x[0] != '@', tweet.split(' ')))  # Remove words with @
            tweet = re.sub(r"http\S+", "", tweet)  # Remove url with http
            tweet = re.sub(r"www\S+", "", tweet)  # Remove url with www
            tweet = tweet.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            tweet = tweet.lower()
            self.tweets[tweet_id] = tweet.split()  # tweet_id : list of words from tweet

    # Initialize means randomly.
    def __initialize_means(self):
        number_of_tweets = len(self.tweets)
        # Use random seed if provided to class.
        if self.random_seed is not None:
            rand = np.random.default_rng(self.random_seed)
        else:
            rand = np.random.default_rng()
        rand_integer_validation = []
        keys = list(self.tweets.keys())
        # Get k random, unique numbers and use as index into the tweets to initialize means.
        for i in range(self.k):
            integer = rand.integers(number_of_tweets + 1)
            while integer in rand_integer_validation:  # Check if integer is unique. Generate new if not.
                integer = rand.integers(number_of_tweets + 1)
            rand_integer_validation.append(integer)
            key = keys[integer]
            self.centroids[key] = set(self.tweets[key])

    # Assign each tweet to a cluster based on its distance from the centroids.
    def __assign_clusters(self):
        tweets = {key: value for (key, value) in self.tweets.items()}
        # Create dictionary with key and a list with the key and value set
        self.clusters = {key: [list((key, value))] for (key, value) in self.centroids.items()}
        for key in list(self.centroids.keys()):
            tweets.pop(key)  # Remove centroids

        # Loop through tweets and find the closest centroid. Assign it to the cluster dictionary.
        for key, value in tweets.items():
            tweet = key, set(value)
            closest_centroid = None
            closest_centroid_dist = math.inf
            for k, v in self.centroids.items():
                dist = KMeans.calculate_jaccard_distance(v, tweet[1])
                if dist < closest_centroid_dist:
                    closest_centroid = k
                    closest_centroid_dist = dist
            temp_array = self.clusters.get(closest_centroid)
            temp_array.append(list(tweet))  # Append list with key and value set
            self.clusters[closest_centroid] = temp_array

    # Calculate new cluster means.
    def __calculate_cluster_mean(self):
        # Store current state before new cluster means are found.
        self.previous_centroids = {key: value for (key, value) in self.centroids.items()}
        self.centroids = {}
        # Loop through cluster tweets. Assign the tweet that has the smallest running total using the jaccard
        # distance between all tweets in the cluster.
        for key, value in self.clusters.items():
            new_centroid = None
            new_centroid_value = None
            new_centroid_sum = math.inf
            for i in range(len(value)):
                summation = 0
                tweet = value[i][1]
                for j in range(len(value)):
                    summation += KMeans.calculate_jaccard_distance(tweet, value[j][1])
                if summation < new_centroid_sum:
                    new_centroid = value[i][0]
                    new_centroid_value = value[i][1]
                    new_centroid_sum = summation
            self.centroids[new_centroid] = new_centroid_value

    # Fit model to data.
    # Stop when the centroid values don't change after one iteration.
    def fit(self, tweets, k=2, random_seed=None):
        different = True
        self.k = k
        self.random_seed = random_seed
        self.__preprocess_tweets(tweets)
        self.__initialize_means()
        self.__assign_clusters()
        while different:
            comparison_array = []
            self.__calculate_cluster_mean()
            self.__assign_clusters()
            for key in self.previous_centroids.keys():
                if key in self.centroids:
                    comparison_array.append(True)
                else:
                    comparison_array.append(False)
            if all(comparison_array):
                different = False

    # Calculate sse
    def sse(self):
        sse = 0
        for values in self.clusters.values():
            summation = 0
            centroid_value = values[0][1]
            for index in range(len(values)):
                summation += (KMeans.calculate_jaccard_distance(centroid_value, values[index][1])) ** 2
            sse += summation
        return sse

    # Print cluster tweets and their sizes.
    def print_clusters(self):
        cluster_number = 1
        for values in self.clusters.values():
            print('Cluster ' + str(cluster_number) + ': size = ' + str(len(values)))
            for index in range(len(values)):
                print(self.original_tweets[values[index][0]])
            cluster_number += 1
            print('\n')

    # Print cluster count only.
    def print_cluster_count(self):
        cluster_number = 1
        for values in self.clusters.values():
            print('Cluster ' + str(cluster_number) + ': size = ' + str(len(values)))
            cluster_number += 1
