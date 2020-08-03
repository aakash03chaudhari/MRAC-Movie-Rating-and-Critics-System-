import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mysite import helper
from mysite.helper import Dict
from sklearn.cluster import KMeans

def call_KM(genre1,genre2,genre3):
        movies = pd.read_csv('mysite/movies.csv')
        ratings = pd.read_csv('mysite/ratings.csv')

        # genre1='Adventure'
        # genre2='Sci-Fi'
        # genre3='Action'
        my_clusters=0
        helper.set_Variables(genre1,genre2,genre3)

        genre_ratings = helper.get_genre_ratings(ratings, movies, [genre1, genre2], [Dict[genre1], Dict[genre2]])
        biased_dataset = helper.bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)

        print( "Number of records: ", len(biased_dataset))
        biased_dataset.head()
        helper.draw_scatterplot(biased_dataset[Dict[genre2]],Dict[genre2], biased_dataset[Dict[genre1]], Dict[genre1],'mysite/static/mysite/Normal.png')
        # plt.savefig('mysite/static/mysite/Normal.png')
        #
        # plt.close('mysite/static/mysite/Normal.png')

        X = biased_dataset[[Dict[genre2],Dict[genre1]]].values


        # TODO: Create an instance of KMeans to find two clusters
        kmeans_1 = KMeans(n_clusters=2, random_state=0)
        predictions = kmeans_1.fit_predict(X)
        helper.draw_clusters(biased_dataset, predictions,'mysite/static/mysite/TwoCluster.png')
        # plt.savefig('mysite/static/mysite/TwoCluster.png')
        # plt.close('TwoCluster.png')

        # TODO: Create an instance of KMeans to find three clusters
        kmeans_2 = KMeans(n_clusters=3, random_state=1)
        predictions_2 = kmeans_2.fit_predict(X)
        helper.draw_clusters(biased_dataset, predictions_2,'mysite/static/mysite/ThreeCluster.png')
        # plt.savefig('mysite/static/mysite/ThreeCluster.png')
        # plt.close('ThreeCluster.png')

        # TODO: Create an instance of KMeans to find four clusters
        kmeans_3 = KMeans(n_clusters=4, random_state=3)
        predictions_3 = kmeans_3.fit_predict(X)
        helper.draw_clusters(biased_dataset, predictions_3,'mysite/static/mysite/FourCluster.png')
        # plt.savefig('mysite/static/mysite/FourCluster.png')
        # plt.close('FourCluster.png')



        possible_k_values = range(2, len(X)+1, 5)
        errors_per_k = [helper.clustering_errors(k, X) for k in possible_k_values]
        list(zip(possible_k_values, errors_per_k))
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_xlabel('K - number of clusters')
        ax.set_ylabel('Silhouette Score (higher is better)')
        ax.plot(possible_k_values, errors_per_k)
        fig.savefig('mysite/static/mysite/score.png')
        plt.close(fig)


        # Ticks and grid
        xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
        ax.set_xticks(xticks, minor=False)
        ax.set_xticks(xticks, minor=True)
        ax.xaxis.grid(True, which='both')
        yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
        ax.set_yticks(yticks, minor=False)
        ax.set_yticks(yticks, minor=True)
        ax.yaxis.grid(True, which='both')


        # TODO: Create an instance of KMeans to find seven clusters
        kmeans_4 = KMeans(n_clusters=7, random_state=6)
        predictions_4 = kmeans_4.fit_predict(X)
        helper.draw_clusters(biased_dataset, predictions_4,'mysite/static/mysite/BestCluster.png', cmap='Accent')
        # plt.savefig('mysite/static/mysite/BestCluster.png')
        # plt.close('BestCluster.png')

        biased_dataset_3_genres = helper.get_genre_ratings(ratings, movies,
                                                             [genre1, genre2, genre3],
                                                             [Dict[genre1], Dict[genre2], Dict[genre3]])
        biased_dataset_3_genres = helper.bias_genre_rating_dataset(biased_dataset_3_genres, 3.2, 2.5).dropna()
        print( "Number of records: ", len(biased_dataset_3_genres))


        X_with_action = biased_dataset_3_genres[[Dict[genre2],
                                                 Dict[genre1],
                                                 Dict[genre3]]].values

        # TODO: Create an instance of KMeans to find seven clusters
        kmeans_5 = KMeans(n_clusters=7)
        predictions_5 = kmeans_5.fit_predict(X_with_action)
        helper.draw_clusters_3d(biased_dataset_3_genres, predictions_5,'mysite/static/mysite/3DCluster.png')
        # plt.savefig('mysite/static/mysite/3DCluster.png')
        # plt.close('3DCluster.png')

        #Merge the two tables then pivot so we have Users X Movies dataframe
        ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )
        user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
        user_movie_ratings.iloc[:6, :10]
        n_movies = 30
        n_users = 18
        most_rated_movies_users_selection = helper.sort_by_rating_density(user_movie_ratings, n_movies, n_users)
        most_rated_movies_users_selection.head()

        helper.draw_movies_heatmap(most_rated_movies_users_selection,'mysite/static/mysite/HeatMap.png')
        # plt.savefig('mysite/static/mysite/HeatMap.png')
        # plt.close('HeatMap.png')


#call_KM('Adventure','Romance','Drama')
