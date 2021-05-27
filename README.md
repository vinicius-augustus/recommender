# Recommender system
A recommender system, or a recommendation system (sometimes replacing 'system' with a synonym such as platform or engine), is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item.

Recommender systems are used in a variety of areas, with commonly recognised examples taking the form of playlist generators for video and music services, product recommenders for online stores, or content recommenders for social media platforms and open web content recommenders. These systems can operate using a single input, like music, or multiple inputs within and across platforms like news, books, and search queries. There are also popular recommender systems for specific topics like restaurants and online dating. Recommender systems have also been developed to explore research articles and experts, collaborators, and financial services.

In this particular project of a recommendation system, I chose to use movies as an example of item, therefore, a movie recommendation system.

A recommendation system can be made in several ways, each way with its specific characteristics and performance for a business model, in this project I chosen to build movie recommender systems based on K-Nearest Neighbour (k-NN), Matrix Factorization (MF) as well as Neural-based. The data that I have chosen to work on is the MovieLens dataset collected by GroupLens Research. This dataset has 100,000 ratings given by 943 users for 1682 movies, with each user having rated at least 20 movies. The ratings are based on a scale from 1 to 5.

The project is divided into three stages:
1. Data Preprocessing
2. Model Building
3. Results Analysis and Conclusion

## k-NN-based and MF-based Collaborative Filtering — Data Preprocessing
For k-NN-based and MF-based models, the built-in dataset ml-100k from the Surprise Python sci-kit was used. Surprise is a good choice to begin with, to learn about recommender systems. It is suitable for building and analyzing recommender systems that deal with explicit rating data.

## k-NN-based Collaborative Filtering — Model Building
Data is split into a 75% train-test sample and 25% holdout sample. GridSearchCV carried out over 5 -fold, is used to find the best set of similarity measure configuration (sim_options) for the prediction algorithm. It uses the accuracy metrics as the basis to find various combinations of sim_options, over a cross-validation procedure.

## MF-based Collaborative Filtering — Model Building
Matrix Factorization compresses user-item matrix into a low-dimensional representation in terms of latent factors. These latent factors provide hidden characteristics about users and items. A user’s interaction with an item is modelled as the product of their latent vectors.

The MF-based algorithm used is Singular Vector Decomposition (SVD).


GridSearchCV is used to find the best configuration of the number of iterations of the stochastic gradient descent procedure, the learning rate and the regularization term.
Based on GridSearch CV, the RMSE value is 0.9530. The RMSE value of the holdout sample is 0.9430. The MSE and the MAE values are 0.889 and 0.754.

## Neural- based Collaborative Filtering — Data Preprocessing
The data file that consists of users, movies, ratings and timestamp is read into a pandas dataframe for data preprocessing.
Movies and users need to be enumerated to be used for modeling. Variables with the total number of unique users and movies in the data are created, and then mapped back to the movie id and user id.
The minimum and maximum ratings present in the data are found. Ratings are then normalized for ease of training the model.

## Neural-based Collaborative Filtering — Model Building
Embeddings are used to represent each user and each movie in the data. These embeddings will be of vectors size n that are fit by the model to capture the interaction of each user/movie.

Both the users and movies are embedded into 50-dimensional (n = 50) array vectors for use in the training and test data. Training is carried out on 75% of the data and testing on 25% of the data.

To capture the user-movie interaction, the dot product between the user vector and the movie vector is computed to get a predicted rating.

The Adam optimizer is used to minimize the accuracy losses between the predicted values and the actual test values.

From the training and validation loss graph, it shows that the neural-based model has a good fit. The plot of training loss has decreased to a point of stability. The plot of validation (test) loss has also decreased to a point of stability and it has a small gap from the training loss.
The MSE and MAE values from the neural-based model are 0.075 and 0.224.

# Results Analysis and Conclusion
Neural-based collaborative filtering model has shown the highest accuracy compared to memory-based k-NN model and matrix factorization-based SVD model.
