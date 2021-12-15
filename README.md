# Active Learning performance in sentiment analysis with Spanish tweets
Cloud based active learning project for sentiment analysis for tweets in Spanish.

## Goal

To train a neural network model with 100% of labeled data points and compare it with other 3 trained models using active learning techniques.

## Solution
The solution consists in four main phases:
1.	Obtaining a dataset  of tweets in Spanish language.
2.	Pre-processing of tweets using TF-IDF.
3.	Trainig of the models.
4.	Compare and obtain results.

![Solution process](/images/solution_process.png "Solution process")

## Active Learning

Active learning is a group of machine learning techniques for the Smart selection of data labeling to obtain a good performance in a model without necessarily labeling all the data.
In Active Learning, the learning algorithm is allowed to proactively select the subset of available examples to be labeled next from a pool of yet unlabeled instances. The fundamental belief behind the concept is that a Machine Learning algorithm could potentially achieve a better accuracy while using fewer training labels if it were allowed to choose the data it wants to learn from. Such an algorithm is referred to as an active learner. Active learners are allowed to dynamically pose queries during the training process, usually in the form of unlabeled data instances to be labeled by what is called an oracle, usually a human annotator. 

![Active learning pipeline](/images/active_learning_process.png "Active learning pipeline")

In this figure we can see an standard active learning cycle, where a smart querying of the data may give a good accuracy compared with an stochastic sampling of the same data.
Active Learning can be implemented through different scenarios. In essence, deciding whether or not to query a specific label comes down to deciding whether the gain from obtaining the label offsets the cost of collecting that information. In practice, making that decision can take several forms depending on whether the scientist has a limited budget or simply tries to minimize his/her labeling bill. Overall, three different categories of active learning can be listed:
* The stream-based selective sampling scenario, which consists in determining if it would be sufficiently beneficial to enquire for the label of a specific unlabeled entry in the dataset. As the model is being trained and is presented with a data instance, it immediately decides if it wants to see the label. The disadvantage of that approach naturally comes from the absence of guarantee that the data scientist will stay within his/her budget.
* The pool-based sampling scenario is also the best-known one. It attempts to evaluate the entire dataset before selecting the best query, or a set of best queries. The active learner is usually initially trained on a fully labeled fraction of the data which generates a first version of the model which is subsequently used to identify which instances would be the most beneficial to inject in the training set for the next iteration (or active learning loop).  One of its biggest downsides comes from its memory-greediness.
* The membership query synthesis scenario might not be applicable to all cases as it implies the generation of synthetic data. In this scenario, the learner is allowed to construct its own examples for labeling. This approach is promising to solve cold-start problems (like in search) when generating a data instance is easy to do.

## Uncertainty sampling
The approach used to determine which data instance to label next is referred to as a querying strategy. Below, we are listing those that are the most commonly used and studied ones:

### Classification uncertainty
The most popular among those strategies is the least confidence strategy. In classification uncertainty the model selects those instances with the lowest confidence level to be labeled next. 
Classification uncertainty is defined by:

![Uncertainty formula](/images/uncertainty_formula.png "Uncertainty formula")

For example, if you have classes [0, 1, 2] and classification probabilities [0.1, 0.2, 0.7], the most likely class according to the classifier is 2 with uncertainty 0.3. If you have three instances with class probabilities

~~~~
proba = np.array([[0.1 , 0.85, 0.05],
                  [0.6 , 0.3 , 0.1 ],
                  [0.39, 0.61, 0.0 ]])
~~~~

the corresponding uncertainties are:
~~~~
1 - proba.max(axis=1)
~~~~
[3]: array([0.15, 0.4 , 0.39])

In the above example, the most uncertain sample is the second one. When querying for labels based on this measure, the strategy selects the sample with the highest uncertainty.

### Classification margin
Another common strategy is called margin sampling or classification margin: in this case, the algorithm selects the instances where the margin between the two most likely labels is narrow, meaning that the classifier is struggling to differentiate between those two most likely classes. 
It is defined by

![Margin formula](/images/margin_formula.png "Margin formula")

where x1^ and x2^ are the first and second most likely classes. Using the same example we used for classification uncertainty, if the class probabilities are

~~~~
proba = np.array([[0.1 , 0.85, 0.05],
                  [0.6 , 0.3 , 0.1 ],
                  [0.39, 0.61, 0.0 ]])
~~~~

the corresponding margins are:
~~~~
part = np.partition(-proba, 1, axis=1)
margin = - part[:, 0] + part[:, 1]
~~~~
~~~~
margin
~~~~
[8]: array([0.75, 0.3 , 0.22])

When querying for labels, the strategy selects the sample with the smallest margin, since the smaller the decision margin is, the more unsure the decision. In this case, it would be the third sample. For this ternary classification problem, the classifier margin plotted against the first two probabilities are the following.

### Classification entropy

The two strategies aim at helping the model discriminate among specific classes and overall do a great job reducing specific classification error. However, if the objective function consists of reducing log-loss, an entropy-based where the learner simply queries the unlabeled instance for which the model has the highest output variance in its prediction, is usually more appropriate. 
It is defined by:

![Entropy formula](/images/entropy_formula.png "Entropy formula")

where pk is the probability of the sample belonging to the kk-th class. Heuristically, the entropy is proportional to the average number of guesses one has to make to find the true class. In our usual example
~~~~
proba = np.array([[0.1 , 0.85, 0.05],
                  [0.6 , 0.3 , 0.1 ],
                  [0.39, 0.61, 0.0 ]])
~~~~
The corresponding entropies are:
~~~~
from scipy.stats import entropy

entropy(proba.T)
~~~~
[12]: array([0.51818621, 0.89794572, 0.66874809])

The closer the distribution to uniform, the larger the entropy.

## Methodology
For the proper execution of this project, it was necessary to structure a specific pipeline. 

![Methodology](/images/methodology.png "Methodology")

### 1- Active learning research
First, it was necessary to understand what active learning is and its advantages, weaknesses, classification and implications. Please visit the "State of the art" section in this readme for more information about the sources used in this project.

### 2- Gather relevant information
This was the phase where the selection of papers helped to define the scope of the project.

### 3- Spanish text dataset
In this phase, a research of spanish text datasets was made, a number of 6 different datasets were found. The selection factor of the dataset was that the data points were labeled manually and a size of the dataset of at least 1000 data points.

#### The dataset
The selected dataset was found in TASS: Workshop on Semantic Analysis at SEPLN page. TASS has been celebrated since 2012, and its original aim was the furthering of the research on sentiment analysis in Spanish. TASS remains to foster sentiment analysis in Spanish, but it wants to promote other tasks related to semantic analysis in Spanish, so the Organization of TASS invites to the research community to proposes new tasks related to semantic analysis in Spanish. 
There are several datasets available for the TASS community and it is necessary to as for permission in order to get access to them. The data was distributed in five different data sets.
*	Spain
*	Peru
*	Costa Rica
*	Uruguay
*	Mexico
These datasets contain Spanish tweets made by people from the referred country. Each one of these have around one thousand tweets in Spanish. There is no specific criteria for the selection of the tweets.
The datasets have the following features:
*	user: Usuer's id
*	content: Body of the tweet
*	date: The date the tweet was twitted.
*	lang: The language of the tweet.
*	sentiment: The predominant sentiment of the tweet. This variable was manually labeled.
For this project these five datasets were concatenated and used as a unique global dataset. The image of the head and tail of the global dataset is shown as follows.

![The original dataset](/images/dataset.png "The original dataset")

### 4- Data cleaning and exploratory analysis

#### Exploratory analysis
Once a proper dataset was obtained and cleaned, the next step was to explore it and find its characteristics. We found out the following.
*	The dataset is complete, no data point in no feature have empty data.
*	It consists in 4800 data points. It includes the five datasets. (MX, ES, CR, PE, UY)
*	Sentiment tags were labeled as: positive (P), negative (N), neutral (NEU), not labeled (NONE).
*	The distribution of sentiment labels are as follows:

![Distribution plot](/images/distribution_1.png  "Distribution plot")
![Distribution dataset](/images/distribution_2.png  "Distribution dataset")

The biggest portion of the dataset is labeled as Negative with 39%, followed by positive  with 29% %. Neutral (14%) and None (18%) will not be considered in the training process.

*	Regarding language distribution, all tweets are in Spanish.
*	The longest tweet has 139 characters.
*	The shortest tweet has 18 characters.
*	The mean tweet length consists in 89 characters.
Also a tokenization process was needed to understand the word frequency. The following plots represent the 20 most frequent words in general and by each sentiment label.

##### All tweets
![](/images/all_plot.png)
![](/images/all_cloud.png)

A we can see, “ir” (Spanish for “go”), “si”, (“yes”) and “hacer” (“do”) are the most common words, because its wide use in language to express a whole variety of concepts.

##### Positive

![](/images/positive_plot.png)
![](/images/positive_cloud.png)

In positive tweets, a presence of wellbeing and gratefulness-related words are the predominant in 20 most common words. 

##### Negative

![](/images/negative_plot.png)
![](/images/negative_cloud.png)

In negative tweets, the predominant use of their 20 most common words are used to express desire.

##### Neutral

![](/images/neutral_plot.png)
![](/images/neutral_cloud.png)

Neutral tweets most common words contain a mix of both negative and positive groups. Laughing expressions are more common in neutral tweets than any other sentiment.

#### Cleaning process
For the data cleaning process it was necessary a to eliminate stopwords, roman and Arabic numbers, lemmatize, eliminate laugh allegories, erase punctuation signs and lowcap every letter. A new feature was added with the cleaned version of each tweet.

![](/images/dataset_2.png)

Then a new dataset with only the features needed to input in the neural network was made. The features were the cleaned tweets and the sentiment labeling. It was also necessary to delete the data points with NONE and NEU labels so the dataset would only have human-labeled negative and positive tweets.

![](/images/dataset_3.png)

### 5- Feature extraction

For the feature extraction TF-IDF vectorizing was used. TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. For this we create vectors of TF-IDF values depending on the total of words in the corpus. If our complete set of words were 37,000, then if a tweet has 89 words, the vectorized representation will have 89 values between 0 and 3, depending in its frequency and 36,911 zeros which represent the lack of use of those words in the tweet. So frequency is a vital factor in the value of each word represented in the vector.
This is a widely used method to vectorize a document to perform many tasks such as ranking and clustering.
It is the base of Google Searching algorithms.
The formula:
TF-IDF(i, j) = TF(i,j) * IDF(i), where:
TF = Term Frequency
IDF = Inverse Document Frequency
TF(i,j) = Term i frequency in document j / Total words in document j
IDF(i) = log(Total documents/documents with term i),

![](/images/tfidf.png)

For this we used the TF-IDF vectorizer function in Scikit-Learn. 

### 6- Creation of a modelwith 100% of labeled data
The first step for the model comparison is the training and testing of a model with 100% of labels. Both the all-tagged model and the 3 active learning models were trained with a sequential multi layered perceptron neural network with the same parameters. 
The neural network for the 4 objects of study are as follows:
*	Input layer (TF-IDF matrix)
*	64 neurons hidden layer
*	Drop out (0.9)
*	64 neurons hidden layer
*	Drop out (0.9)
*	Output layer (y)
All executions had 25 epochs.
The accuracy score was 76.07% and the f1 score was 70.76%. It was considered low but as the accuracy is not an important factor for the goals of this research, it doesn´t represent a disadvantage.

### 7- Creation of models using active learning
The next step was to execute the same neural network 3 times generating 3 different active learning models. For this project the pool-based scenario was selected for the three models because it doesn´t need a specific pipeline to be structured. In the three cases 100 randomly selected datapoints were used for the input in the first execution of the model. The rest of the data points were reserved as the pool. The results were very poor as expected. Then in every new execution, 100 more data points were added to the input. Which 100 of the data points select for a better accuracy and f1 score? That is the difference between the 3 active learning models. Each one of these 3 models had a different querying technique for uncertainty, where the most uncertain values were labeled and added to the input. As explained before these querying techniques for uncertainty are: uncertainty, margin and entropy.  
It required 25 executions of the neural network to complete all the inputs in the pool. Each execution adding the most uncertain 100 data points in the pool as each querying technique identified. Each execution had 25 epochs for a proper comparison between these 3 models and the original model with 100% of labeled data.

### 8- Comparison analysis between original model and active learning models
In the following plots we can see the progression of these 25 executions and their accuracy and F1 scores.

![Accuracy score comparison](/images/accuracy.png "Accuracy score comparison")
![Accuracy score comparison](/images/f1.png "F1 score comparison")

### 9- Results and conclusion

#### Results
As we can see in the comparison plots first 20 the scores in the first 15% of the labeled data were not acceptable, but as the smart selection of data points got labeled the scores started to get closer to the original model results with 100% of labeled data.
As we can see in image 3.17 classification margin querying technique was the first to achieve the same f1 score as the original model with only ~55% of labeled data. The second to achieve this was the classification uncertainty querying technique with ~63%, and classification entropy querying technique at ~72%. Both margin and entropy even got better f1 score than the original model.

#### Conclusions
Active learning techniques can easily help supervised machine learning projects to save time and money in the labeling phase. This can represent a huge budget difference if the labeling process have to be executed by a specialized profile like medics or researchers. The results provided a clear example of how active learning may work in sentiment analysis projects involving Spanish text.
Even though the same results were achieved between the original and the 55% of labeled data using active learning, since 20% of labeled data, active learning models achieved almost the same scores than the original. This may allow in some cases were the scores are favorable or accuracy is not a critical factor, to even label less data.

## State of the art
The following sources were elemental to obtain a clear state of the art vision in active learning and sentiment analysis in Spanish. 

### A Review of Sentiment Analysis in Spanish,  Carlos Henríquez Miranda,  Jaime Guzmán,  Universidad Autónoma, Barranquilla, Colombia
This paper describes a complete pipeline of a sentiment analysis study for tweets in Spanish. It was important to understand the aspects involving Spanish language in a sentiment analysis project.

### Active Learning via Membership Query Synthesis for Semi-supervised Sentence Classification, Raphael Schumann, Ines Rehbein, Institute for Computational Linguistics Heidelberg University, Germany, Leibniz ScienceCampus Heidelberg/Mannheim
This article is an example of an active learning process using Membership Query Sythesis. This was elemental to get the viability of Membership Query Synthesis in the project. This article was important for the decision-making of the election of the pool based sampling scenario.

### Stream-based active learning for sentiment analysis in the financial domain,  JasminaSmailović
Due to the necessity of deciding which scenario is the best for this project it was required to also analyse the possibility of using stream-based active learning. Due to lack of justification, pool-based scenario was selected instead of stream-based.

### Natural Language Processing with Python, Steven Bird, Ewan Klein, Edward Loper, O'Reilly Media, Inc.
This book helped with some technical guidance in the pre-processing phase of the project.

### Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, Aurélien Géron, O'Reilly Media, Inc.
The tutorials of the different methods covered in this book clarified the use and implementation of Keras and Scikit-Learn in this project.

## Bibliography
[1] Introduction to active learning, Jennifer Prendki, https://www.kdnuggets.com/2018/10/introduction-active-learning.html

[2] Uncertainty sampling, modal documentation, 
https://modal-python.readthedocs.io/en/latest/content/query_strategies/uncertainty_sampling.html 

