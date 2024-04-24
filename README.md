# Genre-Classification

This repository contains code used for a task of classifying movies scripts into a certain Genre.

## Dataset description 

The dataset contains roughly 18.000 songs in the training set and 8.000 songs in the testing set.

## Task description

The task is to classifiy each melody into a specific genre based on the given lyrics.

## Labels Distribution

- 'Metal': 10.0%
- 'Hip-Hop': 12.0%
- 'Country': 10.0%
- 'Jazz': 8.0%
- 'Electronic': 8.0%
- 'Pop': 14.0%
- 'Folk': 6.0%
- 'Rock': 18.0%
- 'R&B': 6.0%
- 'Indie': 6.0%


## Genres:
- Folk
- Indie
- Rock
- Hip-Hop
- R&B
- Country
- Metal
- Pop
- Jazz
- Electronic

## Experiments Results

No preprocessing 
Validation F1 with BOW and SVC:  0.43
Test F1 with BOW and SVC:  0.43

Standard scaling, class weights
Validation F1 with BOW and SVC:  0.28
Test F1 with BOW and SVC:  0.29

No more scaling, class weights
Validation F1 with BOW and SVC:  0.37
Test F1 with BOW and SVC:  0.37

No more scaling, class weights
Validation F1 with tfidf and SVC:  0.39
Test F1 with tfidf and SVC:  0.40

Xgboost
Validation F1 with tfidf and SVC:  0.42
Test F1 with tfidf and SVC:  0.41


SentenceTrasnformer 'all-MiniLM-L6-v2'
Validation F1 with sent_transf and XGB:  0.41
Test F1 with sent_transf and XGB:  0.40

Text difficulty features and Xgboost model
Validation F1 with tfidf and SVC:  0.33
Test F1 with tfidf and SVC:  0.33

Fasttext model
Train F1 with tfidf and SVC:  0.37
Test F1 with tfidf and SVC:  0.36

Word2vec embeddings with SVC model over them
Train F1 with word2vec embeddings and SVC:  0.34
Test F1 with word2vec embeddings and SVC:  0.33

Text similarity knn like model (calculate average embedding for lyrics that have one label in train, a centroid of the embeddings of all lyrics belonging to that class, followed by determining most similar sentence transformer embedding of the test from the text set to the averages of all texts from the train set that belong to a single table)
Train F1 with tfidf and SVC:  0.37
Test F1 with tfidf and SVC:  0.36

## Conclusions 

To sum up, different approaches were tried and out of all the best results were tried out with a vanilla Support Vector Machines Classifier and using bag of words feature extraction via sklearn's CountVectorizer.

## Future works and developments

Some other things that could be tried out are stylistic markers specyfic to songs such as rhyme, rhythm, length of verses etc.

Perhaps some other transformers architectures could be tried out as well. 

Lastly, definitely vanilla ANNs, CNNs and LSTMs architectures could be tried out having KerasEmbeddings as the vectorization method in mind.

## References and bibliography

- https://radimrehurek.com/gensim/models/word2vec.html
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
- https://xgboost.readthedocs.io/en/stable/python/index.html
- https://github.com/lucianistrati/Word-Complexity-Estimation
- https://github.com/lucianistrati/Matching-Parliamentary-Questions-to-relevant-Ministries
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
  
[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)
