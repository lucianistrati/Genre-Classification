Genre Classification 2nd Project Text Mining
Lucian Istrati - Group 511

Dataset description 

The dataset contains roughly 18.000 songs in the training set and 8.000 songs in the testing set.

Task description 

The task is to classifiy each melody into a specific genre based on the given lyrics

Experiments Results

No preprocessing 
Validation F1 with BOW and SVC:  0.4368582938573465
Test F1 with BOW and SVC:  0.4300792043340485

standard scaling, class weights
Validation F1 with BOW and SVC:  0.28702594081392935
Test F1 with BOW and SVC:  0.2920095736837071

no more scaling, class weights
Validation F1 with BOW and SVC:  0.3735202770235661
Test F1 with BOW and SVC:  0.37553606249576055

no more scaling, class weights
Validation F1 with tfidf and SVC:  0.3977945082830646
Test F1 with tfidf and SVC:  0.40514561946587313

xgboost
Validation F1 with tfidf and SVC:  0.4228495468160438
Test F1 with tfidf and SVC:  0.41463699995742037

xgboost (no val data used)
Test F1 with tfidf and SVC:  0.4136205103557636

preprocessing
Test F1 with tfidf and SVC:  0.4236576632571136

SentenceTrasnformer 'all-MiniLM-L6-v2'
Test F1 with sent_transf and XGB:  0.4040942523434254

difficulty features
Test F1 with tfidf and SVC:  0.3386425540725486


Conclusions 

To sum up, different approaches were tried and out of all
Future developments