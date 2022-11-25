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
