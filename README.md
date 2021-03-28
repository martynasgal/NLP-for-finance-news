# NLP-for-finance-news
NLP for Finance News

Two csv files are provided: train_data.csv and test_data.csv. Each of them contains two columns “Sentence” and “Bad Sentence”. The “Sentence” column contains sentences coming from financial news streams. The “Bad Sentence” column is just a flag that is 1 when the sentence is a bad sentence (i.e., does not report any news by itself) and 0 when is actually reporting something. You have to create a program calls “clean_news.py” that uses the training data to train an XGBClassifier model with a TfidfVectorizer and then uses the resulting model to predict bad sentences on the test data:

clean_news.py --train train_data.csv –-test test_data.csv –-pred pred_data.csv  --max_feat 100 –-num_folds 10

The program shouldn’t write anything to either train_data.csv or test_data.csv. It will use the labels in train_data.csv for training and the labels in test_data.csv to measure the test error only. It will also create a file called pred_data.csv  that has the same sentences as in test_data.csv (“Sentence” column), but will have a “Predicted Bad Sentence” column with the prediction for each of the 1000 test sentences. max_feat is the maximum number of features (max_features) used  by TfidfVectorizer and num_folds is the number of k-folds used for cross validation. The program should report the train F1 score and the test F1 score.
