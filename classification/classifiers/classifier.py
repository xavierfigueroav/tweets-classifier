import joblib
import os

from .preprocesser import preprocess


class LogisticRegressionClassifier:
    def __init__(self):
        module_dir = os.path.dirname(__file__)
        self.models_dir = os.path.join(module_dir, 'models')
        self.load_model()

    def load_model(self):
        model_path = os.path.join(self.models_dir, 'logit.model')
        self.model = joblib.load(model_path)

        tfidf_path = os.path.join(self.models_dir, 'tfidf.model')
        self.tfidf_model = joblib.load(tfidf_path)

    def load_data(self, data):
        ids = []
        tweets = []
        for tweet in data:
            tweet_id = tweet.get('id_str')
            tweet = tweet.get('full_text')
            ids.append(tweet_id)
            tweets.append(tweet)
        return ids, tweets

    def vectorize_tweets(self, data):
        return self.tfidf_model.transform(data).todense()

    def classify(self, tweets):
        ids, tweets = self.load_data(tweets)
        tweets = [preprocess(tweet) for tweet in tweets]

        if len(tweets) == 0:
            return []
        else:
            features = self.vectorize_tweets(tweets)
            labels = self.model.predict(features)
            results = {}
            for id, label in zip(ids, labels):
                results[label] = results.get(label, [])
                results[label].append(id)
            return results
