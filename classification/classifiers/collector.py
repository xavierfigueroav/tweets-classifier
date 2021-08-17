import csv
import os


def collect_tweets():
    tweets = []
    module_dir = os.path.dirname(__file__)
    tweets_path = os.path.join(module_dir, 'data', 'tweets.csv')
    with open(tweets_path) as csv_file:
        reader = csv.reader(csv_file)
        for id, text in reader:
            url = f'https://twitter.com/twitter/statuses/{id}'
            tweet = {'id_str': url, 'full_text': text}
            tweets.append(tweet)
    return tweets[:50]
