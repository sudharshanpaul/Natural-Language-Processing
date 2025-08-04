import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import numpy as np

def process_tweet(tweet):
    """
    Perform preprocessing on a tweet:
    - Remove RT, hyperlinks, and hashtags (only # symbol)
    - Tokenize
    - Lowercase
    - Remove stopwords and punctuation
    - Stemming
    Returns a list of cleaned, stemmed tokens.
    """
    # Remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # Remove hashtags (only the # symbol)
    tweet = re.sub(r'#', '', tweet)

    # Tokenize
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)

    # Remove stopwords and punctuation
    stopwords_english = set(stopwords.words('english'))
    tokens_clean = [word for word in tokens if word not in stopwords_english and word not in string.punctuation]

    # Stemming
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(word) for word in tokens_clean]

    return tokens_stemmed


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
      for word in process_tweet(tweet):
        pair = (word, y)
        # if pair in freqs:
        #   freqs[pair] += 1
        # else:
        #   freqs[pair] = 1
        freqs[pair] = freqs.get(pair, 0) + 1

    return freqs