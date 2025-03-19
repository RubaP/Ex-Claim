import re
import logging

url_pattern = re.compile(r'https?://\S+|www\.\S+')
logger = logging.getLogger("Cross-lingual-claim-detection")


def preprocessTweets(dataset):
    """
    Preprocess the complete dataset
    :param dataset: dataset
    :return: processed dataset in dictionary format
    """
    logger.info("Processing started")
    for language in dataset:
        for data_type in dataset[language]:
            df = dataset[language][data_type]
            df['preprocessed_text'] = df['tweet_text'].apply(lambda x: processTweetText(x))
            dataset[language][data_type] = df

    logger.info("Preprocessing completed")
    return dataset


def processTweetText(tweet):
    """
    Preprocess a tweet text by replacing URL with special token URL
    :param tweet: tweet text
    :return: preprocessed tweet text
    """
    return re.sub(url_pattern, "@link", tweet.strip())
