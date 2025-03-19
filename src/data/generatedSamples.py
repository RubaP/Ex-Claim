import os
import time
import pandas as pd
import requests
import uuid

import src.util.Util as Util


# Function to translate a list of texts using Microsoft Translator Text API
def translate_texts(texts, to_lang):
    subscription_key = 'f00ef25f1d2041b7896bb0793aa4744f'  # Replace with your subscription key
    endpoint = 'https://api.cognitive.microsofttranslator.com/'
    path = '/translate?api-version=3.0'
    params = {
        'to': to_lang
    }
    constructed_url = endpoint + path

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': "uksouth",
        'Content-Type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Prepare the request body
    body = [{'text': text} for text in texts]

    # Send the request to the API
    response = requests.post(constructed_url, params=params, headers=headers, json=body)
    if response.status_code != 200:
        print("Error")
        print(response.status_code)
        print(response.text)
    results = response.json()
    # Extract and return the translated texts
    translated_texts = [result['translations'][0]['text'] for result in results]
    return translated_texts


def readCheckThat2022(path, training_languages):
    """
    Read CheckThat 2022 Dataset task files
    :param path: project directory path
    :return: dataset in dictionary format e.g. {language: train }
    """
    dataset = {}
    dataset_dir = path + "Data/" + "CheckThat-2022/"

    for language in training_languages:
        task_dir = dataset_dir + language + "/" + "verifiable-claim-detection"
        if os.path.exists(task_dir):
            train = pd.read_csv(task_dir + "/train.tsv", sep="\t")
            dataset[language] = train

    return dataset


config = Util.readAllConfig()
path = config["path"]
training_languages = ["dutch", "turkish", "english", "arabic", "bulgarian"]
target_language_map = {
    "english": ["en"],
    "arabic": ["ar"],
    "dutch": ["nl"],
    "bulgarian": ["bg"],
    "turkish": ["tr"]
}
dataset_dir = path + "Data/" + "CheckThat-2022/"

# Read dataset
dataset = readCheckThat2022(path, training_languages)

for language in training_languages:
    if language == "dutch":
        continue
    print("----------------------------------------------")
    print("Sampling for language : ", language)
    train = dataset[language]
    target_language = target_language_map[language]

    majority_class_label = train["class_label"].value_counts().idxmax()
    print("Majority class label : ", majority_class_label)
    number_of_samples = abs(train[train["class_label"] == 1].shape[0] - train[train["class_label"] == 0].shape[0])
    print("Number of samples needed: ", number_of_samples)

    other_train_merged = None

    for other_language in training_languages:
        if language == other_language:
            continue

        other_train = dataset[other_language]
        other_train = other_train[other_train["class_label"] == majority_class_label]
        print(f"Majority data from {other_language} size - {other_train.shape[0]}")

        if other_train_merged is None:
            other_train_merged = other_train
        else:
            other_train_merged = pd.concat([other_train_merged, other_train], ignore_index=True)

    print("Merged other training data size - ", other_train_merged.shape[0])

    sampled_data = other_train_merged.sample(number_of_samples)
    print("Sampling completed with size: ", sampled_data.shape[0])

    texts_to_translate = sampled_data['tweet_text'].tolist()
    print("Target language: ", target_language)
    translated_texts = []
    print("Character count: ", sampled_data['tweet_text'].str.len().sum())

    for i in range(0, number_of_samples, 10):
        print(i)
        batch = texts_to_translate[i:i+10]
        translated_texts.extend(translate_texts(batch, target_language))
        time.sleep(10)
    sampled_data["tweet_text"] = translated_texts
    task_dir = dataset_dir + language + "/" + "verifiable-claim-detection"
    sampled_data.to_csv(task_dir + "/sample.tsv", sep="\t", index=False)
