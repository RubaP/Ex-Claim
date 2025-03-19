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
    dataset = None
    dataset_dir = path + "Data/" + "CheckThat-2022/"

    for language in training_languages:
        task_dir = dataset_dir + language + "/" + "verifiable-claim-detection"
        if os.path.exists(task_dir):
            test = pd.read_csv(task_dir + "/test_gold.tsv", sep="\t")

            if dataset is None:
                dataset = test
            else:
                dataset = pd.concat([dataset, test], ignore_index=True)

    print("Total number of test instances: ", dataset.shape)
    return dataset


config = Util.readAllConfig()
path = config["path"]
training_languages = ["dutch", "turkish", "english", "arabic", "bulgarian"]
target_language_map = {
    "czech": ["cs"],
    "french": ["fr"],
    "german": ["de"],
    "greek": ["el"],
    "gujarati": ["gu"],
    "hungarian": ["hu"],
    "italian": ["it"],
    "japanese": ["ja"],
    "kannada": ["kn"],
    "korean": ["ko"],
    "marathi": ["mr"],
    "persian": ["fa"],
    "russian": ["ru"],
    "spanish": ["es"],
    "swedish": ["sv"],
    "thai": ["th"],
    "ukrainian": ["uk"],
    "vietnamese": ["vi"]
}
dataset_dir = path + "Data/" + "CheckThat-2022/synthetic/"

# Read dataset
dataset = readCheckThat2022(path, training_languages)
number_of_samples = 250

for language, code in target_language_map.items():
    print("Sampling for language : ", language)

    sampled_data = dataset.sample(number_of_samples)
    print("Sampling completed with size: ", sampled_data.shape[0])

    texts_to_translate = sampled_data['tweet_text'].tolist()

    translated_texts = []
    print("Character count: ", sampled_data['tweet_text'].str.len().sum())

    for i in range(0, number_of_samples, 10):
        print(i)
        batch = texts_to_translate[i:i+10]
        translated_texts.extend(translate_texts(batch, code))
        time.sleep(10)
    sampled_data["tweet_text"] = translated_texts
    sampled_data.to_csv(dataset_dir + "verifiable-claim-detection-" + language + ".tsv", sep="\t", index=False)
