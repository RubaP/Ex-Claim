import pandas as pd

languages = ["arabic", "bulgarian", "dutch", "english", "turkish"]
data_name = "checkthat"
path = "D:/Data/All/"
dataset = None

for language in languages:
    test = pd.read_csv(path + data_name + "_" + language + ".tsv", sep="\t")
    test["source"] = data_name
    test["language"] = language

    if dataset is None:
        dataset = test
    else:
        dataset = pd.concat([dataset, test], ignore_index=True)

languages = ["czech", "french", "german", "greek", "gujarati", "hungarian", "italian", "japanese", "kannada",
                      "korean", "marathi", "persian", "russian", "spanish", "swedish", "thai", "ukrainian",
                      "vietnamese"]
data_name = "synthetic"
path = "D:/Data/CheckThat-2022/synthetic/"

for language in languages:
    test = pd.read_csv(path + "verifiable-claim-detection-" + language + ".tsv", sep="\t")
    test["source"] = data_name
    test["language"] = language

    dataset = pd.concat([dataset, test], ignore_index=True)

data_name = "kazemi"
path = "D:/Data/Kazemi-2021/verifiable-claim-detection/"

languages = ["bn", "en", "hi", "ml", "ta"]

for language in languages:
    test = pd.read_csv(path + "kazemi_" + language + ".csv", index_col=False)
    test = test.drop(['Unnamed: 0'], axis=1)
    test["source"] = data_name
    test["language"] = language
    test["topic"] = ""
    test["tweet_id"] = ""
    test["tweet_url"] = ""
    test = test.rename(columns={"text": "tweet_text"})

    dataset = pd.concat([dataset, test], ignore_index=True)


print("Combined test data size: ", dataset.shape)
dataset.to_csv("D:/Data/All/combined_test.csv", index=False)



