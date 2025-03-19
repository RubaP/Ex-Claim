import logging
import pandas as pd
import src.util.Util as Util
import torch

logger = logging.getLogger("Cross-lingual-claim-detection")


def evaluateTestData(model, model_name, input_vectors, test_data, data_type, path):
    """
    Evaluate the performance of model in given dataset
    :param model: trained model
    :param model_name: name of the trained model
    :param input_vectors: input vectors in dictionary format
    :param test_data: test data
    :param data_type: test or dev_test
    :param path: path to store the results
    """
    device = Util.getDevice()
    model.to(device)
    model.eval()
    model_col = model_name.replace(".pth", "")

    for language in test_data:
        if data_type in test_data[language]:
            logger.info(f"Evaluating {data_type} data from {language}")
            X = input_vectors[language][data_type]

            X = X.to(device)
            with torch.no_grad():
                output = model(X)
            _, predictions = torch.max(output, dim=1)
            predictions_array = predictions.cpu().numpy()
            test_data[language][data_type][model_col] = predictions_array

    data = pd.concat([v[data_type] for v in test_data.values()], ignore_index=True)
    data = data.drop('preprocessed_text', axis=1)
    data.to_csv(path + "Data/All/combined_" + data_type + "_predictions.csv", index=False)


def evaluateTestDataWithEntity(model, model_name, word_embeddings, entity_indexes, test_data, data_type, path):
    """
    Evaluate the performance of model in given dataset
    :param model: trained model
    :param model_name: name of the trained model
    :param word_embeddings: word embeddings in dictionary format
    :param entity_indexes: entity indexes in dictionary format
    :param test_data: test data
    :param data_type: test or dev_test
    :param path: path to store the results
    """
    device = Util.getDevice()
    model.to(device)
    model.eval()
    model_col = model_name.replace(".pth", "")

    for language in test_data:
        if data_type in test_data[language]:
            logger.info(f"Evaluating {data_type} data from {language}")
            WE = word_embeddings[language][data_type]
            E = entity_indexes[language][data_type]

            WE = WE.to(device)
            E = E.to(device)
            with torch.no_grad():
                output = model(WE, E)
            _, predictions = torch.max(output, dim=1)
            predictions_array = predictions.cpu().numpy()
            test_data[language][data_type][model_col] = predictions_array

    data = pd.concat([v[data_type] for v in test_data.values()], ignore_index=True)
    data = data.drop('preprocessed_text', axis=1)
    data.to_csv(path + "Data/All/combined_" + data_type + "_predictions.csv", index=False)