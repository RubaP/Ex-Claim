import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from tqdm import tqdm
import logging
import src.util.Util as Util

logger = logging.getLogger("Cross-lingual-claim-detection")


class DatasetWithEntity(Dataset):
    def __init__(self, word_embedding, entity_types, labels):
        self.word_embedding = word_embedding
        self.entity_types = entity_types
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        we = self.word_embedding[idx]
        e = self.entity_types[idx]
        label = self.labels[idx]
        return we, e, label


def getDataLoader(input_vectors, class_labels, languages, data_types, batch_size, shuffle):
    """
    Get data loader for the dataset
    :param input_vectors: input vectors in dictionary format
    :param class_labels: class labels in dictionary format
    :param languages: list of languages to consider for data creation
    :param data_types: list of data types to consider for data creation
    :param batch_size: batch_size required for the training
    :param shuffle: boolean indicating whether the data should be shuffled or not
    :return: DataLoaded object
    """
    logger.info(f"Creating tensor data using {data_types} of {languages}")

    X = []
    Y = []

    for language in class_labels:
        if language in languages:
            for data_type in class_labels[language]:
                if data_type in data_types:
                    X.append(input_vectors[language][data_type])
                    Y.append(class_labels[language][data_type])

    X_tensor = torch.cat(X, 0)
    Y_tensor = torch.cat(Y, 0)
    logger.info(f"Tensor dataset size - {X_tensor.size()}")

    tensor_data = TensorDataset(X_tensor, Y_tensor)
    data_loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return data_loader


def getDataLoaderWithEntityData(word_embedding, entity_types, class_labels, languages, data_types, batch_size, shuffle):
    """
    Get data loader for the dataset
    :param word_embedding: input vectors in dictionary format
    :param entity_types: entity types
    :param class_labels: class labels in dictionary format
    :param languages: list of languages to consider for data creation
    :param data_types: list of data types to consider for data creation
    :param batch_size: batch_size required for the training
    :param shuffle: boolean indicating whether the data should be shuffled or not
    :return: DataLoaded object
    """
    logger.info(f"Creating tensor data using {data_types} of {languages}")
    device = Util.getDevice()

    WE = []
    E = []
    Y = []

    for language in class_labels:
        if language in languages:
            for data_type in class_labels[language]:
                if data_type in data_types:
                    WE.append(word_embedding[language][data_type])
                    E.append(entity_types[language][data_type])
                    Y.append(class_labels[language][data_type])

    WE_tensor = torch.cat(WE, 0)
    E_tensor = torch.cat(E, 0)
    Y_tensor = torch.cat(Y, 0)

    # Mandatory to reassign the moved tensor
    WE_tensor = WE_tensor.to(device)
    E_tensor = E_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    tensor_data = DatasetWithEntity(WE_tensor, E_tensor, Y_tensor)
    data_loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return data_loader


def getOptimizer(model, learning_rate):
    """
    Get torch Optimizer
    :param model: model to be trained
    :param learning_rate: learning rate
    :return: optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer


def getScheduler(optimizer, train_loader, epochs):
    """
    Get Scheduler
    :param optimizer: optimizer
    :param train_loader: training loader
    :param epochs: number of epochs
    :return: scheduler
    """
    total_steps = len(train_loader)*epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps//2, gamma=0.1)
    return scheduler


def getLossCriterion():
    """
    Get a loss criteria
    :return: criterion
    """
    criterion = nn.CrossEntropyLoss()
    return criterion


def getInputSize(input_vectors):
    input_size = 0

    for language in input_vectors:
        for data_type in input_vectors[language]:
            input_size = input_vectors[language][data_type].size(-1)
            break

        if input_size > 0:
            break

    logger.info(f"Training input size - {input_size}")
    return input_size


def trainModelWithEntityData(model, train_loader, validation_loader, learning_rate, epochs, best_model_name):
    logger.info("Model training started")
    device = Util.getDevice()
    model.to(device)
    criterion = getLossCriterion()
    criterion.to(device)
    optimizer = getOptimizer(model, learning_rate)
    train_loss = []
    validation_loss = []

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for we, e, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(we, e)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_training_loss = total_loss / len(train_loader)
        average_validation_loss = getValidationLossWithEntityData(model, validation_loader, criterion)

        # Save best model with lower validation loss
        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            torch.save(model.state_dict(), best_model_name)
            logger.info(f"Best model saved at epoch {epoch + 1} with the best loss {average_validation_loss:.4f}")

        logger.info(f"Epoch {epoch + 1}/{epochs}, Avg Training Loss: {average_training_loss:.4f}, "
                    f"Avg Validation Loss: {average_validation_loss:.4f}, ")
        train_loss.append(average_training_loss)
        validation_loss.append(average_validation_loss)

    logger.info(f"Training loss list: {train_loss}")
    logger.info(f"Validation loss list: {validation_loss}")


def trainModel(model, train_loader, validation_loader, learning_rate, epochs, best_model_name):
    logger.info("Model training started")
    device = Util.getDevice()
    model.to(device)
    criterion = getLossCriterion()
    criterion.to(device)
    optimizer = getOptimizer(model, learning_rate)
    train_loss = []
    validation_loss = []

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            optimizer.zero_grad()

            # Forward pass
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_training_loss = total_loss / len(train_loader)
        average_validation_loss = getValidationLoss(model, validation_loader, criterion)

        # Save best model with lower validation loss
        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            torch.save(model.state_dict(), best_model_name)
            logger.info(f"Best model saved at epoch {epoch + 1} with the best loss {average_validation_loss:.4f}")

        logger.info(f"Epoch {epoch + 1}/{epochs}, Avg Training Loss: {average_training_loss:.4f}, "
                    f"Avg Validation Loss: {average_validation_loss:.4f}, ")
        train_loss.append(average_training_loss)
        validation_loss.append(average_validation_loss)

    logger.info(f"Training loss list: {train_loss}")
    logger.info(f"Validation loss list: {validation_loss}")


def getValidationLoss(model, validation_loader, criterion):
    """
    Get validation data loss
    :param model: trained model
    :param validation_loader: data loader for validation data
    :param criterion: criterion
    :return: validation data loss
    """

    model.eval()
    validation_loss = 0.0
    device = Util.getDevice()

    with torch.no_grad():
        for val_inputs, val_targets in validation_loader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            val_outputs = model(val_inputs)
            validation_loss += criterion(val_outputs, val_targets).item()

    validation_loss /= len(validation_loader)

    return validation_loss


def getValidationLossWithEntityData(model, validation_loader, criterion):
    """
    Get validation data loss
    :param model: trained model
    :param validation_loader: data loader for validation data
    :param criterion: criterion
    :return: validation data loss
    """

    model.eval()
    validation_loss = 0.0

    with torch.no_grad():
        for val_we, val_e, val_targets in validation_loader:
            val_outputs = model(val_we, val_e)
            validation_loss += criterion(val_outputs, val_targets).item()

    validation_loss /= len(validation_loader)

    return validation_loss


def loadModel(model, path):
    """
    Load saved model
    :param model: model template
    :param path: path to saved model
    :return: pytorch model
    """
    model.load_state_dict(torch.load(path))
    logger.info(f"Successfully loaded model from {path}")
    return model
