import os
import torch
import constants
import pandas as pd
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    # This assumes that train.csv is in the same directory as this Python script.
    # You might want to change this.
    data_path = "./data/train.csv"
    all_data_df = pd.read_csv(data_path)
    print("Loaded all data")
    training_df = all_data_df.sample(frac=0.8, ignore_index=True)
    print("Loaded training data")
    # By concatenating the two dataframes together, any duplicates will be in training_df
    # Thus if we get rid of all duplicates, we are left with only the validation data
    val_df = pd.concat([all_data_df, training_df]).drop_duplicates(keep=False)
    print("Loaded validation data")
    # TODO: Training and validation dataset are the same.
    train_dataset = StartingDataset(training_df)
    val_dataset = StartingDataset(val_df)
    model = StartingNetwork()

    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()
