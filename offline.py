

#Offline Training
# INPUT: An autoencoder=(encoder/decoder), dataset=(train/validation) sets, and hyperparameters
#        Note for this training we train on a large volume of complete videos, this is the pre-training
# OUTPUT: Trained autoencoder, graph / history of training
def train_offline(autoencoder, dataset, settings, board):
    #Must make sure to evenly evaluate all autoencoder.enc_sizes to ensure it functions for all of them
    pass


#This function will parse terminal inputs from the user and then perform offline training
def main_offline():
    pass


if __name__ == "__main__":
    main_offline()