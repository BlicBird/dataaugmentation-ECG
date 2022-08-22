
"""
Trainer is a class that handles the training of the model.
The main function is train().
This is instatiated and called from within apipeline.py.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import time
from tqdm import tqdm
import torch
from torch import float32, optim
from torchmetrics.functional import accuracy, precision, recall, f1_score

class Trainer(object):

    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device

    def get_optimiser(self, model):
        """
            Gets the optimiser and attaches it to the model parameters.
            This uses the "optimiser" key in the config.

            Parameters
            ----------
            model : nn.Module
                The model to attach the optimiser to.

            Raises
            ------
            NotImplementedError
                Unknow optimiser type.
        """
        if not hasattr(self, 'optimiser'):
            if (opt:=self.config.get('optimiser', 'adam')) == 'adam':
                self.optimiser = optim.Adam(model.parameters(), lr=self.config.get('lr', 0.001))
            elif opt=='sgd':
                self.optimiser = optim.SGD(model.parameters(), lr=self.config.get('lr', 0.001))
            else:
                raise NotImplementedError(f'Not implemented or unknown optimiser {opt}.')

    def get_f1n(self, y_pred, target, i, thr=0.5):
        """
            This is the multi-label F1 score defined in the 2018 ICBEB paper.

            Parameters
            ----------
            y_pred : torch.Tensor
                Tensor of shape (Batch, Classes) containing the independent floating predictions.
            target : torch.Tensor
                Tensor of shape (Batch, Classes) containing 0s and 1s. Is is a multi-label binary
                one-hot encoding of the targets.
            i : int
                Which class to calculate the F1 score for.
            thr : float, optional
                The decision threshold, by default 0.5.
                After training, this should be determined separately on a validation set.
                It could also be determined separately for each class.

            Returns
            -------
            float
                The F1 score for class i.
        """
        yb = (y_pred>thr).float()
        n_ix = yb[:,i].sum()
        n_xi = target[:,i].sum()
        n_tp = (yb[:,i]*target[:,i]).sum()

        f1 = 2 * n_tp / ( n_ix + n_xi )
        return f1.item()

    def update_running_history(self, mode, running_history, y_pred, target, loss):
        """
        The running history is a dictionary that contains the stats for the 
        current epoch.
        """
        running_history[f'{mode}_loss'].append(loss.item())
        
        # Overall F1 score
        f_scores = np.array([self.get_f1n(y_pred, target, i) for i in range(target.shape[1])])
        
        for i, f in enumerate(f_scores):
            running_history[f'{mode}_f1_{i}'].append(round(f, 2))

        running_history[f'{mode}_f1'].append(np.mean(f_scores))

        return running_history
    
    def train(self, model, train_dataset, val_dataset, augs):
        """
            The main function of the trainer. Performs a train loop 
            as well as a validation loop.

            Parameters
            ----------
            model : torch.nn.Module
                The model to be fit. Must include a .forward() method.

            train_dataset : torch.utils.data.Dataset
                The dataset to be used for training.

            val_dataset : torch.utils.data.Dataset
                The dataset to be used for validation.

            augs : list
                A list of augmentation instances to be applied to the data.

            Returns
            -------
            history : pd.DataFrame
                A frame with a summary of the training history.
        """

        self.get_optimiser(model)

        model = model.to(self.device)
            
        epochs = self.config.get('epochs', 10)

        # How often to evaluate the validation set.
        evaluation_interval = self.config.get('evaluation_interval', 1)

        history = defaultdict(list)

        # Get the validation data.
        # x_val, y_val = val_dataset.X, val_dataset.y
        # x_val = torch.tensor(x_val, dtype=float32).to(self.device)
        # y_val = torch.tensor(y_val, dtype=float32).to(self.device)
        
        # val_dataloader = torch.utils.data.DataLoader(
        #         val_dataset,
        #         batch_size=self.config.get('batch_size', 64),
        #         shuffle=True,
        #         num_workers=self.config.get('num_workers', 4),
        #         )
        # x_val, y_val = next(iter(val_dataloader))
        # x_val = torch.tensor(x_val, dtype=float32).to(self.device)
        # y_val = torch.tensor(y_val, dtype=float32).to(self.device)
        # y_val = y_val.unsqueeze(1)

        x_val, y_val = val_dataset[:]
        x_val = torch.tensor(x_val, dtype=float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=float32).to(self.device)

        for ep in tqdm(range(epochs), disable=(not self.config.get('verbose', True))):                
            
            # Here we keep track of the stats for each epoch.
            # Running_history is separate from history,
            # because we want to keep track of the stats across batches.
            running_history = defaultdict(list)
            running_times   = defaultdict(float)
            
            ####################
            ##### TRAINING #####
            ####################

            # Set model to train mode.
            model.train()

            # Get a new shuffled dataset
            
            # The dataloader is a generator that yields batches.
            # We iterate over it, and it calls __getitem__ on the dataset.
            # This will yield x_batches, y_batch.
            # I have written it to return numpy arrays, since these are
            # probably easier for certain augmentations.
            dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.get('batch_size', 64),
                shuffle=True,
                num_workers=self.config.get('num_workers', 4),
                )
            
            for x_batch, y_batch in dataloader:

                # y_batch = y_batch.unsqueeze(1)
                
                # Augment the batch.
                t0 = time.time()
                for aug in augs:
                    x_batch, y_batch = aug(x_batch, y_batch)
                running_times['augmentation'] += time.time() - t0; t0 = time.time()

                # Send to tensor and GPU.
                x_batch = torch.tensor(x_batch, dtype=float32).to(self.device)
                y_batch = torch.tensor(y_batch, dtype=float32).to(self.device)
                running_times['to_device'] += time.time() - t0; t0 = time.time()
                
                # Zero gradients
                self.optimiser.zero_grad()
                running_times['zero_grad'] += time.time() - t0; t0 = time.time()

                # Forward pass
                y_pred = model.forward(x_batch)
                running_times['forward'] += time.time() - t0; t0 = time.time()

                # Compute loss
                loss = model.criterion(y_pred, y_batch.float())
                running_times['finding_loss'] += time.time() - t0; t0 = time.time()

                # Backward pass
                loss.backward()
                running_times['backward'] += time.time() - t0; t0 = time.time()

                # Update weights
                self.optimiser.step()
                running_times['optimiser_step'] += time.time() - t0; t0 = time.time()

            
            ######################
            ##### VALIDATION #####
            ######################
            
            # Only evaluate every few epochs.
            if (ep+1)%evaluation_interval==0:

                print('EVALUATION in PROGRESS')

                # Update running history for training.
                # This can also be put into the training section to get
                # average stats across all batches.
                running_history = self.update_running_history('train', running_history, y_pred, y_batch, loss)
                running_times['update_running_history'] += time.time() - t0; t0 = time.time()

                # Set model to eval mode. No need to compute gradients.
                model.eval()
                with torch.no_grad():
                    
                    yv_pred = model.forward(x_val)
                    lossv = model.criterion(yv_pred, y_val)
                    running_history = self.update_running_history('val', running_history, yv_pred, y_val, lossv)

                # History etc is only saved when the validation loop is run.

                # Add the epoch number for ease of plotting later.
                running_history['epoch'].append(int(ep+1))
            
                # Update history
                for key, list_ in running_history.items():
                    history[key].append(np.array(list_).mean())

                # Print epoch
                if self.config.get('verbose', True):
                    print(f'\n\nEpoch {ep+1}/{epochs}\n')

                # Print timings
                if self.config.get('verbose', True):
                    print('----- Timings -----')
                    for key, time_ in running_times.items():
                        print(f'{key.title().replace("_"," ")}: {time_:.2f}s')
                    print()

                # Print history
                if self.config.get('verbose', True):
                    print('----- Performance -----')
                    for key, list_ in history.items():
                        if key.lower() == 'epoch': continue
                        
                        # This block is just used for formatting the output.
                        start = ''
                        if 'f1_' in key:
                            print(f'{start}{key.title().replace("_"," ")}: {list_[-1]:.2f}', end='  ')                            
                        else:
                            if 'f1' in key: start = '\n'
                            print(f'{start}{key.title().replace("_"," ")}: {list_[-1]:.4f}')
                        
                    print()

        history_df = pd.DataFrame(history)
        return history_df