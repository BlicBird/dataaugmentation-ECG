
"""
This is the main file, from which the other files are called.
Here, we run load the data, create the ML model, train the model using 
a specified set of augmentations, and save the results.

This code is provided as a starting point in order to quickly test your 
augmentations. This is because the focus should be on the augmentations, 
rather than the model itself. Feel completely free to change this code, 
or even scrap it entirely if you have your own way of doing things.

To understand this file, you should open model.py, which contains the 
class definition for the model, created in PyTorch, trainer.py, which 
trains the model, dataset.py, which contains the class definition for
the dataset (which inherits from the standard PyTorch Dataset class),
augmentations.py, which contains the class definitions for the
augmentations, and config.py, which contains the configuration file.

In have written extensive comments in the code to help understand this
Please read these and the docstrings, but do not hesitate to reach out
if you have any questions.
"""

# Standard imports
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from tqdm import tqdm

# Other imports
import torch
from datetime import datetime
import argparse
import hjson as json
import itertools
import importlib
from  torch.utils.data import DataLoader

# Custom classes
# Note that we do not import the augmentations here.
# This is because the augmentations are imported dynamically, 
# since they are specified in the config.
# from dataset import ECGDataset
from model import ChenModel
from trainer import Trainer
from dataloader_segment import ECGDataset
from dataloader_fake import ECGDataset_fake

# This reduces unnecessary warnings.
import warnings
warnings.simplefilter(action='default')

# The root to save the outputs.
RESULTS_PATH = '/sdb1/results/augmentation/'

# This indicates the GPU, and is used to run the model on the GPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    """
        Loads the arguments from the terminal

        Examples
        --------
        $ python3 apipeline.py -c test_config -n test_name
        
        Returns
        -------
        argparser.parse_args() : dict
            dictionary containing the key and values of terminal arguments
    """
    argparser = argparse.ArgumentParser(description='Experiment Configuration')
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The name of the configuration file.')

    argparser.add_argument(
        '-n', '--name',
        dest='name',
        metavar='N',
        default='None',
        help='Name of the experiment.')

    return argparser.parse_args()

def expand_dict(dict_in, remove=[]):
    """
        Takes a dictionary with lists as elements, and expands this into a list of dictionaries.

        Parameters
        ----------
        dict_in : dict
            The input dictionary. Each element should be a list.

        remove : list, optional
            Keys to remove from the dictionary.
            For example, keys, k, for which dict_in[k] is not a list. By default []

        Returns
        -------
        list    
            A list of dictionaries.

        Examples
        --------
        >>> d = {'a':[1,2], 'b':[3,4]}
        >>> expand_dict(d)
        [{'a':1, 'b':3}, {'a':1, 'b':4}, {'a':2, 'b':3}, {'a':2, 'b':4}]
    """
    out = []
    for k in remove:
        if k in dict_in.keys():
            dict_in.pop(k)
    keys = list(dict_in.keys())
    product = itertools.product(*(dict_in[p] for p in keys))
    for item in product:
        out.append({
            keys[i]:item[i]
            for i in range(len(keys))
            if keys[i] not in remove
            })
    return out

def get_class(path):
    """
        This creates a class dynamically from a string.

        Parameters
        ----------
        path : str
            Path to class. This is written as module.class_name, 
            or directory.module.class_name if the module is in a subdirectory.
            This path is relative to the project root directory.
            
            If this does not work, make sure to add this project root folder
            to your Python path. Also, if you are using VSCode, make sure to
            add the project root folder to your workspace.

        Returns
        -------
        Object
            The class object.

        Examples
        --------
        >>> Rotate = get_class('augmentations.RandomRotate')
        >>> rotator = Rotate()
        >>> rotator.__class__
        <class 'augmentations.RandomRotate'>
        >>> x_rotated = rotator(x)
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_aug_combinations(full_aug_config):
    """
        This takes the full augmentation configuration, and expands it into a
        list of list of augmentation class instances.

        The argument full_aug_config is given by config['augmentations'].
        This is a list of dictionaries, each of which contains key "class_name", 
        and optional other keys. The values of these other keys are lists of values.

        Looking just at a single augmentations, this function expands the 
        augmentation configuration into a list of instances. Each instance
        corresponds to one combination of the elements in the value lists.

        This sounds confusing, but is better explained with an example.

        Suppose we have the following configuration:
        
        full_aug_config = [
            {
                "class_name": "augmentations.RotateAndTranslate",
                "rotate": [0, 90],
                "translate": [0, 1]
            },
        ]

        The output of this function is then:
        all_aug_class_expanded_list = [
                [ RotateAndTranslate(rotate=0, translate=0) ],
                [ RotateAndTranslate(rotate=0, translate=1) ],
                [ RotateAndTranslate(rotate=90, translate=0) ],
                [ RotateAndTranslate(rotate=90, translate=1) ],
        ]

        If our initial config contains two classes, then each sublist in the
        output corresponds to a combination of the two classes. For example, we 
        could repeat the above example with Rotate and Translate being two classes.

        full_aug_config = [
            {
                "class_name": "augmentations.Rotate",
                "rotate": [0, 90]
            },

            {
                "class_name": "augmentations.Translate",
                "translate": [0, 1]
            },
        ]

        The output of this function is then:
        all_aug_class_expanded_list = [
            [ Rotate(rotate=0), Translate(translate=0) ],
            [ Rotate(rotate=0), Translate(translate=1) ],
            [ Rotate(rotate=90), Translate(translate=0) ],
            [ Rotate(rotate=90), Translate(translate=1) ],
        ]

        These are applied to the input data in the order they appear in the list 
        within the trainer.

        If an augmentation takes no options (such as augmentations.Identity), then just the class name is given.
        For example:

        full_aug_config = [
            {
                "class_name": "augmentations.Rotate",
                "rotate": [0, 90]
            },

            {
                "class_name": "augmentations.Identity",
            },
        ]

        will lead to:
        all_aug_class_expanded_list = [
            [ Rotate(rotate=0), Identity() ],
            [ Rotate(rotate=0), Identity() ],
            [ Rotate(rotate=90), Identity() ],
            [ Rotate(rotate=90), Identity() ],
        ]

        Note that we instantiate the class with **subconfig, so the names
        of the arguments should be the same as the keys in the subconfig.
    """

    all_aug_class_sublists = []
    all_aug_class_expanded_list = []

    for config in full_aug_config:

        augs_for_one_class = []

        class_name = config.pop('class_name')
        aug_class = get_class(class_name)

        if len(config.keys()) > 0:
            sub_conf_list = expand_dict(config)
            for sub_conf in sub_conf_list:
                augs_for_one_class.append( aug_class(**sub_conf) )
        else:
            augs_for_one_class.append( aug_class() )

        all_aug_class_sublists.append(augs_for_one_class)
    
    all_aug_class_expanded_list = list(itertools.product(*all_aug_class_sublists))

    return all_aug_class_expanded_list

def main(config=None, args=None):
    
    # Get the command-line arguments.
    # These are -c and -n, for the configuration 
    # file and the name of the experiment.
    if not args:
        args = get_args()

    # Set the experiment name.
    if args.name != 'None':
        experiment_name = args.name
    else:
        experiment_name = args.config

    # Create a unique experiment name based on the given name and time.
    experiment_name = experiment_name + '_' + datetime.now().strftime(r"%Y%m%d_%H%M%S")
    print(f"Experiment: {experiment_name}")
    os.makedirs(opj(RESULTS_PATH, experiment_name))

    # Read the config and save it to the results folder.
    with open(f'./configs/{args.config}.json', 'r') as fopen:
        config = json.load(fopen)
    with open(f'{RESULTS_PATH}{experiment_name}/config.json', 'w') as fopen:
        json.dump(config, fopen)

    # Create instances of the classes, and get the combination of augmentations.
    model = ChenModel(config['model'])
    trainer = Trainer(config['train'], device=DEVICE)
    aug_combinations = get_aug_combinations(config['augmentations'])

    # Load the data.
    print('Loading data...') 
    # train_dataset = ECGDataset(config['data'], '/sdb1/china_signal_challenge/', folds=[0,1,2,3,4,5,6,7])
    # val_dataset   = ECGDataset(config['data'], '/sdb1/china_signal_challenge/', folds=[8,9])
     
    path = '/home/danielsantosh/intial_folder/china_signal_challenge'
    data = ECGDataset(path,10000)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    path_aug = '/home/danielsantosh/data_augmentation/fake_ecg/fake_ecg_DCGAN_base_200e_classimb_triple.pkl'
    data_fake = ECGDataset_fake(path_aug)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset, data_fake])

    
    for i_a, augs in enumerate(aug_combinations):
        print(f'\n\nAugmentation: {i_a}\nTraining...')

        history = trainer.train(model, train_dataset, val_dataset, augs)

        history.to_pickle('/home/danielsantosh/data_augmentation/classification_results/DCGAN_base_5000e_classimb_triple.pkl')
        
        history.to_csv(f'{RESULTS_PATH}{experiment_name}/history_{i_a}.csv', index=False)
        
        print('\n', history)

    print(f'Experiment: {experiment_name} complete.')
    return

if __name__=='__main__':
    main()
    print('done')

