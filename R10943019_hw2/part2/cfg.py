

## You could add some configs to perform other training experiments...

LeNet_cfg = {
    'model_type': 'LeNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}

efficient_cfg = {
    'model_type': 'efficient',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}
efficient1_cfg = {
    'model_type': 'efficient1',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}
efficient4_cfg = {
    'model_type': 'efficient4',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}

efficient7_cfg = {
    'model_type': 'efficient7',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}
ResNet_cfg = {
    'model_type': 'ResNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}