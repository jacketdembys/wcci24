import yaml
import os


layers = 1
neurons = 10

# read from path script
for scale in range(1,11,1):

    # build the content of the config file in a dictionary
    config_info = {
            'NUM_EXPERIMENT_REPETITIONS': 1,
            'ROBOT_CHOICE': '6DoF-6R-Puma260',
            'SEED_CHOICE': True,
            'SEED_NUMBER': 0,
            'DEVICE_ID': 0,
            'MODEL': {
                'NAME': 'MLP',
                'NUM_HIDDEN_LAYERS': layers,          
                'NUM_HIDDEN_NEURONS': neurons
            },             
            'TRAIN': {
                'DATASET': {
                    'NUM_SAMPLES': 10000,
                    'JOINT_LIMIT_SCALE': int(scale)
                },
                'CHECKPOINT': {
                    'PRETRAINED_G_MODEL': "",
                    'RESUMED_G_MODEL': "",
                },
                'HYPERPARAMETERS': {
                    'EPOCHS': 100000,
                    'BATCH_SIZE': 128,
                    'SHUFFLE': True,
                    'NUM_WORKERS': 4,
                    'PIN_MEMORY': True,
                    'PERSISTENT_WORKERS': True,
                    'OPTIMIZER_NAME': 'SGD',
                    'LEARNING_RATE': 0.0001,
                    'BETAS': [0.9, 0.999],
                    'EPS': 0.0001,
                    'WEIGHT_DECAY': 0.0,
                    'WEIGHT_INITIALIZATION': 'default',
                    'LOSS': 'l2',
                },
                'PRINT_EPOCHS': True,
                'PRINT_STEPS': 100
            },
    }


    save_path = "config_layers_"+str(int(layers))+"_neurons_"+str(int(neurons))
    if not os.path.exists(save_path):
                os.makedirs(save_path)

    # open a yaml file and dump the content of the dictionary 
    with open(save_path+"/train_scale_"+str(int(scale))+".yaml", 'w') as yamlfile:
        data = yaml.dump(config_info, yamlfile)
        print("Successfully created config files in {}!".format(save_path))