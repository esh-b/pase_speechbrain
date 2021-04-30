# PASE - Self-supervised learning system
This recipe trains a self-supervised learning system consisting of an encoder and several supervised workers. Once the system is trained, the trained encoder having learned good representations of data have been proven to be useful in various downstream tasks which require large amount of data.

### Directory setup
- `downstream`: This directory provides code to use the trained encoder and evaluate the performance of trained encoder on downstream tasks. Currently, there is only one downstream task: speaker classification.
- `models`: This directory contains model definitions for encoder and workers.
- `qrnn`: The QRNN module used by the encoder (optional).

### Self-supervised system setup
- We train the self-supervised system on minilibrispeech dataset.
- **New configuration variables added in yaml file**:
	- `ckpt_save_interval`: This recipe is coded to store the ckpts at the end of epochs at regular intervals (unlike at regular number of minutes used in other recipes). This variable tells at what interval of epochs the ckpts should be saved. A value of `30` means that ckpts are stored every 30 epochs.
	- `num_ckpts_keep`: The number of ckpts to keep. This limits the number of past ckpts files to store. A value of `5` means the trainer will just store past 5 ckpts files of models and remove all those before that.
	- `chunk_size`: The chunk size used as input to the encoder model. We chunk the original wav file into 1s signals by default in the code.
	- `encoder_lr_start`: The starting lr of the encoder
	- `worker_lr_start`: The starting lr of workers
	- `lr_update_interval`: How often to update the lr of encoder and workers. The code uses LinearScheduler for lr annealing.
	- `decay_factor`: This tells how much to decay the current lr when the lr is being updated.
- **Encoder model configuration**:
	- The encoder supports 3 major configurations:
		- Plain encoder with `SincConv` layers and `Conv1d` layers.
		- Use QRNN in addition to the above setup.
		- Use skip connections to the plain decoder or with (plain decoder + Q-RNN).
- **Addition of new worker**:
	- To add a new worker, the worker model definition needs to be added in the `models` directory.
	- Then, the `{worker}_model` variable needs to be defined in the yaml which is used for model initialization when training starts.
	- Then, the `labeller` class for the worker needs to be defined in `labeller.py`. This class gives the labels when given a batch as input.
	- Later, the `{worker}_labeller` variable needs to be added in the yaml file. This class provides the labels for the batch input during training. These labels are then used to compute loss for the worker.
	- Finally, add the worker in the `workers_config` variable under the `regressor` or `classifier` sections accordingly. The train code will then automatically load the worker during training.

### Steps to run self-supervised training
- Once the hyperparameters are configured, the training can be started by using the command `python train.py train.yaml`. This will download the minilibrispeech dataset if not available in the current directory and then train the encoder on the dataset.
- The encoder ckpts will be saved in the `results` directory according to the configuration in the yaml file.

### References
- https://github.com/santi-pdp/pase
- https://github.com/speechbrain/speechbrain
- https://github.com/salesforce/pytorch-qrnn
