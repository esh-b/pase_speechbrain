### Steps to run the downstream task
1. Update the `pretrained_path` in the `train.yaml` file to the directory containing the encoder's ckpt file. Also, update the filename in `paths` key of the `pretrainer` parameter accordingly.
1. Once done, the recipe should work just fine.