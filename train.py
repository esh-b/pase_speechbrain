#!/usr/bin/env python3
"""Recipe for training a speaker-id system. The template can use used as a
basic example for any signal classification task such as language_id,
emotion recognition, command classification, etc. The proposed task classifies
28 speakers using Mini Librispeech. This task is very easy. In a real
scenario, you need to use datasets with a larger number of speakers such as
the voxceleb one (see recipes/VoxCeleb). Speechbrain has already some built-in
models for signal classifications (see the ECAPA one in
speechbrain.lobes.models.ECAPA_TDNN.py or the xvector in
speechbrain/lobes/models/Xvector.py)

To run this recipe, do the following:
> python train.py train.yaml

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Mirco Ravanelli 2021
"""
import os
import sys
import torch
import speechbrain as sb
from speechbrain import Stage
from hyperpyyaml import load_hyperpyyaml
from mini_librispeech_prepare import prepare_mini_librispeech
from utils import flatten_dict


class PASEBrain(sb.Brain):
    workers_cfg = {}

    def __init__(
        self,
        encoder_cfg,
        workers_cfg,
        modules=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        modules = modules or {}
        modules.update(self._add_modules(encoder_cfg, workers_cfg, checkpointer))

        super().__init__(
            modules=modules,
            opt_class=None,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )

    def _add_modules(self, encoder_cfg, workers_cfg, checkpointer):
        encoder_worker_modules = {}

        for w_type, w_list in workers_cfg.items():
            for w_name, w_cfg in w_list.items():
                if 'model' not in w_cfg:
                    raise ValueError(f'Expected a model definition for worker {w_name}')
                encoder_worker_modules[f'{w_name}'] = w_cfg['model']
                if 'labeller' not in w_cfg:
                    raise ValueError(f'Expected a model definition for worker {w_name}')
                encoder_worker_modules[f'{w_name}_labeller'] = w_cfg['labeller']
                self.workers_cfg[w_name] = {'type': w_type}
                checkpointer.add_recoverable(w_name, w_cfg['model'])

        if not encoder_worker_modules:
            raise ValueError('Expected atleast one worker')

        if 'model' not in encoder_cfg:
            raise ValueError('Expected a model definition for the encoder')
        encoder_worker_modules['encoder'] = encoder_cfg['model']
        checkpointer.add_recoverable('encoder', encoder_cfg['model'])

        return encoder_worker_modules

    def init_optimizers(self):
        self.encoder_optim = self.hparams.encoder_config['optimizer'](self.modules['encoder'].parameters())

        for w_type, w_list in self.hparams.workers_config.items():
            for w_name, w_cfg in w_list.items():
                self.workers_cfg[w_name]['optim'] =  w_cfg['optimizer'](self.modules[w_name].parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable('encoder_optim', self.encoder_optim)
            for name, cfg in self.workers_cfg.items():
                self.checkpointer.add_recoverable(f'{name}_optim', cfg['optim'])

    def init_workers_losses(self):
        for w_type, w_list in self.hparams.workers_config.items():
            for w_name, w_cfg in w_list.items():
                self.workers_cfg[w_name]['loss'] = getattr(torch.nn, w_cfg['loss'])()

    def on_fit_start(self):
        super().on_fit_start()

        self.init_workers_losses()

    def fit_batch(self, batch):
        # Managing automatic mixed precision
        # if self.auto_mix_prec:
        #     with torch.cuda.amp.autocast():
        #         outputs = self.compute_forward(batch, Stage.TRAIN)
        #         loss = self.compute_objectives(outputs, batch, Stage.TRAIN)
        #         self.scaler.scale(loss).backward()
        #         if self.check_gradients(loss):
        #             self.scaler.step(self.optimizer)
        #         self.optimizer.zero_grad()
        #         self.scaler.update()

        outputs = self.compute_forward(batch, Stage.TRAIN)  # outputs = (h, chunk, preds, labels)
        losses = self.compute_objectives(outputs, batch, Stage.TRAIN)

        losses['avg'].backward()

        if self.check_gradients(losses['avg']):
            for w_name, w_cfg in self.workers_cfg.items():
                w_cfg['optim'].step()
            self.encoder_optim.step()

        # self.encoder_optim.zero_grad()

        # return losses.detach().cpu()
        return losses['avg']

    def evaluate_batch(self, batch, stage):
        out = self.compute_forward(batch, stage=stage)
        losses = self.compute_objectives(out, batch, stage=stage)
        return losses['avg']

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules['encoder'](feats)

        preds = {}
        for name in self.workers_cfg:
            preds[name] = self.modules[name](embeddings)
        return preds

    def prepare_features(self, wavs, stage):
        wavs, lens = wavs
        # if wavs.dim() == 2:
        #     wavs = wavs.unsqueeze(2)

        return wavs, lens

    def compute_objectives(self, predictions, batch, stage):
        preds = predictions
        labels = {
            'decoder': self.modules.decoder_labeller(batch.sig[0]),
            'mfcc': self.modules.mfcc_labeller(batch.sig[0])[:, :100, :],
        }

        total_loss = 0
        losses = {}

        # if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
        #     spkid = torch.cat([spkid, spkid], dim=0)
        #     lens = torch.cat([lens, lens])

        self.encoder_optim.zero_grad()

        for name in self.workers_cfg:
            self.workers_cfg[name]['optim'].zero_grad()
            loss = self.workers_cfg[name]['loss'](preds[name], labels[name])
            losses[name] = loss
            total_loss += loss

        losses["avg"] = total_loss / len(self.workers_cfg)
        print('.....', losses)
        return losses

    def _update_optimizers_lr(self, epoch):
        old_lr, new_lr = self.hparams.lr_annealing['encoder'](epoch)
        sb.nnet.schedulers.update_learning_rate(self.encoder_optim, new_lr)

        old_lr, new_lr = self.hparams.lr_annealing['workers'](epoch)
        for _, cfg in self.workers_cfg.items():
            sb.nnet.schedulers.update_learning_rate(cfg['optim'], new_lr)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

            if epoch % self.hparams.halved_epochs == 0:
                self._update_optimizers_lr(epoch)

            self.checkpointer.save_and_keep_only(meta=stage_stats, min_keys=["loss"])


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json`, `valid.json`,  and `valid.json` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
          sig = sb.dataio.dataio.read_audio(wav)
          sig = sig[100:16100]
          yield sig

    datasets = {}
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["sig"],
        )
    return datasets


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_mini_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "split_ratio": [80, 10, 10],
        },
    )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    pase_brain = PASEBrain(
        encoder_cfg=hparams['encoder_config'],
        workers_cfg=hparams['workers_config'],
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    pase_brain.fit(
        epoch_counter=pase_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = pase_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
