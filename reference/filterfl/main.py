import argparse
import datetime
import random
import numpy as np
import torch
import yaml
import logging
import algorithm
import checkpoint
from handle import Handle

logger = logging.getLogger("logger")


def seed(seeds):
    np.random.seed(seeds)
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    random.seed(seeds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated learning attack and defence')
    parser.add_argument('--params', default='cifar_params.yaml', dest='params')
    parser.add_argument('--defense', default='None', dest='defense')
    parser.add_argument('--resume-folder', default=None, dest='resume_folder',
                         help='an existing saved_models/model_* directory containing ckpt.pt; '
                              'resumes that run instead of starting a new one (params.yaml and '
                              '--defense from that run are reused, --params/--defense are ignored)')
    parser.add_argument('--checkpoint-every', type=int, default=5, dest='checkpoint_every',
                         help='save a resumable checkpoint every N epochs (default 5)')
    parser.add_argument('--skip-stage2', action='store_true', dest='skip_stage2',
                         help='Stage-2 ablation (CODE_NOTES.md): feed Stage 1\'s raw per-class '
                              'image for class 0 directly into Stage 3, skipping gan_defense\'s '
                              'trigger-filtering construction entirely. Default off, unchanged behaviour.')
    parser.add_argument('--degenerate-pair', action='store_true', dest='degenerate_pair',
                         help='load d1 into BOTH discriminators in gan_defense, ignoring d2 '
                              '(G_temp) entirely. Default off, unchanged behaviour.')
    parser.add_argument('--stage2-loss', default='reference', choices=['reference', 'paper'],
                         dest='stage2_loss',
                         help='Stage 2 loss formula (CODE_NOTES.md "Stage 2 loss: paper-faithful '
                              'vs shipped-code"): "reference" (default) keeps gan.py\'s shipped, '
                              'undocumented loss unchanged; "paper" switches to the paper\'s '
                              '3-term (gamma=0.75, lambda=0.125) formulation, dropping the '
                              'shipped 5th term.')
    args = parser.parse_args()

    if args.resume_folder:
        with open(f'{args.resume_folder}/params.yaml', 'r') as f:
            params_loaded = yaml.load(f, Loader=yaml.FullLoader)
        current_time = params_loaded['current_time']
    else:
        with open(f'./{args.params}', 'r') as f:
            params_loaded = yaml.load(f, Loader=yaml.FullLoader)
        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        params_loaded['defence_method'] = args.defense

    params_loaded['checkpoint_every'] = args.checkpoint_every
    params_loaded['skip_stage2'] = args.skip_stage2
    params_loaded['degenerate_pair'] = args.degenerate_pair
    params_loaded['stage2_loss'] = args.stage2_loss
    seed(params_loaded['seed'])
    logger.info(f'Start training')
    handle = Handle(current_time=current_time, params=params_loaded,
                    name=params_loaded.get('name', format(params_loaded['type'])),
                    resume_folder=args.resume_folder)
    handle.load_data()
    logger.info(f'load data done')
    handle.create_model()
    logger.info(handle.model)
    logger.info(f'create model done')
    if handle.params['is_poison']:
        logger.info(f"Poisoned following participants: {handle.adversarial_namelist}")

    if args.resume_folder:
        ckpt_path = f'{handle.folder_path}/ckpt.pt'
        ckpt = checkpoint.load(ckpt_path, handle.model)
        if list(handle.adversarial_namelist) != ckpt['adversarial_namelist']:
            raise RuntimeError(
                'resumed adversarial_namelist does not match the checkpoint -- the data '
                'pipeline is not reproducing the original run deterministically, refusing '
                'to continue with a mismatched poisoning setup')
        handle.start_epoch = ckpt['epoch'] + 1
        handle.params['lr'] = ckpt['lr']
        handle.params['poison_lr'] = ckpt['poison_lr']
        handle.count_all_attack = ckpt['count_all_attack']
        handle.count_all_defense = ckpt['count_all_defense']
        handle.count_correct_defense = ckpt['count_correct_defense']
        logger.info(f'resumed from {ckpt_path} at epoch {handle.start_epoch}')

    with open(f'{handle.folder_path}/params.yaml', 'w') as f:
        yaml.dump(handle.params, f)
    if handle.params['algorithm'] == 'FedAvg':
        algorithm.FedAvg(handle)
