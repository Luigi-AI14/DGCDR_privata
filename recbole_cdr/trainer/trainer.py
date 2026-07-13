
r"""
recbole_cdr.trainer.trainer
################################
"""

import numpy as np
import torch
from recbole.trainer import Trainer
from recbole_cdr.utils import train_mode2state


class CrossDomainTrainer(Trainer):
    r"""Trainer for training cross-domain models. It contains four training mode: SOURCE, TARGET, BOTH, OVERLAP
    which can be set by the parameter of `train_epochs`
    """

    def __init__(self, config, model):
        super(CrossDomainTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
            else:
                super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result


class DCDCSRTrainer(Trainer):
    r"""Trainer for training DCDCSR models."""

    def __init__(self, config, model):
        super(DCDCSRTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if scheme == 'BOTH':
                super().fit(train_data, None, verbose, saved, show_progress, callback_fn)
            else:
                if self.split_valid_flag and valid_data is not None:
                    source_valid_data, target_valid_data = valid_data
                    if scheme == 'SOURCE':
                        super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                    else:
                        super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result

class DGCDRTrainer(CrossDomainTrainer):
    r"""Trainer for training DGCDR models with a separate CLUB optimizer."""

    def _build_optimizer(self, **kwargs):
        # 1. Main optimizer (excluding CLUB networks)
        main_params = []
        club_params = []
        for name, param in self.model.named_parameters():
            if 'club_' in name:
                club_params.append(param)
            else:
                main_params.append(param)
                
        main_optimizer = torch.optim.Adam(main_params, lr=self.config['learning_rate'])
        
        # 2. Inject CLUB optimizer into the model
        self.model.club_optimizer = torch.optim.Adam(club_params, lr=self.config['learning_rate'] * 0.1, weight_decay=1e-4)
        
        return main_optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        from recbole.utils import set_color, get_gpu_usage
        from torch.nn.utils.clip_grad import clip_grad_norm_
        from tqdm import tqdm

        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            if hasattr(self.model, 'club_optimizer'):
                self.model.club_optimizer.zero_grad()
            
            losses = loss_func(interaction)
            
            if hasattr(self.model, 'club_optimizer'):
                club_loss = losses[-1]
                main_losses = losses[:-1]
            else:
                club_loss = None
                main_losses = losses
                
            if isinstance(main_losses, tuple):
                loss = sum(main_losses)
                loss_tuple = tuple(per_loss.item() for per_loss in main_losses)
                if club_loss is not None:
                    loss_tuple = loss_tuple + (club_loss.item(),)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = main_losses
                if club_loss is not None:
                    loss_tuple = (main_losses.item(), club_loss.item())
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    total_loss = main_losses.item() if total_loss is None else total_loss + main_losses.item()
                
            self._check_nan(loss)
            
            if club_loss is not None:
                club_loss.backward(retain_graph=True)
                
            loss.backward()
            
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                
            self.optimizer.step()
            if club_loss is not None:
                self.model.club_optimizer.step()
            
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss
