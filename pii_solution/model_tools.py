import torch
import transformers
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


def freeze_net(model: transformers.PreTrainedModel, num_layers: int, embeddings: bool) -> None:
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = not embeddings
    for idx, layer in enumerate(model.base_model.encoder.layer):
        grad_req_layer = idx >= num_layers
        for param in layer.parameters():
            param.requires_grad = grad_req_layer


def init_clf_bias(model: transformers.PreTrainedModel, bias_positive: float, bias_negative: float) -> None:
    with torch.no_grad():
        model.classifier.bias[:12] = bias_positive
        model.classifier.bias[12] = bias_negative


class UnfreezingCallback(TrainerCallback):
    def __init__(self, min_layers=None, max_layers=None, num_epochs_unfreeze_embeddings=0):
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.num_epochs_unfreeze_embeddings = num_epochs_unfreeze_embeddings

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        total_layers = len(model.base_model.encoder.layer)
        min_layers = self.min_layers or 0
        max_layers = self.max_layers or total_layers

        if total_layers < max([self.min_layers, self.max_layers]):
            print('Inconsistent params')
        num_epochs = args.num_train_epochs
        current_epoch = state.epoch
        num_epochs_left = num_epochs - current_epoch

        freeze_embeddings = num_epochs_left > self.num_epochs_unfreeze_embeddings
        if state.epoch <= 1:
            num_layers_to_freeze = max_layers
        elif num_epochs_left <= 1:
            num_layers_to_freeze = min_layers
        else:
            num_layers_to_freeze = min_layers + (max_layers - min_layers) * (1 - (current_epoch - 1) / (num_epochs - 2))

        print(f'[{state.epoch}] Total: {total_layers}, freeze: {num_layers_to_freeze}, embs: {freeze_embeddings}')
        freeze_net(model, num_layers_to_freeze, freeze_embeddings)


def get_optimizer_params(model, lr_head, lr_lm, lr_emb, wd_rate=2.6):
    # differential learning rate and weight decay
    learning_rate = lr_lm
    no_decay = ['bias', 'gamma', 'beta']
    num_layers = len(model.base_model.encoder.layer)
    group1=[f'layer.{idx}.' for idx in range(0, num_layers // 3)]
    group2=[f'layer.{idx}.' for idx in range(num_layers // 3, num_layers // 3 * 2)]
    group3=[f'layer.{idx}.' for idx in range(num_layers // 3 * 2, num_layers)]
    group_all=group1+group2+group3
    optimizer_parameters = [
        {'params': [p for n, p in model.base_model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.05, 'lr': lr_emb},  # embeddings
        {'params': [p for n, p in model.base_model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.05, 'lr': learning_rate/wd_rate},
        {'params': [p for n, p in model.base_model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.05, 'lr': learning_rate},
        {'params': [p for n, p in model.base_model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.05, 'lr': learning_rate*wd_rate},
        {'params': [p for n, p in model.base_model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.0},
        {'params': [p for n, p in model.base_model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.0, 'lr': learning_rate/wd_rate},
        {'params': [p for n, p in model.base_model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in model.base_model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.0, 'lr': learning_rate*wd_rate},
        {'params': [p for n, p in model.named_parameters() if model.base_model_prefix not in n], 'lr': lr_head, "momentum" : 0.99},  # classifier head
    ]
    return optimizer_parameters
