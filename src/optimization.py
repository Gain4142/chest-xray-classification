"""Explicit parameter groups and per-update warmup/cosine schedule."""
import math
import torch
from transformers import get_cosine_schedule_with_warmup


def parameter_groups(model, lr, weight_decay, llrd_decay=0.75):
    if getattr(model, 'training_scope', None) != 'llrd':
        return [{'params': [p for p in model.parameters() if p.requires_grad],
                 'lr': lr, 'weight_decay': weight_decay, 'group_name': 'selected'}]
    net = model.network
    count = len(net.vit.encoder.layer)
    groups = [{'params': list(net.vit.embeddings.parameters()),
               'lr': lr * llrd_decay ** (count + 1),
               'weight_decay': weight_decay, 'group_name': 'embeddings'}]
    for i, layer in enumerate(net.vit.encoder.layer):
        groups.append({'params': list(layer.parameters()),
                       'lr': lr * llrd_decay ** (count - i),
                       'weight_decay': weight_decay, 'group_name': f'block_{i}'})
    groups.append({'params': list(net.vit.layernorm.parameters()) + list(net.classifier.parameters()),
                   'lr': lr, 'weight_decay': weight_decay, 'group_name': 'norm_and_head'})
    actual = [id(p) for group in groups for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('LLRD groups must cover every trainable parameter exactly once')
    return groups


def build_scheduler(optimizer, kind, total_steps, warmup_ratio):
    if kind == 'constant':
        return None
    if kind != 'warmup_cosine' or total_steps < 2:
        raise ValueError('Invalid scheduler or too few update steps')
    warmup_steps = min(total_steps - 1, max(1, math.ceil(total_steps * warmup_ratio)))
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def group_report(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [{'name': g.get('group_name', str(i)), 'peak_lr': g['lr'],
             'parameters': sum(p.numel() for p in g['params']),
             'parameter_names': [names[id(p)] for p in g['params']]}
            for i, g in enumerate(optimizer.param_groups)]
