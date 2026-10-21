# ===== FedG2C ADD BEGIN B：沿用 Gefra 的命名参数消息格式 =====
"""Source: flcore/client.py:get_pretrain_model/set_pretrain_model,
flcore/server.py:execute/get_global_message. No geometry summaries are sent.
Runtime messages are distinct from disk checkpoint envelopes and metadata.
Unlike Gefra named_parameters, preserve any required shared state buffers too;
the server requires identical buffers instead of averaging them.
"""
from collections.abc import Mapping
import torch


def named_weights(state):
    """Take a detached CPU snapshot instead of Gefra's live Parameter references."""
    return [(name, tensor.detach().cpu().clone()) for name, tensor in state.items()]


def state_from_named_weights(weights):
    if not isinstance(weights, (list, tuple)) or not weights:
        raise ValueError('weight must be a nonempty list of (name, tensor) pairs')
    state = {}
    for item in weights:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError('Each weight item must be (name, tensor)')
        name, tensor = item
        if not isinstance(name, str) or not name.startswith(('experts.', 'composer.')):
            raise ValueError('Only FedG2C experts.* and composer.* can be shared')
        if name in state:
            raise ValueError('Duplicate shared parameter name: ' + name)
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError('Shared values must be finite tensors')
        state[name] = tensor.detach().cpu().clone()
    return state


def client_message(state):
    # Gefra uses num_samples=1. Keep it as a compatibility field; this module's
    # uniform baseline/reference weights do not derive from target sample counts.
    return {'num_samples': 1, 'weight': named_weights(state)}


def states_from_client_messages(messages):
    if not isinstance(messages, Mapping) or not messages:
        raise ValueError('Expected client ID -> local message mapping')
    if len({str(cid) for cid in messages}) != len(messages):
        raise ValueError('Client IDs must be distinct after string normalization')
    states = []
    for message in messages.values():
        if not isinstance(message, Mapping) or set(message) != {'num_samples', 'weight'}:
            raise ValueError('Use num_samples and weight only; geometry/private fields are not uploaded')
        if type(message['num_samples']) is not int or message['num_samples'] != 1:
            raise ValueError('Current Gefra-compatible protocol uses num_samples=1')
        states.append(state_from_named_weights(message['weight']))
    return states
# ===== FedG2C ADD END B =====
