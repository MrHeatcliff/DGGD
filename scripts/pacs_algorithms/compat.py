"""Local runtime compatibility for native algorithms; does not alter ERM jobs."""
from collections import OrderedDict
import torch
from torch import nn
from domainbed import algorithms


def linear_ce_per_sample_grads(features, logits, labels):
    """Exact per-example gradients of sum-reduced CE for a linear classifier."""
    residual = logits.softmax(dim=1) - torch.nn.functional.one_hot(labels, logits.shape[1]).to(logits.dtype)
    weight = torch.einsum('nc,nd->ncd', residual, features).flatten(start_dim=1)
    return OrderedDict(weight=weight, bias=residual)


class FishrCompatible(algorithms.Fishr):
    """Replace only BackPACK's per-example linear gradient extraction."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.classifier, nn.Linear), 'Analytic adapter requires the default linear head'
        original = self.classifier
        with torch.random.fork_rng(devices=[]):
            classifier = nn.Linear(original.in_features, original.out_features, bias=original.bias is not None)
        classifier.load_state_dict(original.state_dict())
        self.classifier = classifier
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.classifier.register_forward_pre_hook(self._capture_features)
        self._init_optimizer()

    def _capture_features(self, module, inputs):
        self._classifier_features = inputs[0]

    def _get_grads(self, logits, labels):
        return linear_ce_per_sample_grads(self._classifier_features, logits, labels)


def algorithm_class(name):
    return FishrCompatible if name == 'Fishr' else algorithms.get_algorithm_class(name)
