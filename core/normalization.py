import torch
import torch.nn as nn

class Normalizer(nn.Module):
    """Feature normalizer that accumulates statistics online."""

    def __init__(self, batch_size, feature_size, name, device, max_accumulations=10 ** 6, std_epsilon=1e-8):
        super(Normalizer, self).__init__()
        self._name = name
        self._device = device
        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon
        
        self._acc_count = 0
        self._num_accumulations = 0
        self._acc_sum = torch.zeros((batch_size, feature_size), dtype=torch.float, device = device)
        self._acc_sum_squared = torch.zeros((batch_size, feature_size), dtype=torch.float, device = device)

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate and self._num_accumulations < self._max_accumulations:
            # stop accumulating after a million updates, to prevent accuracy issues
            self._accumulate(batched_data)
        return torch.einsum('ij,ikj->ikj', 1/self._std_with_epsilon(), (batched_data - self._mean().unsqueeze(1)))

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return torch.einsum('ij,ikj->ikj', self._std_with_epsilon(), normalized_batch_data) + self._mean().unsqueeze(1)

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        data_sum = torch.sum(batched_data, dim=1)
        squared_data_sum = torch.sum(batched_data ** 2, dim=1)
        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += batched_data.shape[1]
        self._num_accumulations += 1

    def _mean(self):
        safe_count = max(self._acc_count, 1)
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = max(self._acc_count, 1)
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return std +  self._std_epsilon

    def get_acc_sum(self):
        return self._acc_sum
