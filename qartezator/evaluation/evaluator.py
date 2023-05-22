import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


class EvaluatorOnline(nn.Module):
    def __init__(self, scores, clamp_image_range=None):
        """
        :param scores: dict {score_name: EvaluatorScore object}
        """
        super().__init__()
        LOGGER.info(f'{type(self)} init called')
        self.scores = nn.ModuleDict(scores)

        self.interval_names = []
        self.groups = []
        self.clamp_image_range = clamp_image_range

        LOGGER.info(f'{type(self)} init done')

    def forward(self, batch):
        """
        Calculate and accumulate metrics for batch. To finalize evaluation and obtain final metrics, call evaluation_end
        """
        result = {}
        with torch.no_grad():
            if self.clamp_image_range is not None:
                batch['target_img'] = torch.clamp(batch['target_img'],
                                          min=self.clamp_image_range[0],
                                          max=self.clamp_image_range[1])

            for score_name, score in self.scores.items():
                result[score_name] = score(batch['predicted_img'], batch['target_img'])
        return result

    def process_batch(self, batch: Dict[str, torch.Tensor]):
        return self(batch)

    def evaluation_end(self, states=None):
        """:return: dict with (score_name, group_type) as keys, where group_type can be either 'overall' or
            name of the particular group arranged by area of mask (e.g. '10-20%')
            and score statistics for the group as values.
        """
        LOGGER.info(f'{type(self)}: evaluation_end called')

        self.groups = np.array(self.groups)

        results = {}
        for score_name, score in self.scores.items():
            LOGGER.info(f'Getting value of {score_name}')
            cur_states = [s[score_name] for s in states] if states is not None else None
            total_results, group_results = score.get_value(groups=self.groups, states=cur_states)
            LOGGER.info(f'Getting value of {score_name} done')
            results[(score_name, 'total')] = total_results

            for group_index, group_values in group_results.items():
                group_name = self.interval_names[group_index]
                results[(score_name, group_name)] = group_values

        LOGGER.info(f'{type(self)}: reset scores')
        self.groups = []
        for sc in self.scores.values():
            sc.reset()
        LOGGER.info(f'{type(self)}: reset scores done')

        LOGGER.info(f'{type(self)}: evaluation_end done')
        return results
