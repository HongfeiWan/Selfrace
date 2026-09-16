"""Manual diagnostics moved from simulator/dynamics.py."""

from _bootstrap import add_project_paths

add_project_paths()

import torch

from dynamics import DiscreteActionSpace

test=DiscreteActionSpace(torch.device('cuda'), config={})
print(test.get_all_actions())
print(test.get_action(torch.tensor([0])))
