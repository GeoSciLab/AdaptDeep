# AdaptDeep

## Abstract

Restoring fine-grained and detailed information from time-evolving coarse-scale geophysical fields has consistently posed a considerable challenge. Current deep learning approaches addressing this issue require fine-scale ground truth data as supervision, which is often unavailable due to limitations in existing observational systems and the scarcity of widespread high-precision sensors, particularly in economically underdeveloped regions. Here, we present AdaptDeep, a self-supervised restoration framework that accomplishes domain adaptation from the coarse-scale source domain to the fine-scale target domain through knowledge distillation. Two pretext tasks, cropped field reconstruction and temporal augmentation-assisted contrastive learning, are incorporated into the model to leverage spatial and temporal correlations in the target domain. The neural network in AdaptDeep employs a global propagation structure to leverage bidirectional information and global context for enhanced long-range dependencies and robustness against estimation errors. In experiments, AdaptDeep correctly identifies local, fine structures and significantly restores 81.2% detailed information in sea surface temperature fields.

## Code and Installation

releasing soon
