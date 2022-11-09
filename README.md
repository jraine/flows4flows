# flows4flows

A utility package for creating flows for flows models using nflows and PyTorch.

All code in the repository was used for the flows4flows pre-print available at https://arxiv.org/abs/2211.02487.

## Provides

`FlowForFlow` - main class for handling the flow with flows for base distributions. Handles the loss calculation and transformation.

Several predefined classes for standard context methods defined.

Distance penalty classes for additional distance loss term in training a flow4flow model.

## Installation

```
python3 -m pip install .
```

## Usage

```
from nflows import transforms 
from nflows.distributions import StandardNormal 
import ffflows.models as fff



## Create a base flow to train, just as with nflows
base_flow = fff.BaseFlow(transforms.CompositeTransform(transform_list_base),
                     StandardNormal)


## Create a flow4flow model from a nflows.transformer and one/two base flows (depending on application)
f4flow = fff.DeltaFlowForFlow(transforms.CompositeTransform(transform_list_f4f),
                               base_flow)
```

## Citation

Please cite the arXiv paper https://arxiv.org/abs/2211.02487

```
@misc{https://doi.org/10.48550/arxiv.2211.02487,
  doi = {10.48550/ARXIV.2211.02487},
  url = {https://arxiv.org/abs/2211.02487},
  author = {Klein, Samuel and Raine, John Andrew and Golling, Tobias},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Flows for Flows: Training Normalizing Flows Between Arbitrary Distributions with Maximum Likelihood Estimation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Contact

[Sam Klein](https://github.com/sambklein) &
[Johnny Raine](https://github.com/jraine)
