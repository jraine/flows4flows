# flows4flows

A utility package for creating flows for flows models using nflows and PyTorch.

All code in the repository was used for the flows4flows pre-print.

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