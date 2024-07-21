#!/bin/bash

args=(
    "partial_fine_tuning_eval.py"
    "unsupervised_partial_fine_tuning_eval.py"
    # "imagenet_pretrain_eval.py"
)

for arg in "${args[@]}"; 
do
    python3 "$arg"
done
```

```