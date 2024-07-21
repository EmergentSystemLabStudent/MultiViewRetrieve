#!/bin/bash

args=(
    "partial_fine_tuning.py"
    "unsupervised_partial_fine_tuning.py"
)

for arg in "${args[@]}"; 
do
    python3 "$arg"
done
```

```