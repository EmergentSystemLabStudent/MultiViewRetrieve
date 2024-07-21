# Object Instance Retrieval in Assistive Robotics: Leveraging Fine-Tuned SimSiam with Multi-View Images Based on 3D Semantic Map

This is a tranining code of [SimView](https://arxiv.org/pdf/2404.09647) paper.

## Dataset Preparation
Download [MIRO](https://arxiv.org/abs/1603.06208) dataset
```bash
bash miro_downloader.sh
python3 redefine_miro.py
```
If you want to download [ObjectPI](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ho_PIEs_Pose_Invariant_Embeddings_CVPR_2019_paper.pdf) dataset.  
Download it [here](https://github.com/chihhuiho/PIE_pose_invariant_embedding) and place it in your dataset folder.

## Training
Fine-tune partial parameters with classifier.
```bash
python3 partial_fine_tuning.py
```

Fine-tune partial parameters without classifier.
```bash
python3 unsupervised_partial_fine_tuning.py
```

Fine-tune all parameters with classifier.
```bash
python3 full_fine_tuning.py
```

Fine-tune all parameters without classifier.
```bash
python3 unsupervised_full_fine_tuning.py
```

## Evaluation
Evaluate the model updated partial parameters with classifier.
```bash
python3 partial_fine_tuning_eval.py
```

Evaluate the model updated partial parameters without classifier.
```bash
python3 unsupervised_partial_fine_tuning_eval.py
```

Evaluate the model updated all parameters with classifier.
```bash
python3 full_fine_tuning_eval.py
```

Evaluate the model updated partial parameters without classifier.
```bash
python3 unsupervised_full_fine_tuning_eval.py
```
