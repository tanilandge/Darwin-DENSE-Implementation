# Darwin-DENSE-Implementation
This repository contains the code to train a federated learning model using NVIDIA FLARE for prostate cancer detection with a variant of DENSE implemented to tackle data heterogenity. The setup follows a semi-supervised federated learning scheme over 5 simulated clients.

### 1. Set up the repo for prostate cancer training here:  https://github.com/Yukiya-Umimi/ITUNet-for-PICAI-2022-Challenge/tree/main
### 2. Set-up your FL set up accordingly: https://github.com/ShubhamK-Yadav/Darwin-Project-FL
### 3. Set up these files in the following structure

```bash
  prostate/prostate_2D/job_configs/picai_fedsemi/
  ├── app/
  │   └── config/
  │       ├── config_fed_client.json
  │       └── config_fed_server.json
  │   └── components/
  │       ├── dense_handler.py
  │   └── dense_distill.py
  │   └── dense_generate.py 
  └── meta.json
```
