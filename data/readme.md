### Data

We host the UniWM dataset on Hugging Face: [`fly1113/UniWM_Dataset`](https://huggingface.co/datasets/fly1113/UniWM_Dataset). 

- [`go_stanford`](https://huggingface.co/datasets/fly1113/UniWM_Dataset/resolve/main/go_stanford.tar), [`recon`](https://huggingface.co/datasets/fly1113/UniWM_Dataset/resolve/main/recon.tar), [`sacson`](https://huggingface.co/datasets/fly1113/UniWM_Dataset/resolve/main/sacson.tar), [`scand`](https://huggingface.co/datasets/fly1113/UniWM_Dataset/resolve/main/scand.tar) used for both training and evaluation.
- [`tartandrive`](https://huggingface.co/datasets/fly1113/UniWM_Dataset/resolve/main/tartandrive.tar) reserved for unseen evaluation only.

Download and extract all splits into `data/`

After extraction, the directory structure should look like:

```
data/
├── go_stanford/
│   ├── traj_0000/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── n.jpg
│   │   └── traj_data.pkl
│   ├── traj_0001/
│   └── ...
└── ...
```

Each `traj_xxxx/` folder contains a sequence of egocentric frames (`0.jpg`, `1.jpg`, ..., `n.jpg`) and a `traj_data.pkl` file storing the per-step metadata (e.g., actions, poses) for that trajectory. The other splits follow the same layout.
