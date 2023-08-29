# IMSFR

Efficient and Robust Video Object Segmentation Through Isogenous Memory Sampling and Frame Relation Mining (TIP-2023)

Training
Data preparation

I recommend either softlinking (ln -s) existing data or use the provided download_datasets.py to structure the datasets as our format. download_datasets.py might download more than what you need -- just comment out things that you don't like. The script does not download BL30K because it is huge (>600GB) and we don't want to crash your harddisks.

BL30K

BL30K is a synthetic dataset proposed in MiVOS.

You can either use the automatic script download_bl30k.py or download it manually from MiVOS. Note that each segment is about 115GB in size -- 700GB in total. You are going to need ~1TB of free disk space to run the script (including extraction buffer).
Training commands

    CUDA_VISIBLE_DEVICES=[a,b] OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port [cccc] --nproc_per_node=2 train.py --id [defg] --stage [h]

We implemented training with Distributed Data Parallel (DDP) with two 11GB GPUs. Replace a, b with the GPU ids, cccc with an unused port number, defg with a unique experiment identifier, and h with the training stage (0/1/2/3).

The model is trained progressively with different stages (0: static images; 1: BL30K; 2: 300K main training; 3: 150K main training). After each stage finishes, we start the next stage by loading the latest trained weight.

(Models trained on stage 0 only cannot be used directly. See model/model.py: load_network for the required mapping that we do.)

The .pth with _checkpoint as suffix is used to resume interrupted training (with --load_model) which is usually not needed. Typically you only need --load_network and load the last network weights (without checkpoint in its name).

Inference

    eval_davis_2016per101.py for DAVIS 2016 validation set
    eval_davispreoneconv1046181localgparapoolperandfinaltip.py for DAVIS 2017 validation and test-dev set (controlled by --split)
    eval_youtubeoneconv10146181localgparapoolpeone.py for YouTubeVOS 2018/19 validation set (controlled by --yv_path)


Citation

Please cite our paper if you find this repo useful!
    
    @inproceedings{dang2023efficient,
      title={Efficient and Robust Video Object Segmentation through Isogenous Memory Sampling and Frame Relation Mining},
      author={Dang, Jisheng and Zheng, Huicheng and Lai, Jinming and Yan, Xu and Guo, Yulan},
      journal={IEEE Transactions on Image Processing},
      year={2023},
      pages={3924-3938}, 
      publisher={IEEE}
    }
