# SymNet
As a part of [HAKE](http://hake-mvig.cn/) project (HAKE-Object).

#### **News**: (2022.12.19) HAKE 2.0 is accepted by TPAMI!

(2022.12.7) We release a new project [OCL](https://mvig-rhos.com/ocl) ([paper](https://arxiv.org/abs/2212.02710)). Data and code are coming soon.

(2022.11.19) We release the interactive object bounding boxes & classes in the interactions within AVA dataset (2.1 & 2.2)! [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA), [[Paper]](https://arxiv.org/abs/2211.07501).

(2022.03.28) We release the code of multiple attribute recognition mentioned in PAMI version

(2022.02.14) We release the human body part state labels based on AVA: [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA).

(2021.10.06) Our extended version of [SymNet](https://github.com/DirtyHarryLYL/SymNet) is accepted by TPAMI! Paper and code are coming soon.

(2021.2.7) Upgraded [HAKE-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec) is released! Images/Videos --> human box + ID + skeleton + part states + action + representation. [[Description]](https://drive.google.com/file/d/1iZ57hKjus2lKbv1MAB-TLFrChSoWGD5e/view?usp=sharing), Full demo: [[YouTube]](https://t.co/hXiAYPXEuL?amp=1), [[bilibili]](https://www.bilibili.com/video/BV1s54y1Y76s)

<!-- (2020.10.27) The code of [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)) ([Paper](https://arxiv.org/abs/2010.16219)) in NeurIPS'20 is released! -->

(2020.6.16) Our larger version [HAKE-Large](https://github.com/DirtyHarryLYL/HAKE#hake-large-for-instance-level-hoi-detection) (>120K images, activity and part state labels) is released!

This is the code accompanying our CVPR'20 and TPAMI'21 papers: **Symmetry and Group in Attribute-Object Compositions** [![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2004.00587), **Learning Single/Multi-Attribute of Object with Symmetry and Group** [![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2110.04603)

<!-- **Symmetry and Group in Attribute-Object Compositions**. [[arXiv](https://arxiv.org/abs/2004.00587)]
*[Yong-Lu Li](https://dirtyharrylyl.github.io/), [Yue Xu](https://silicx.github.io/), Xiaohan Mao, [Cewu Lu](http://mvig.sjtu.edu.cn/)*

**Learning Single/Multi-Attribute of Object with Symmetry and Group**. [[arXiv](https://arxiv.org/abs/2110.04603)]
*[Yong-Lu Li](https://dirtyharrylyl.github.io/), [Yue Xu](https://silicx.github.io/), [Xinyu Xu](https://xuxinyu.website) ,Xiaohan Mao, [Cewu Lu](http://mvig.sjtu.edu.cn/)* -->

![Overview](./data/overview.png)

If you find this repository useful for you, please consider citing our paper.
```
---SymNet-PAMI
@article{li2021learning,
  title={Learning Single/Multi-Attribute of Object with Symmetry and Group},
  author={Li, Yong-Lu and Xu, Yue and Xu, Xinyu and Mao, Xiaohan and Lu, Cewu},
  journal={TPAMI},
  year={2021}
}
---SymNet-CVPR
@inproceedings{li2020symmetry,
	title={Symmetry and Group in Attribute-Object Compositions},
	author={Li, Yong-Lu and Xu, Yue and Mao, Xiaohan and Lu, Cewu},
	booktitle={CVPR},
	year={2020}
}
```

## Prerequisites

**Packages**: Install using `pip install -r requirements.txt`

**Datasets**: Download and re-arrange with:
	
	cd data; bash download_data.sh

**Features and pretrained models**: Features for compositional ZSL (CZSL) setting<sup>[1]</sup> will be downloaded together with the datasets. Features for generalized compositional ZSL (GCZSL) setting<sup>[2]</sup> can be extracted using:

	python utils/dataset/GCZSL_dataset.py [MIT/UT]


For multiple attribute recognition, we re-organize the metadata of aPY/SUN datasets with pre-extracted ResNet-50 feature in 4 files `{APY/SUN}_{train/test}.pkl`.
You can download them from [Link](https://drive.google.com/file/d/1xkdxbgBhE1S7HdeaUtn_8rtm5lE0Dx6z/view) and put them into `./data` folder.

Pretrained models and intermediate results can be downloaded from here: [Link](https://drive.google.com/drive/folders/1qcgAeEeXakX3-RsFM3pKfKsj7F18XBHA?usp=sharing). Please unzip the `obj_scores.zip` to `./data/obj_scores` and `weights.zip` to `./weights`.


## Compositional Zero-shot Leaning (CZSL)

These are commands for the split and evaluation metrics introduced by [1].

### Training a object classifier

Before training a SymNet model, train an object classifier by running:

	python run_symnet.py --network fc_obj --name UT_obj_lr1e-3 --data UT --epoch 300 --batchnorm --lr 1e-3 --force
	python run_symnet.py --network fc_obj --name UT_obj_lr1e-3 --data UT --epoch 300 --batchnorm --lr 1e-3

Then store the intermediate object results:

	python test_obj.py --network fc_obj --name MIT_obj_lr3e-3 --data MIT --epoch 1120 --batchnorm
	python test_obj.py --network fc_obj --name UT_obj_lr1e-3 --data UT --epoch 140 --batchnorm

The results file will be stored in `./data/obj_scores` with names `MIT_obj_lr3e-3_ep1120.pkl` and `UT_obj_lr1e-3_ep140.pkl` (in the examples above).

### Training a SymNet

To train a SymNet with the hyper-parameters in our paper, run:

	python run_symnet.py --name MIT_best --data MIT --epoch 400 --obj_pred MIT_obj_lr3e-3_ep1120.pkl --batchnorm --lr 5e-4 --bz 512 --lambda_cls_attr 1 --lambda_cls_obj 0.01 --lambda_trip 0.03 --lambda_sym 0.05 --lambda_axiom 0.01
	python run_symnet.py --name UT_best --data UT --epoch 700 --obj_pred UT_obj_lr1e-3_ep140.pkl --batchnorm  --wordvec onehot  --lr 1e-4 --bz 256 --lambda_cls_attr 1 --lambda_cls_obj 0.5 --lambda_trip 0.5 --lambda_sym 0.01 --lambda_axiom 0.03



### Model Evaluation

	python test_symnet.py --name MIT_best --data MIT --epoch 320 --obj_pred MIT_lr3e-3_ep1120.pkl --batchnorm
	python test_symnet.py --name UT_best --data UT --epoch 600 --obj_pred UT_lr1e-3_ep140.pkl --wordvec onehot --batchnorm



Method | MIT (top-1) | MIT (top-2) |MIT (top-2) | UT (top-1) | UT (top-2) | UT (top-3)  
-- | -- | -- | -- | -- | -- | -- |
Visual Product  | 9.8/13.9 | 16.1 | 20.6 | 49.9 | / | / 
LabelEmbed (LE) | 11.2/13.4| 17.6 | 22.4 | 25.8 | / | / 
~- LEOR            | 4.5          | 6.2  | 11.8 |  /       | / | / 
~- LE + R          | 9.3          | 16.3 | 20.8 |  /       | / | / 
~- LabelEmbed+    | 14.8*         |  /   |  /   | 37.4| / | / 
AnalogousAttr | 1.4          |  /   |  /   | 18.3  |  /  |  /  
Red Wine        | 13.1         | 21.2 | 27.6 | 40.3  |  /  |  /   
AttOperator    | 14.2         | 19.6 | 25.1 | 46.2  | 56.6 | 69.2 
TAFE-Net           | 16.4         | 26.4 | 33.0 | 33.2  |  /  |  /  
GenModel       | 17.8         |  /   |  /   | 48.3  |  /  |  /  
**SymNet (Ours)** | **19.9** | **28.2** | **33.8** | **52.1**  |**67.8** |  **76.0** 



## Generalized Compositional Zero-shot Leaning (GCZSL)

These are commands for the split and evaluation metrics introduced by [2].

### Training a object classifier

	python run_symnet.py --network fc_obj --data MITg --name MITg_obj_lr3e-3 --bz 2048 --test_bz 2048  --lr 3e-3 --epoch 1000 --batchnorm --fc_cls 1024

	python run_symnet.py --network fc_obj --data UTg --name UTg_obj_lr1e-3 --bz 2048 --test_bz 2048 --lr 1e-3 --epoch 700 --batchnorm  --fc_cls 1024			

To store the object classification results of both valid and test set, run:

	python test_obj.py --network fc_obj --data MITg --name MITg_obj_lr3e-3 --bz 2048 --test_bz 2048  --epoch 980 --batchnorm --fc_cls 1024 --test_set val
	python test_obj.py --network fc_obj --data MITg --name MITg_obj_lr3e-3 --bz 2048 --test_bz 2048  --epoch 980 --batchnorm --fc_cls 1024 --test_set test

	python test_obj.py --network fc_obj --data UTg --name UTg_obj_lr1e-3 --bz 2048 --test_bz 2048 --epoch 660 --batchnorm  --fc_cls 1024 --test_set val
	python test_obj.py --network fc_obj --data UTg --name UTg_obj_lr1e-3 --bz 2048 --test_bz 2048 --epoch 660 --batchnorm  --fc_cls 1024 --test_set test


### Trainig a SymNet
To train a SymNet for GCZSL, run:

	python run_symnet_gczsl.py --data MITg --name MITg_best --epoch 1000 --obj_pred MITg_obj_lr3e-3_val_ep980.pkl --test_set val --lr 3e-4 --bz 512 --test_bz 512 --batchnorm  --lambda_cls_attr 1 --lambda_cls_obj 0.01 --lambda_trip 1 --lambda_sym 0.02 --lambda_axiom 0.02 --triplet_margin 0.3

	python run_symnet_gczsl.py --data UTg --name UTg_best --epoch 300 --obj_pred UTg_obj_lr1e-3_val_ep660.pkl --test_set val --lr 1e-3 --bz 512 --test_bz 512 --wordvec onehot --batchnorm --lambda_cls_attr 1 --lambda_cls_obj 0.01 --fc_compress 512 --lambda_trip 1 --lambda_sym 0.02 --lambda_axiom 0.01


### Model Evaluation
	
	python test_symnet_gczsl.py --data MITg --name MITg_best --epoch 1000 --obj_pred MITg_obj_lr3e-3_test_ep980.pkl --bz 512 --test_bz 512 --batchnorm  --triplet_margin 0.3 --test_set test --topk 1
	python test_symnet_gczsl.py --data MITg --name MITg_best --epoch 1000 --obj_pred MITg_obj_lr3e-3_val_ep980.pkl --bz 512 --test_bz 512 --batchnorm  --triplet_margin 0.3 --test_set val --topk 1

	python test_symnet_gczsl.py --data UTg --name UTg_best --epoch 290 --obj_pred UTg_obj_lr1e-3_test_ep660.pkl --bz 512 --test_bz 512 --batchnorm --wordvec onehot --fc_compress 512 --test_set test --topk 1
	python test_symnet_gczsl.py --data UTg --name UTg_best --epoch 290 --obj_pred UTg_obj_lr1e-3_val_ep660.pkl --bz 512 --test_bz 512 --batchnorm --wordvec onehot --fc_compress 512 --test_set val --topk 1


MIT-States evaluation results (with metrics of TMN<sup>[2]</sup>)

Model | Val Top-1 AUC | Val Top-2 AUC | Val Top-3 AUC | Test Top-1 AUC | Test Top-2 AUC | Test Top-3 AUC | Seen | Unseen | HM
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
AttOperator  | 2.5 | 6.2 | 10.1 | 1.6 | 4.7 | 7.6 | 14.3    | 17.4 | 9.9 
Red Wine      | 2.9 | 7.3 | 11.8 | 2.4 | 5.7 | 9.3 | 20.7    | 17.9 | 11.6
LabelEmbed+  | 3.0 | 7.6 | 12.2 | 2.0 | 5.6 | 9.4 | 15.0    | 20.1 | 10.7
GenModel     | 3.1 | 6.9 | 10.5 | 2.3 | 5.7 | 8.8 | 24.8    | 13.4 | 11.2
TMN               | 3.5 | 8.1 | 12.4 | 2.9 | 7.1 | 11.5| 20.2    | 20.1 | 13.0
**SymNet (CVPR)** | **4.3** | **9.8** | **14.8** | **3.0** | **7.6** | **12.3** | 24.4 | **25.2** | **16.1**
**SymNet (TPAMI)** | **5.4** | **11.6** | **16.6** | **4.5** | **10.1** | **15.0** | **26.2** | **26.3** | **16.8**
**SymNet (Latest Update)** | **5.8** | **12.2** | **17.8** | **5.3** | **11.3** | **16.5** | **29.5** | **26.1** | **17.4**

UT-Zappos evaluation results (with metrics of CAUSAL<sup>[3]</sup>)

Model | Unseen | Seen | Harmonic | Closed | AUC
-- | -- | -- | -- | -- | -- 
LabelEmbed  | 16.2 | 53.0 | 24.7 | 59.3 | 22.9
AttOperator | 25.5 | 37.9 | 27.9 | 54.0 | 22.1
TMN        | 10.3 | 54.3 | 17.4 | **62.0** | 25.4
CAUSAL     | **28.0** | 37.0 | **30.6** | 58.6 | 26.4
**SymNet (Ours)** | 10.3 | **56.3** | 24.1 | 58.7 | **26.8**


## Multiple Attribute Recognition


### Trainig a SymNet
To train a SymNet for multiple attribute recognition, run:

	python run_symnet_multi.py --name APY_best --data APY --rmd_metric sigmoid --fc_compress 256 --rep_dim 128  --test_freq 1  --epoch 100 --batchnorm --lr 3e-3 --bz 128 --lambda_cls_attr 1 --lambda_trip 1 --lambda_sym 5e-2 --bce_neg_weight 0.05 --lambda_cls_obj 5e-2 --lambda_axiom 1e-3  --lambda_multi_rmd 5e-2  --lambda_atten 1

	python run_symnet_multi.py --name SUN_best --data SUN --rmd_metric rmd --fc_compress 1536 --rep_dim 128 --test_freq 5 --epoch 150 --batchnorm --lr 5e-3 --bz 128  --lambda_cls_attr 1 --lambda_trip 5e-2 --lambda_sym 8e-3 --bce_neg_weight 0.4 --lambda_cls_obj 3e-1 --lambda_axiom 1e-3 --lambda_multi_rmd 6e-2 --lambda_atten 6e-1


### Model Evaluation
	
	python test_symnet_multi.py --data APY --name APY_best --epoch 78 --batchnorm --rep_dim 128 --fc_compress 256
	python test_symnet_multi.py --data SUN --name SUN_best --epoch 95 --batchnorm --rep_dim 128 --fc_compress 1536



Evaluation results on aPY and SUN (with metrics of mAUC)

Model 				| aPY	 	| SUN 		|
-- 					| -- 		| -- 		| 
ALE 				| 69.2 		| 74.5  	|
HAP 				| 58.2 		| 76.7		|
UDICA 				| 82.3 		| 85.8		|
KDICA 				| 84.7 		|	/		|
UMF 				| 79.7 		| 80.5		|
AMT 				| 84.5 		| 82.5  	|
FMT 				| 70.5		| 75.5  	|
GALM 				| 84.2 		| 86.5		|
**SymNet (Ours)**	| **86.1**	| **88.4**  |

## Tips

### Use Customized Dataset

Take UT as example, beside reorganizing the images to `data/ut-zap50k-original/images/[attribute]_[object]/`:

- If you are using customized pairs composed by our provided attributes and objects, only the pair lists in `data/ut-zap50k-original/compositional-split/` need to be updated.

- If you also use customized attributes and objects, there are several additional files to modify in folder `utils/aux_data/`:

  1. `UT_attrs.json` and `UT_objs.json` are attribute and object list, stored as `dict`. The keys are original names and values are names in pre-trained GloVe vocabs.

  2. `glove_UT.py` contains GloVe vectors for the attributes and objects. In our paper, `glove.6B.300d.txt` is used.

  3. `UT_weight.py` contains loss weights for each individual attribute or object class (only `attr_weight` and `obj_weight`) (`pair_weight` is never used and can be set to 1). In practice, these weights can help the training on imbalanced data. Each weight is computed by **-log(p)**, where **p** is the occurrence frequency of an attribute or object in train set. E.g. a five-image dataset have attribute labels `[a,a,a,b,b]`, then the `attr_weight` for `a` and `b` is `[-log0.6, -log0.4]`. You may clip the values to prevent large or zero weights.

<!-- 
## TODO
- [ ] Unified backbone
- [ ] Tips for hyperparameters and tuning
- [ ] Some possible tricks
- [ ] New module for multi-label attribute recognition
- [ ] Torch version -->


## Acknowledgement
The dataloader and evaluation code are based on [Attributes as Operators](https://github.com/Tushar-N/attributes-as-operators)<sup>[1]</sup> and [Task-Driven Modular Networks](https://github.com/facebookresearch/taskmodularnets)<sup>[2]</sup>.



## Reference

[1] [Attributes as Operators: Factorizing Unseen Attribute-Object Compositions](https://arxiv.org/abs/1803.09851)

[2] [Task-Driven Modular Networks for Zero-Shot Compositional Learning](https://arxiv.org/abs/1905.05908)

[3] [A causal view of compositional zero-shot recognition](https://arxiv.org/abs/2006.14610)
