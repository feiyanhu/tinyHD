# TinyHD: Efficient Video Saliency Prediction With Heterogeneous Decoders Using Hierarchical Maps Distillation
Implementation of WACV 2023 paper "TinyHD: Efficient Video Saliency Prediction With Heterogeneous Decoders Using Hierarchical Maps Distillation".
![](example1.gif)
## If you find the work useful please cite
[Paper link](https://openaccess.thecvf.com/content/WACV2023/papers/Hu_TinyHD_Efficient_Video_Saliency_Prediction_With_Heterogeneous_Decoders_Using_Hierarchical_WACV_2023_paper.pdf)

````
@InProceedings{Hu_2023_WACV,
    author    = {Hu, Feiyan and Palazzo, Simone and Salanitri, Federica Proietto and Bellitto, Giovanni and Moradi, Morteza and Spampinato, Concetto and McGuinness, Kevin},
    title     = {TinyHD: Efficient Video Saliency Prediction With Heterogeneous Decoders Using Hierarchical Maps Distillation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {2051-2060}
}
````

## Install Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install python library dependencies.
```bash
pip install -r requirements.txt
```

## How to run the code
#### Download pretrained teacher models (HD2S) weights and Kinetic400 metadata
The HD2S teacher weights can be downloaded from [here](https://studentiunict-my.sharepoint.com/:u:/g/personal/uni307680_studium_unict_it/EVyDIERfwcdOnAF84v1b1VQBlDNxxhOdI-nAIafqwVV7Lg?download=1) or from [HD2S repository](https://github.com/perceivelab/hd2s)

Kinetic 400 metadata can be downloaded from [This link](https://drive.google.com/file/d/17B0KnCumwsJzWh7GcSVCRe_J5kUYi9tH/view?usp=share_link). Please place the meta data file the directory "src/dataset/metadata/"

#### Configuration file
In order to train, evaluate, load model and generate saliency maps, configuration files are needed. Default configurations files are placed in "src/config/"
+ 'train_config_multi.yaml' is used to train TinyHD-M (16 outputs).
+ 'train_config_multi_rc.yaml' is used to train TinyHD-M (16 outputs) with reduces channels (2, 4 times).
+ 'train_config_single.yaml' is used to train TinyHD-S (1 output).

+ 'eval_config_multi.yaml' is used to evaluate TinyHD-M (16 outputs).
+ 'eval_config_multi_rc.yaml' is used to evaluate TinyHD-M (16 outputs) with reduced channels (2, 4 times).
+ 'eval_config_single.yaml' is used to evaluate TinyHD-S (1 outputs).

The following parameters in the config file can be changed.

#### Model weights
The repo contains the following weight files:
+ 'weights/d1d2d3_S_lt.pth' : Single output w/o channel reduction (DFH1K)
+ 'weights/d123s_rc2_rc1T.pth' : Single output with 2 times channel reduction (DFH1K) and teacher assistant
+ 'weights/d123s_rc4_rc1T.pth' : Single output with 4 times channel reduction (DFH1K) and teacher assistant
+ 'weights/d1d2d3_M_lt.pth' : Multiple output w/o channel reduction (DFH1K)
+ 'weights/d123m_rc2_rc1T.pth' : Multiple output with 2 times channel reduction (DFH1K) and teacher assistant
+ 'weights/d123m_rc4_rc1T.pth' : Multiple output with 4 times channel reduction (DFH1K) and teacher assistant
+ 'weights/ucf_d123s.pth' : Single output w/o channel reduction (UCF-Sport)
+ 'weights/ucf_d123m.pth' : Multiple output w/o channel reduction (UCF-Sport)

#### Quick start
To generate saliency maps of a video:
```bash
python quick_start.py -model_weights_path -video_path -output_path
```
For example using multi-output model trained with DHF1K:
```bash
python quickstart.py ../weights/d1d2d3_M_lt.pth ../example_data/0068.AVI ../output_examples/
```

#### Training
To start training for DHF1K dataset with auxillary dataset Kinetic400:
```bash
python train.py config/train_config_multi.yaml -model_name
```

```bash
python train.py config/train_single_single.yaml -model_name
```

#### Generating saliency map
In order to generate saliency maps:
##### DFH1K
```bash
python generate.py config/eval_config_multi.yaml
```

```bash
python generate.py config/eval_config_single.yaml
```
##### UCF-sport
```bash
python generate.py config/eval_config_multi_ucf.yaml
```

```bash
python generate.py config/eval_config_single.yaml
```

#### Evaluation
To evaluate the performance, saliency maps has to be generated in the previous step.
```bash
python compute_metrics.py -saliency_maps_path -dataset_path -data_type
```
data_type can be 'dhf1k' and 'ucf'.

## Acknowledgement
This publication has been financially supported by:
+ Science Foundation Ireland (SFI) under grant number SFI/12/RC/2289 P2
+ Regione Sicilia, Italy, RehaStart project (grant identifier: PO FESR 2014/2020, Azione 1.1.5, N. 08ME6201000222, CUP G79J18000610007)
+ University of Catania, Piano della Ricerca di Ateneo, 2020/2022,Linea2D
+ MIUR,Italy,Azione1.2“Mobilita` dei Ricercatori” (grant identifier: Asse I, PON R&I 2014- 2020, id. AIM 1889410, CUP: E64I18002520007).