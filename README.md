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
+'train_config_multi.yaml' is used to train TinyHD-M (16 outputs).
+'train_config_single.yaml' is used to train TinyHD-S (1 output).

+'eval_config_multi.yaml' is used to evaluate TinyHD-M (16 outputs).
+'eval_config_multi_rc.yaml' is used to evaluate TinyHD-M (16 outputs) with reduced channels (2, 4 times).
+'eval_config_single.yaml' is used to evaluate TinyHD-S (1 outputs).

#### Training