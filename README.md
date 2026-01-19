# [TCSVT 2025] Semantic Differentiation Aids Oriented Small Object Detection


## Dependencies
This repo depends on the [MMRotate-0.3.0](https://github.com/open-mmlab/mmrotate), please follow the official setups for installation.

## Datasets
We use SODA-A and Tiny-DOTA in our experiments. And the preparation of SODA-A dataset please refer to [SODA-mmrotate](https://github.com/shaunyuan22/SODA-mmrotate). 
For Tiny-DOTA dataset, we follow [C3Det](https://github.com/ChungYi347/Interactive-Multi-Class-Tiny-Object-Detection) and provide the data split in our work for reproduction 
(`train` for trianing and `val-test` for evaluation), which can be found in `tinydota_data` directory. 
Note you may need running the split scripts in `tools\data\dota\split` to produce sub-patches first.
More details please refer to [our paper](https://ieeexplore.ieee.org/abstract/document/10847719) and C3Det. 

## Training
 - Single GPU:
```
python ./tools/train.py ${CONFIG_FILE} 
```

 - Multiple GPUs:
```
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

## Evaluation
 - Single GPU:
```
python ./tools/test.py ${CONFIG_FILE} ${WORK_DIR} --eval bbox
```

 - Multiple GPUs:
```
bash ./tools/dist_test.sh ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --eval bbox
```

##  Citation
Please cite our work if you find our work and codes helpful for your research.
```
@ARTICLE{SemDiff,
  author={Yuan, Xiang and Cheng, Gong and Yao, Ruixiang and Han, Junwei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Semantic Differentiation Aids Oriented Small Object Detection}, 
  year={2025},
  volume={35},
  number={6},
  pages={5966-5979}
}

@ARTICLE{SODA,
  author={Cheng, Gong and Yuan, Xiang and Yao, Xiwen and Yan, Kebing and Zeng, Qinghua and Xie, Xingxing and Han, Junwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Towards Large-Scale Small Object Detection: Survey and Benchmarks}, 
  year={2023},
  volume={45},
  number={11},
  pages={13467-13488}
}
```
