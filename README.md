# Traffic Light Detection and Classification using TensorFlow
---

Based on the work of [coldKnight/TrafficLight_Detection-TensorFlowAPI](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI).

---

## Setup Environment
1. create conda environment `conda create -n tf14 python=3.5` and activate it `activate tf14`
2. install tensorflow, for CPU `pip install tensorflow==1.4.0` of for GPU
    ```
    conda install cudnn=6.0
    conda install cudatoolkit=8.0
    pip install tensorflow-gpu==1.4.0
    ```
3. clone tensorflow/models `git clone https://github.com/tensorflow/models/ --branch r1.4.0 --single-branch` and go to last commit that supports TensorFlow 1.4 `git reset --hard 1f34fcafc1454e0d31ab4a6cc022102a54ac0f5b`.
4. go into `cd models/research` and create protobuf files (if you have not installed protoc install it) by running
`protoc object_detection/protos/*.proto --python_out=.` or something like `“C:\Program Files\protoc-3.4.0-win32\bin\protoc.exe” object_detection/protos/*.proto --python_out=.` on windows
5. install slim `pip install -e slim/.` and delf `pip install -e delf/.`
6. copy the `object_detection` folder to your workspace
7. install some dependencies `pip install matplotlib==3.0.2 Pillow==5.4.1`



## Simulator
### Generate Train and Test Dataset
1. Download [labeled images](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/viewusp=sharing) by [coldKnight](https://github.com/coldKnight)
2. copy `sim_training_data` folder into `data/image_data/`
3. run `simulation_data.ipynb` in `data/` in order to generate`sim_test.record` and `sim_train.record`.

### Setup Model for Training
1. [select a model](https://github.com/tensorflow/models/blob/1f34fcafc1454e0d31ab4a6cc02202a54ac0f5b/research/object_detection/g3doc/detection_model_zoo.md) e.g. `ssd_mobilenet_v1_coco` and unpack it your workspace
2. download the corresponding [configuration](https://github.com/tensorflow/models/tree/1f34fcafc1454e0d31ab4a6cc02202a54ac0f5b/research/object_detection/samples/configs)
3. update configuration (these are the settings we chose, trained witha GeForce GTX 1050 Ti ~10h of training)
```
num_classes: 90 => num_classes: 4
height: 300 => height: 400
width: 300 => width: 400
batch_size: 24 => 8
num_steps: 200000 => num_steps: 25000
max_detections_per_class: 100 => max_detections_per_class: 10
max_total_detections: 100 => max_total_detections: 10
fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt" =>fine_tune_checkpoint: "ssd_mobilenet_v1_coco_2017_11_17/model.ckpt"
PATH_TO_BE_CONFIGURED => data
num_examples: 8000 => num_examples: 974
```
remove `data_augmentation_options` as the dataset has already beenaugmented and `max_evals: 10` as we want to test on all testdata.

### Train the Model
1. copy `train.py` and `eval.py` from `object_detection` in the root of you workspace
2. start training
    ```
    python train.py --logtostderr --train_dir=./trained_models --pipeline_config_path=ssd_mobilenet_v1_coco.config
    ```
3. while traing you can evaulate the model on the test dataset by running `
    ```
    python eval.py --logtostderr --pipeline_config_path=ssd_mobilenet_v1_coco.config --checkpoint_dir=trained_models --eval_dir=trained_models/eval/ --run_once=True
    ```
4. show training progress `tensorboard --logdir=trained_models`

### Freezing the Graph
1. copy `export_inference_graph.py` from `object_detection` in the root of you workspace
2. freeze the graph
    ```
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix trained_models/model.ckpt-25000 --output_directory frozen_graph
    ```
3. fix `ValueError: Protocol message RewriterConfig has no "layout_optimizer" field.` by editing line 72 in `object_detection/exporter.py`. Change `layout_optimizer` to `optimize_tensor_layout`.

