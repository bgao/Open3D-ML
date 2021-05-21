import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

import torch

framework = "torch"
cfg_file = "ml3d/configs/pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
cfg.dataset['dataset_path'] = "/opt/datasets/Kitti"

Pipeline = _ml3d.utils.get_module("pipeline", cfg.pipeline.name, framework)
Model    = _ml3d.utils.get_module("model", cfg.model.name, framework)
Dataset  = _ml3d.utils.get_module("dataset", cfg.dataset.name)

dataset = Dataset("/opt/datasets/Kitti")
model = Model(**cfg.model, device="cpu")
pipeline = Pipeline(model, dataset=dataset, device="cpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_kitti_202105140715.pth"

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

for i in range(1):
    data = dataset.get_split("test").get_data(i)
    print(len(data['full_point']))
    boxes = pipeline.run_inference(data)
    print(boxes)
