import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

import torch

cfg_file = "ml3d/configs/pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/home/bgao/onet/src/lidar/dataset/Kitti"
dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="cpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
    os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)
module = torch.jit.script(model)
module.save("./logs/pointpillars.pt")

# class InferenceEnd(torch.nn.Module):
#     def __init__(self, model):
#         super(InferenceEnd, self).__init__()
#         self.model = model

#     def forward(self, results:Tuple[torch.Tensor, torch.Tensor, torch.Tensor], data:torch.Tensor):
#         self.model.inference_end(results, data)

# inference_end = InferenceEnd(model)
# torch.jit.script(inference_end).save("./logs/inference_end.pt")

# test_split = dataset.get_split("test")
# data = test_split.get_data(0)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
# result = pipeline.run_inference(data)
# print(result)

# evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()
# pipeline.run_train()

