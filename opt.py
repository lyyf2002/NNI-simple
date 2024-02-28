import argparse

import nni
from nni.experiment import Experiment
parser = argparse.ArgumentParser(description='opt')
# 可以加上一些其他如数据集的参数，这些是不需要调参的，这个示例没有使用到下面的参数
parser.add_argument("--data_choice","-d", default="FBYG15K", type=str, choices=["DBP15K", "DWY", "FBYG15K", "FBDB15K"],
                    help="Experiment path")
parser.add_argument("--data_rate", type=float, default=0.8)
parser.add_argument("--port", "-p", type=int, default=8081)


params = parser.parse_args()

search_space = {
        "momentum": {"_type": "choice", "_value": [0.5, 0.9, 0.99]},
        "learning_rate": {"_type": "loguniform", "_value": [0.001, 0.1]},
        "hidden_dim": {"_type": "choice", "_value": [32, 48, 64, 128]},
        "batch_size_train": {"_type": "choice", "_value": [64, 128, 256, 512]},
    }


experiment = Experiment('local')
# 为了不影响原始的train，只有nni=1的时候才会调用nni的代码
cmd = f'python ./main.py --nni 1'

experiment.config.trial_command = cmd
# opt.py与train.py在同一个目录下，所以这里可以直接用'.'
experiment.config.trial_code_directory = '.'

experiment.config.experiment_working_directory = './experiments'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 500
experiment.config.trial_concurrency = 3
experiment.config.max_trial_duration = '240h'
experiment.config.training_service.gpu_indices = [0,1,2]
experiment.config.trial_gpu_number = 1
experiment.config.training_service.use_active_gpu = True
experiment.config.training_service.max_trial_number_per_gpu = 1
experiment.run(params.port)
