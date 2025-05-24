import subprocess
import re
import os
import hashlib
import itertools

seed = [0]
model = ["CNN", "resnet18", "resnet34", "resnet50", "vgg11_bn"]
# model = ["resnet18"]
batch_size = [64, 256]
# lr = [1e-3, 5e-4, 1e-4]
lr = [1e-3, 1e-4]
resnet_pretrain = [False, True]
data_aug = [False, True]


all_experiments = list(itertools.product(seed, model, batch_size, lr, resnet_pretrain, data_aug))
print(f"Total number of jobs = {len(all_experiments)}")
for idx, exp in enumerate(all_experiments):
    main_cmd = f"nohup python -u train.py" \
                f" --index {idx}" \
                f" --seed {exp[0]}" \
                f" --model {exp[1]}" \
                f" --batch_size {exp[2]}" \
                f" --lr {exp[3]}"

    if exp[4]:
        main_cmd += f" --pretrained"
    if exp[5]:
        main_cmd += f" --augment"

    main_cmd += f" > result/exp{idx}.log 2>&1"

    print('-------------------------------------')
    print(f'{idx}                     {main_cmd}')
    print('-------------------------------------')

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    process = subprocess.Popen(main_cmd, shell=True, env=env)
    return_code = process.wait()
