import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# Parameters
font_size = 16
exp = { 0 : "ResNet18" , 1 : "ResNet18+Augment",2 : "Resnet34", 3:"VGG11_BN", 4:"ResNet50"}

def randomcolor():
    colors = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = "#" + ''.join([random.choice(colors) for i in range(6)])
    return color

plt.figure(figsize=(8, 6))
plt.xlabel("Time")
plt.ylabel(f"Accuracy(%)")
plt.title(f"Fashion-MNIST Test Dataset Accuracy")

xmax = 0
for k,v in exp.items():
    df = pd.read_csv(os.path.join("result", f'{k}', "result.csv"))
    plt.plot(df.index / 20, df['0'], color=randomcolor(), label=f"{v}")

plt.xlim([0, 1])
plt.xticks(np.arange(0, 1, 0.2))

plt.ylim([80, 100])
plt.yticks(np.arange(80, 100, 4))

plt.legend(loc="upper right", bbox_to_anchor=(0.95, 0.91), prop={'size': font_size*0.6})

plt.savefig(f"acc_line.png", bbox_inches='tight')
plt.close()
