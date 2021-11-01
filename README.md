## Introduction
​
This repo contains an implementation of the [SIMPLE](https://www.sciencedirect.com/science/article/pii S1161030118304234) model, written in Python, and wrapped in an OpenAI gym environment for reinforcement learning research. The intent was to develop a simple gym environment to test the tractability of reinforcement learning applied to crop models at a finer scale than a full growing season. In this case, the reinforcement learning agent is tasked with optimizing the irrigation schedule on a daily basis.  Preliminary experiments suggest that reinforcement learning has potential to assist in irrigation optimization, but  further research in the crop model and reinforcement learning application is required to better understand and realize said potential. 
​
We also recommend [CropGym](https://github.com/BigDataWUR/crop-gym), another OpenAI gym environment written in Python for reinforcement learning research on crop models, and based on the well studied  [Python Crop Simulation Environment (PCSE)](https://pcse.readthedocs.io/en/stable/index.html). 

## Usage

1. Setup a new virtual environment (or conda environment) and install the requirements specified in requirements.txt into that environment.
2. Activate the Python virtual environment
3. Test the simple environment by running `python simple.py`.
4. Train the model with: `python train_model.py`.  By default, results and plots are created in the ray_results directory