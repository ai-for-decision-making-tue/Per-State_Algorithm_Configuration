# Algorithm Configuration in Sequential Decision-Making
This repository contains the code for the paper "Algorithm Configuration in Sequential Decision-Making", accepted at CPAIOR 2025.
A pre-print version can be found at: https://zenodo.org/records/14936678.

## Cloning the Repository with Dependencies

When cloning the repository, it's essential to also download the required submodules:

```bash
git clone --recurse-submodules https://github.com/ai-for-decision-making-tue/Per-State_Algorithm_Configuration.git
```

## Structure
The repository is divided in two main parts, based on the treated use-case:

### Games
`games` contains everything related to the game use-case, i.e. Connect Four.

All the result files are stored in results/AlgorithmConfiguration/connect_four/connect_four_configuration_exp_c/, hereafter named results_dir.  

To allow for independent data analysis, the data required for creating the figures in step 5 for analyze_performance.ipynb is stored in the directory, so this notebook can be run without going through any of the previous steps.

0. Install psac environment:  
``` conda env create -f games_experiments_files/env.yaml ```  
This will create a conda environment psac, which will be required for steps 2-4.

1. Train the AlphaZero policy:  
``` bash games_experiments_files/run_alpha_zero.sh ```  
This will train the AlphaZero policy that will be used in the game-level SDM. Both for the Per-State Algorithm Configuration, and as the opponent that our algorithm plays against. The model will be saved in the directory models/connect_four_long_run
2. Train PSAC agent:  
``` bash games_experiments_files/run_game_experiment.sh ```  
This will use PPO to train a PSAC agent. The agent will be stored in results_dir/{start_train_time}_ppo.pt.

3. Obtain the best per-instance baseline:  
``` bash games_experiments_files/run_baseline.sh ```  
Thils will do a grid-search to find the best per-instance configuration. The results will be stored in the results_dir/model_performances_c_values

4. Obtain results PSAC and per-instance agents:
In games/experiments.py fill in the file_name of the trained PPO agent from step 2 in line 105 and the best c_value from step 3 in line 104 (see the results file for the c_value with the highest performance).  
``` bash games_experiments_files/run_evaluate_psac.sh  ```  
In the results_dir this will create the directories model_performances_baseline_vs_ppo, model_performances_baseline_vs_random_c and model_performances_ppo_vs_random_c.

5. Create figures:
The figures found in the paper and some extra figures can be re-created by running the analyze_c_selected.ipynb and analyze_performance.ipynb in the games_experiment_files.


### Warehousing
`warehousing` contains everything related to the stochastic combinatorial optimization use-case, i.e. warehouse collaborative order-picking.

The code for the warehousing use-case is based on the [DynaPlex library](https://github.com/DynaPlex/DynaPlex), which is based on C++.
It requires CMake to build the C++ code, you can find more details in the link above.

0. Setting up DynaPlex and Python environment:
First, install the required python dependencies by running:\
``` conda env create -f dynaplex/python/environment.yml ``` \
Then you need to setup the CMakeUserPreset file, then build the library using the WinPB or LinPB preset (depending on your OS).\
Activate the new environment and, from `dynaplex/python`, run:\
``` pip install -e . ```\
More details on these procedures can be found here: https://dynaplex-documentation.readthedocs.io/.
Finally, install the dask library by running:\
``` pip install dask ```

1. Run grid search on stationary environments:\
Run `dask_test_grid_search.py`\

2. Run manual Per-State Algorithm Configuration on the non-stationary environment, based on the grid search results:\
Run `dask_psac_manual.py`. This will apply the parameters reported in the paper, which are based on the grid search results from step 1.

3. Run the trained PSAC agent on the non-stationary environment:\
Run `dask_test_psac_trained.py`. This will run the pre-trained PSAC agent on the non-stationary environment, to replicate the results from the paper.

4. Train the PSAC agent on the non-stationary environment:\
Run `psac_ppo_trainer.py`. This will train a new PSAC agent on the non-stationary environment.
To evaluate it, use the script of step 3, but change the file name of the trained policy in line 18 of `dask_test_psac_trained.py`.


## Citation

If you use this code or the results in your research, please use the following BibTeX entry:

```
@inproceedings{begnardi2025psac,
  author       = "Begnardi, Luca and Meijenfeldt, Bart von and Zhang, Yingqian and Jaarsveld, Willem van and Baier, Hendrik",
  title        = "Algorithm Configuration in Sequential Decision-Making",
  booktitle    = "22nd International Conference, CPAIOR 2025, Melbourne, VIC, Australia, November 10–13, 2025, Proceedings, Part I",
  year         = "2025",
  month        = "7",
}
```