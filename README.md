## Task Relation-aware Continual User Representation Learning

The revised source code for [**Task Relation-aware Continual User Representation Learning**](https://arxiv.org/abs/2306.01792) paper, accepted at KDD 2023.


## Abstract
User modeling, which learns to represent users into a low-dimensional representation space based on their past behaviors, got a surge of interest from the industry for providing personalized services to users. Previous efforts in user modeling mainly focus on learning a task-specific user representation that is designed for a single task. However, since learning task-specific user representations for every task is infeasible, recent studies introduce the concept of universal user representation, which is a more generalized representation of a user that is relevant to a variety of tasks. Despite their effectiveness, existing approaches for learning universal user representations are impractical in real-world applications due to the data requirement, catastrophic forgetting and the limited learning capability for continually added tasks. In this paper, we propose a novel continual user representation learning method, called TERACON, whose learning capability is not limited as the number of learned tasks increases while capturing the relationship between the tasks. The main idea is to introduce an embedding for each task, i.e., task embedding, which is utilized to generate task-specific soft masks that not only allow the entire model parameters to be updated until the end of training sequence, but also facilitate the relationship between the tasks to be captured. Moreover, we introduce a novel knowledge retention module with pseudo-labeling strategy that successfully alleviates the long-standing problem of continual learning, i.e., catastrophic forgetting. Extensive experiments on public and proprietary real-world datasets demonstrate the superiority and practicality of TERACON. 

![](https://github.com/Sein-Kim/TERACON_Revised/assets/76777494/a30959a2-95a2-414f-a4c2-49c216d728ee)

## Dataset

- You can download the datasets from this url from [CONURE](https://arxiv.org/abs/2009.13724)<br>

  - TTL: https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view<br>

  - MovieLens: https://grouplens.org/datasets/movielens/25m/

- For your own custom dataset, format it as follows: <br>
  - Format: `Input Sequence` `,,` `"Targets"` <br>
  - For e.g.,<br>
    ~~~
    0,0,0,0,0,0,1,2,3,4,,2,3,4
    0,0,0,0,0,1,2,5,7,3,,8
    0,0,0,5,6,7,2,2,4,5,,10
    0,0,0,0,0,8,9,3,4,4,,20
    ~~~
  - Please refer to the example datasets in the `example` folder.

## Hyperparameters

`--datapath:` Path of dataset.<br>
usage example :`--dataset ./ColdRec/original_desen_finetune_click_nosuerID.csv`

`--paths:` Path of previous task model.<br>
usage example :`--paths ./saved_models/task1.t7`

`--savepath` Storage path of current model.<br>
usage example : `--savepath ./saved_models/task2.t7`

`--n_tasks:`  Total number of the tasks.<br>
usage example :`--n_tasks 2`

`--datapath_index:` Path of item index dictionary (i.e., `Data/Session/index.csv`).<br>
usage example :`--datapath_index Data/Session/index.csv`
- Note that, the file `index.csv` is automatically generated when running Task 1.
Specifically, when running the dataset for Task 1, the `data_loader` generates the `index.csv` file, which contains the index information for all items in Task 1.<br>

`--lr:` Learning rate of model.<br>
usage example : `--lr 0.0001`

`--alpha:` Controls the contribution of the knowledge retention.<br>
usage example : `--alpha 0.7`

`--smax:` Positive scaling hyper-parameter.<br>
usage example : `--smax 50`


## How to Run

~~~
git clone https://github.com/Sein-Kim/TERACON_Revised.git
cd TERACON_Revised
mkdir saved_model
mkdir Data/Session
~~~

- Download `TTL` data from [here](https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view) to `ColdRec`.
- To train the model for Task 1, please run the `train_task1.py`.
  ~~~
  python train_task1.py --epochs 10 --lr 0.001 --batch 32
  ~~~

- Then run other tasks by
  ~~~
  sh ttl_train.sh
  ~~~

<!-- 

<br>
E.g., if train the tasks of TTL in the paper, learn sequentially<br>

~~~
python train_teracon.py --lr 0.0001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_finetune_click_nosuerID.csv' --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task1.t7" --savepath "./saved_models/task2" --n_tasks 2
~~~

<br>

~~~
python train_teracon.py --lr 0.0001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_finetune_like_nosuerID.csv' --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task2.t7" --savepath "./saved_models/task3" --n_tasks 3
~~~

<br>

~~~
python train_teracon.py --lr 0.0001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_age.csv' --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task3.t7" --savepath "./saved_models/task4" --n_tasks 4
~~~

<br>

~~~
python train_teracon.py --lr 0.0001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_gender.csv' --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task4.t7" --savepath "./saved_models/task5" --n_tasks 5
~~~

<br>

~~~
python train_teracon.py --lr 0.0001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_lifestatus.csv' --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task5.t7" --savepath "./saved_models/task6" --n_tasks 6
~~~
-->
- To perform inference on past tasks using the current model, please run the `inference_past_tasks.py`.
  - As an example, if you have trained up to Task 5 using the TTL dataset, run the following `ttl_inference.sh` to proceed with the inference on past tasks:
    ~~~
    sh ttl_inference.sh
    ~~~ 


## Backbone network Code
- The data_loder code and basic backbone network are refered to<br>

  - https://github.com/yuangh-x/2022-NIPS-Tenrec

  - https://github.com/syiswell/NextItNet-Pytorch

### Cite (Bibtex)
- Please refer the following paer, if you find TERACON useful in your research:
  - Kim, Sein and Lee, Namkyeong and Kim, Donghyun and Yang, Minchul and Park, Chanyoung. "Task Relation-aware Continual User Representation Learning." KDD 2023.
  - Bibtex
```
@article{kim2023task,
  title={Task Relation-aware Continual User Representation Learning},
  author={Kim, Sein and Lee, Namkyeong and Kim, Donghyun and Yang, Minchul and Park, Chanyoung},
  journal={arXiv preprint arXiv:2306.01792},
  year={2023}
}
```
