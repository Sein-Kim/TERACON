## Task Relation-aware Continual User Representation Learning

The revised source code for [**Task Relation-aware Continual User Representation Learning**](https://arxiv.org/abs/2306.01792) paper, accepted at KDD 2023.


## Abstract
User modeling, which learns to represent users into a low-dimensional representation space based on their past behaviors, got a surge of interest from the industry for providing personalized services to users. Previous efforts in user modeling mainly focus on learning a task-specific user representation that is designed for a single task. However, since learning task-specific user representations for every task is infeasible, recent studies introduce the concept of universal user representation, which is a more generalized representation of a user that is relevant to a variety of tasks. Despite their effectiveness, existing approaches for learning universal user representations are impractical in real-world applications due to the data requirement, catastrophic forgetting and the limited learning capability for continually added tasks. In this paper, we propose a novel continual user representation learning method, called TERACON, whose learning capability is not limited as the number of learned tasks increases while capturing the relationship between the tasks. The main idea is to introduce an embedding for each task, i.e., task embedding, which is utilized to generate task-specific soft masks that not only allow the entire model parameters to be updated until the end of training sequence, but also facilitate the relationship between the tasks to be captured. Moreover, we introduce a novel knowledge retention module with pseudo-labeling strategy that successfully alleviates the long-standing problem of continual learning, i.e., catastrophic forgetting. Extensive experiments on public and proprietary real-world datasets demonstrate the superiority and practicality of TERACON. 

## Dataset

- You can download the datasets from this url from [CONURE](https://arxiv.org/abs/2009.13724)<br>

  - TTL: https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view<br>

  - MovieLens: https://grouplens.org/datasets/movielens/25m/

---
For your own custom dataset, format it as follows: <br>
"[Source (input) Sequence (seperator: ,)]" ,, "[targets]" <br>
For e.g.,<br>
0,0,0,0,0,0,1,2,3,4,,2,3,4<br>
0,0,0,0,0,1,2,5,7,3,,8<br>
0,0,0,5,6,7,2,2,4,5,,10<br>
0,0,0,0,0,8,9,3,4,4,,20<br>
Please refer to the example datasets in the "example" folder.

## Arguments
'datapath_index', i.e., "Data/session/index.csv"
is automatically generated when running task1.
More specifically, if you run the dataset for task1, the data_loader generates index.csv for all items in task1.



## How to Run
First run the task 1
train_task1.py
to get the model which train task1
<br>

~~~
python train_task1.py --epochs 10 --lr 0.001 --batch 32
~~~

---
Then run other tasks by

~~~
python train_teracon.py --epochs 100 --lr 0.001 --batch 1024 --datapath "the data path of task" --datapath_index "item index dictionary path which generated at train_task1.py" --paths "paths of past model" --n_tasks "The total number of tasks"
~~~

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

---
To inference past tasks using current model, use inference_past_tasks.py
<br>
E.g., if train the tasks of TTL in the paper from task 1 to task 5, then inference the model about task 4.

~~~
python inference_past_tasks.py --datapath "./ColdRec/original_desen_age.csv' --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task5.t7" --n_tasks 5 --inference_task 4
~~~

## Backbone network Code
The data_loder code and basic backbone network are refered to<br>

https://github.com/yuangh-x/2022-NIPS-Tenrec

https://github.com/syiswell/NextItNet-Pytorch

---
This code is fitted to TTL data set, if you run the ML datasets please change the paths of data

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
