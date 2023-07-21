## Task Relation-aware Continual User Representation Learning
The revised source code for [**Task Relation-aware Continual User Representation Learning**](https://arxiv.org/abs/2306.01792) paper, accepted at KDD 2023.
## Abstract
User modeling, which learns to represent users into a low-dimensional representation space based on their past behaviors, got a surge of interest from the industry for providing personalized services to users. Previous efforts in user modeling mainly focus on learning a task-specific user representation that is designed for a single task. However, since learning task-specific user representations for every task is infeasible, recent studies introduce the concept of universal user representation, which is a more generalized representation of a user that is relevant to a variety of tasks. Despite their effectiveness, existing approaches for learning universal user representations are impractical in real-world applications due to the data requirement, catastrophic forgetting and the limited learning capability for continually added tasks. In this paper, we propose a novel continual user representation learning method, called TERACON, whose learning capability is not limited as the number of learned tasks increases while capturing the relationship between the tasks. The main idea is to introduce an embedding for each task, i.e., task embedding, which is utilized to generate task-specific soft masks that not only allow the entire model parameters to be updated until the end of training sequence, but also facilitate the relationship between the tasks to be captured. Moreover, we introduce a novel knowledge retention module with pseudo-labeling strategy that successfully alleviates the long-standing problem of continual learning, i.e., catastrophic forgetting. Extensive experiments on public and proprietary real-world datasets demonstrate the superiority and practicality of TERACON.
![](https://github.com/Sein-Kim/TERACON_Revised/assets/76777494/a30959a2-95a2-414f-a4c2-49c216d728ee)
## Dataset
- 논문에서 사용된 Dataset (`TTL` and `Movielens`)은 다음 링크에서 다운로드 할 수 있습니다. (taken from [CONURE](https://arxiv.org/abs/2009.13724))<br>
  - `TTL`: https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view<br>
  - `MovieLens`: https://grouplens.org/datasets/movielens/25m/
- 위의 Dataset이 아닌, 다른 Dataset을 사용할 경우 다음 format 을 맞추어 사용하십시오: <br>
  - Format: `Input Sequence` `,,` `Targets` <br>
  - For e.g.,<br>
    ~~~
    0,0,0,0,0,0,1,2,3,4,,2,3,4
    0,0,0,0,0,1,2,5,7,3,,8
    0,0,0,5,6,7,2,2,4,5,,10
    0,0,0,0,0,8,9,3,4,4,,20
    ~~~
  - 더 자세한 예시로는 `example` 폴더 내의 example datasets 을 참고하십시오.
## Requirments
- Pytorch version: 1.7.1
- Numpy version: 1.19.2
## How to Run
~~~
git clone https://github.com/Sein-Kim/TERACON_Revised.git
cd TERACON_Revised
mkdir -p saved_model Data/Session ColdRec
~~~
- `TTL` data 를 [다음 링크](https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view) 에서 다운로드 한 후, `ColdRec` 폴더에 업로드 하십시오.
- 첫번째 Task (Task 1) 을 학습하기 위하여, 다음과 같이 `train_task1.py` 를 실행하십시오:
  ~~~
  python train_task1.py --epochs 10 --lr 0.001 --batch 32
  ~~~
- 그 후, 후속 Task 진행을 위하여, 다음 script 를 실행하십시오:
  ~~~
  sh ttl_train.sh
  ~~~
- 과거 Task 에 대한 inference 를 진행하기 위해서, `inference_past_task.py` 를 실행하십시오.
  - 만약, `TTL` datasset 을 사용하여 Task 1 부터 Task 5 까지 학습한 이후, inference 를 진행한다면, 다음 script 를 실행하십시오:
    ~~~
    sh ttl_inference.sh
    ~~~
## Arguments
- `--datapath:` Dataset 의 경로.<br>
	- usage example :`--dataset ./ColdRec/original_desen_finetune_click_nosuerID.csv`
- `--paths:` 이전 Task 를 학습한 모델의 경로 .<br>
	- usage example :`--paths ./saved_models/task1.t7`
- `--savepath:` 현재 모델을 저장할 경로.<br>
	- usage example : `--savepath ./saved_models/task2`
- `--n_tasks:`  Task 의 총 개수.<br>
	- usage example :`--n_tasks 2`
- `--datapath_index:` Item index dictionary 의 경로 (i.e., `Data/Session/index.csv`).<br>
	- usage example :`--datapath_index Data/Session/index.csv`
  - `index.csv` 파일은 Task 1 을 진행할 시 자동으로 생성됩니다.
  모델이 `Task 1` 을 학습할 때, `data_loader` 에서 Task 1 의 모든 item 의 index 정보를 가지고 있는 `index.csv` 를 생성합니다.<br>
- `--lr:` Learning rate.<br>
	- usage example : `--lr 0.0001`
- `--alpha:` Knowledge retention 의 강도를 조절하는 hyper parameter.<br>
	- usage example : `--alpha 0.7`
- `--smax:` Positive scaling 을 설정하는 hyper-parameter.<br>
	- usage example : `--smax 50`
## Source code of the backbone network
- Backbone network 의 source code 는 다음 링크를 참고하십시오:
  - https://github.com/yuangh-x/2022-NIPS-Tenrec
  - https://github.com/syiswell/NextItNet-Pytorch
## Cite (Bibtex)
- Please cite the following paer, if you find TERACON useful in your research:
  - Kim, Sein and Lee, Namkyeong and Kim, Donghyun and Yang, Minchul and Park, Chanyoung. “Task Relation-aware Continual User Representation Learning.” KDD 2023.
  - Bibtex
```
@article{kim2023task,
  title={Task Relation-aware Continual User Representation Learning},
  author={Kim, Sein and Lee, Namkyeong and Kim, Donghyun and Yang, Minchul and Park, Chanyoung},
  journal={arXiv preprint arXiv:2306.01792},
  year={2023}
}
```