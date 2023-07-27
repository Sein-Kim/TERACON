python train_task1.py --epochs 10 --lr 0.001 --seed 0
python train_teracon.py --lr 0.001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_finetune_click_nosuerID.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task1.t7" --savepath "./saved_models/task2" --n_tasks 2 --seed 0
python train_teracon.py --lr 0.0001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_finetune_like_nosuerID.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task2.t7" --savepath "./saved_models/task3" --n_tasks 3 --seed 0
python train_teracon.py --lr 0.0001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_age.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task3.t7" --savepath "./saved_models/task4" --n_tasks 4 --seed 0
python train_teracon.py --lr 0.0001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_gender.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task4.t7" --savepath "./saved_models/task5" --n_tasks 5 --seed 0
python train_teracon.py --lr 0.0001 --smax 50 --batch 1024 --datapath "./ColdRec/original_desen_lifestatus.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task5.t7" --savepath "./saved_models/task6" --n_tasks 6 --seed 0
