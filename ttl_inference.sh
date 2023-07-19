python inference_past_tasks.py --datapath "./ColdRec/original_desen_pretrain.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task5.t7" --n_tasks 5 --inference_task 1
python inference_past_tasks.py --datapath "./ColdRec/original_desen_finetune_click_nouserID.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task5.t7" --n_tasks 5 --inference_task 2
python inference_past_tasks.py --datapath "./ColdRec/original_desen_finetune_like_nouserID.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task5.t7" --n_tasks 5 --inference_task 3
python inference_past_tasks.py --datapath "./ColdRec/original_desen_age.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task5.t7" --n_tasks 5 --inference_task 4
python inference_past_tasks.py --datapath "./ColdRec/original_desen_lifestatus.csv" --datapath_index "./Data/Session/index.csv" --paths "./saved_models/task5.t7" --n_tasks 5 --inference_task 5
