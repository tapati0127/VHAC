blink_train_biencoder:
	CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/model/blink/biencoder/train_biencoder.py \
												--data_path data/  \
                        --path_to_model static/biencoder_v1/pytorch_model.bin \
												--output_path static/biencoder  \
												--learning_rate 1e-5  \
												--num_train_epochs 20  \
												--max_context_length 100 \
												--max_cand_length 300 \
												--train_batch_size 8 \
												--eval_batch_size 8  \
												--bert_model xlm-roberta-base \
												--type_optimization all_encoder_layers  \
												--print_interval 100 \
												--eval_interval 9000 \
												--shuffle\
												--data_parallel  

blink_get_top_k_cands:
	CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python blink/biencoder/eval_biencoder.py \
												--path_to_model static/biencoder/pytorch_model.bin \
												--cand_encode_path static/cand_encode.t7\
												--data_path data/blink/blink_format \
												--output_path static \
												--encode_batch_size 8 \
												--eval_batch_size 1 \
												--top_k 20 \
												--save_topk_result \
												--bert_model xlm-roberta-base \
												--mode train,valid \
												--entity_dict_path data/documents.jsonl\
												--data_parallel 

blink_train_crossencoder:
	CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python blink/crossencoder/train_cross.py \
												--data_path static/top20_candidates \
												--output_path static/crossencoder \
												--learning_rate 2e-05 \
												--num_train_epochs 3 \
												--max_context_length 100 \
												--max_cand_length 300 \
												--max_seq_length 400 \
												--train_batch_size 2 \
												--eval_batch_size 2 \
												--bert_model xlm-roberta-base \
												--type_optimization all_encoder_layers \
												--add_linear \
												--eval_interval 7500 \
												--print_interval 100\
												--shuffle\
												--output_eval_file static/crossencoder/eval.txt\
												--warmup_proportion 0.1\
												--data_parallel \
												--top_k 20                                         
                                               

blink_get_encode_vector:
	CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python blink/biencoder/eval_biencoder.py \
												--path_to_model static/biencoder/pytorch_model.bin \
											    --cand_encode_path static/candidate_encode/cand_encode.t7\
											    --output_path static/top_k_candidates \
												--encode_batch_size 16 \
												--eval_batch_size 8 \
												--top_k 16 \
												--bert_model static/bert-base-uncased \
												--mode train,valid,test \
												--entity_dict_path static/documents_new/documents.jsonl

train_all:
	CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python src/model/blink/biencoder/train_biencoder.py \
												--data_path data/blink/blink_format  \
												--output_path static/biencoder  \
												--learning_rate 1e-5  \
												--num_train_epochs 2  \
												--max_context_length 32 \
												--max_cand_length 128 \
												--train_batch_size 8 \
												--eval_batch_size 8  \
												--bert_model ./bert-base-uncased \
												--type_optimization all_encoder_layers  \
												--print_interval 100 \
												--eval_interval 9000 \
												--shuffle\
												--data_parallel 
                                                
	CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python blink/biencoder/eval_biencoder.py \
												--path_to_model static/biencoder/pytorch_model.bin \
												--cand_encode_path static/candidate_encode/cand_encode.t7\
												--data_path data/blink/blink_format \
												--output_path static/top_k_candidates \
												--encode_batch_size 4 \
												--eval_batch_size 1 \
												--top_k 20 \
												--save_topk_result \
												--bert_model ./bert-base-uncased \
												--mode train,valid \
												--entity_dict_path data/blink/documents/documents.jsonl\
												--data_parallel 

	CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python blink/crossencoder/train_cross.py \
												--data_path static/top_k_candidates/top20_candidates \
												--output_path static/crossencoder \
												--learning_rate 2e-05 \
												--num_train_epochs 5 \
												--max_context_length 32 \
												--max_cand_length 128 \
												--max_seq_length 160 \
												--train_batch_size 3 \
												--eval_batch_size 3 \
												--bert_model ./bert-base-uncased \
												--type_optimization all_encoder_layers \
												--add_linear \
												--eval_interval 7500 \
												--print_interval 100\
												--data_parallel  
                                                