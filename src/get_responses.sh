## Done

# CUDA_VISIBLE_DEVICES=1 nohup python src/generate_responses.py --lm tinyllama --statements automatic500 > logs/tinyllama.log& # 1/3 GPU 
# CUDA_VISIBLE_DEVICES=1 nohup python src/generate_responses.py --lm phi2 --statements automatic500 > logs/phi2.log&
# CUDA_VISIBLE_DEVICES=2,3 nohup python src/generate_responses.py --lm alpaca --statements automatic500 > logs/alpaca.log&
# CUDA_VISIBLE_DEVICES=2,3 nohup python src/generate_responses.py --lm llama-7b --statements automatic500 > logs/llama-7b.log& # 2*1/2 GPU 
# CUDA_VISIBLE_DEVICES=4,5 nohup python src/generate_responses.py --lm llama-7b-chat --statements automatic500 > logs/llama-7b-chat.log&
# CUDA_VISIBLE_DEVICES=4,5 nohup python src/generate_responses.py --lm llama-13b --statements automatic500 > logs/llama-13b.log&
# CUDA_VISIBLE_DEVICES=6,7 nohup python src/generate_responses.py --lm llama-13b-chat --statements automatic500 > logs/llama-13b-chat.log&
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup python src/generate_responses.py --lm llama-70b --statements automatic500 > logs/llama-70b.log& # needs 6 GPUs
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup python src/generate_responses.py --lm llama-70b-chat --statements automatic500 > logs/llama-70b-chat.log&
# CUDA_VISIBLE_DEVICES=6 nohup python src/generate_responses.py --lm falcon-7b --statements automatic500 > logs/falcon-7b.log&
# CUDA_VISIBLE_DEVICES=7 nohup python src/generate_responses.py --lm falcon-7b-instruct --statements automatic500 > logs/falcon-7b-instruct.log&

# nohup python src/generate_responses.py --lm gpt-3.5 --statements automatic500 > logs/gpt-3.5.log&
# nohup python src/generate_responses.py --lm gpt-4 --statements automatic500 > logs/gpt-4.log&

# CUDA_VISIBLE_DEVICES=7 nohup python src/generate_responses.py --lm distilbert --statements automatic500 > logs/distilbert.log&
# CUDA_VISIBLE_DEVICES=7 nohup python src/generate_responses.py --lm distilroberta --statements automatic500 > logs/distilroberta.log&
# CUDA_VISIBLE_DEVICES=7 nohup python src/generate_responses.py --lm electra-small --statements automatic500 > logs/electra-small.log&
# CUDA_VISIBLE_DEVICES=0 nohup python src/generate_responses.py --lm roberta-base --statements automatic500 > logs/roberta-base.log&
# CUDA_VISIBLE_DEVICES=0 nohup python src/generate_responses.py --lm roberta-large --statements automatic500 > logs/roberta-large.log&
# CUDA_VISIBLE_DEVICES=1 nohup python src/generate_responses.py --lm bert-base --statements automatic500 > logs/bert-base.log&
# CUDA_VISIBLE_DEVICES=1 nohup python src/generate_responses.py --lm bert-large --statements automatic500 > logs/bert-large.log&

## Test
# CUDA_VISIBLE_DEVICES=3,4 nohup python src/generate_responses.py --lm llama-7b --statements test --generate > logs/test_generation.log&
# CUDA_VISIBLE_DEVICES=3,4 nohup python src/generate_responses.py --lm llama-7b --statements test > logs/test_llama7.log&