MAX_LENGTH=128
BERT_MODEL=bert-base-multilingual-cased

OUTPUT_DIR=user-bias
NUM_EPOCHS=1
BATCH_SIZE=128
SAVE_STEPS=750
SEED=1

#python3 run_yelp.py --data_dir ./ \
#--model_type bert \
#--labels ./labels.txt \
#--model_name_or_path ${BERT_MODEL} \
#--output_dir ${OUTPUT_DIR} \
#--max_seq_length  ${MAX_LENGTH} \
#--num_train_epochs ${NUM_EPOCHS} \
#--per_gpu_train_batch_size ${BATCH_SIZE} \
#--save_steps ${SAVE_STEPS} \
#--seed ${SEED} \
#--do_train \
#--do_eval \
#--do_predict

python3 run_yelp.py --data_dir ./ \
--model_type bert \
--labels ./labels.txt \
--model_name_or_path ${BERT_MODEL} \
--output_dir ${OUTPUT_DIR} \
--max_seq_length  ${MAX_LENGTH} \
--num_train_epochs ${NUM_EPOCHS} \
--per_gpu_train_batch_size ${BATCH_SIZE} \
--save_steps ${SAVE_STEPS} \
--seed ${SEED} \
--do_train \
--do_eval \
--do_predict \
--user_bias