CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port=20001 train.py \
--model anole --data scand --data_dir [dataset path] --decoder_type anole --image_seq_length 784 --input_format anole --output output --note train --report_to none --do_train --bfloat16

