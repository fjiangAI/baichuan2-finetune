nohup accelerate launch --config_file inference.yaml \
    --deepspeed_multinode_launcher standard inference.py \
    --model_path /mntcephfs/data/med/jiangfeng/KST/Baichuan2/normal \
    --data_path ./data/test_single_turn.json \
    --out_file ./output.json \
    > inference.log &