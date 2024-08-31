DJ_PATH=/home/ubuntu/gu/data-juicer/
CODE_YAML=steel_project_starcode.yaml
TXT_YAML=steel_project_txt.yaml

nohup python $DJ_PATH/tools/process_data.py --config $CODE_YAML > dj_log_code 2>&1 &
# nohup python $DJ_PATH/tools/process_data.py --config $TXT_YAML > dj_log_txt 2>&1 &
