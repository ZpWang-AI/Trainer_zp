
script_dir="/home/qwe/test/zpwang/Trainer/src/scripts"
log_dir="/home/qwe/test/zpwang/Trainer/logs"

torun_file=$(find $script_dir -type f -name "${1}*")
num_files=$(echo $torun_file | wc -l)
if [ $num_files -ne 1 ]; then
    echo $torun_file
    echo "wrong num of files"
fi

start_time=$(date +%Y-%m-%d-%H-%M-%S)
filename=$(basename "$torun_file")
filename="${filename%.*}"
echo $torun_file
echo "start running"

log_path="${log_dir}/${start_time}.${filename}.log"
nohup /home/qwe/miniconda3/envs/zpwang_main/bin/python $torun_file > $log_path 2>&1 &
