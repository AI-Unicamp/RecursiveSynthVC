while getopts g:n:p: flag
do
        case "${flag}" in
                g)  gpu=${OPTARG};;
                n)  number=${OPTARG};;
                p)  port=${OPTARG};;
        esac
done
echo "Running container vc$number on gpu $gpu and port $port";

docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=$gpu --runtime=nvidia --userns=host --shm-size 64G -v /work/lucas.ueda/:/workspace/lucas.ueda/ -v /home/lucas.ueda/:/home/lucas.ueda/ -p $port --name vc$number voice_conversion:latest /bin/bash