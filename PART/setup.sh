source /miniconda/bin/activate
apt update
echo "apt update is done!"
apt install screen rsync
echo "install rsync is done!"
aws --endpoint-url xxx/imagenet.zip . --quiet
echo "Imagenet is downloaded!"
unzip -q imagenet.zip
echo "Imagenet is unzipped!"
