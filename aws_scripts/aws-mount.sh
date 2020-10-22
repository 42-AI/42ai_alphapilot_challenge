NAME=`lsblk | grep 100G | cut -d ' ' -f 1`
if [ -d "/data" ]; then
    sudo mkdir /data
fi
sudo mount /dev/$NAME /data
