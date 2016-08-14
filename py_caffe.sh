for req in $(cat python/requirements.txt);do pip3 install $req; done

# for python3 
sudo pip3 install protobuf --upgrade
sudo pip3 install python-dateutil --upgrade
sudo pip3 install protobuf==3.0.0b2
sudo pip3 install easydict
sudo apt install python3-tk
