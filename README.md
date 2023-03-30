#How to train?

##Setup environment
Install packages as:
pip install Demo.yaml

##Training
To train DMICC, we take main.py as an example, run the following commands sequentially to perform our method on STL-10:

python main.py -g 0,1,2,3 -n 8 --batch_size 512 --epochs 5000 --pretrained checkpoint_dir/checkpoint.pth
