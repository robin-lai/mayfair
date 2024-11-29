```commandline
sudo apt-get update
sudo apt-get install vim
```
```commandline
nohup python entry_point_wdl_run_on_sg.py > log3.txt 2>&1 &
0键盘输入， 1 标准输出， 2错误输出
nohup python -u ./cmd.py > cmd.log &  -u avoid output buffing # https://stackoverflow.com/questions/12919980/nohup-is-not-writing-log-to-output-file
```
````commandline
history
sudo apt-get install util-linux
tail -f log5.text not tailf
````
```commandline
conda search "^python$"
conda create -n py37 python==3.7.0
conda activate py37  
conda deactivate
```

```text
pip install pyarrow
pip install parquet E
conda install parquet not work
```

```commandline
git fetch --all --tags --prune
git checkout tags/v1.15.0
sudo apt update
sudo apt install vim -y
```

```text
1.去除aws帐号，密码
2.用ssh更方便，并且github不支持用邮箱帐号了
3.在setting下添加
ssh-keygen -t ed25519 -C "672826043@qq.com"
vim ~/.ssh/id_ed25519 copy 到github
/home/sagemaker-user/.ssh/id_ed25519.pub
```

```text
查看系统版本
cat /proc/version
```