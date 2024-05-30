# smdbz
Sparse Matrix dBZ 稀疏矩阵 dBZ 等级数据读取

## 安装
克隆项目到本地
```bash
$ git clone https://github.com/caiyunapp/smdbz.git
$ cd smdbz
$ python setup.py install # 开发模式执行 python setup.py develop
```

## 测试
```bash
$ cd tests && python test_all.py
```

## 使用
```python
from smdbz import read_smdbz

smdbz = read_smdbz("./global.smdbz")
print(smdbz[2041, 1529])
```