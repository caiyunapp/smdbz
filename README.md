# smdbz
Sparse Matrix dBZ 稀疏矩阵 dBZ 等级数据读取

## 安装
```bash
$ pip install git+https://github.com/caiyunapp/smdbz.git
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