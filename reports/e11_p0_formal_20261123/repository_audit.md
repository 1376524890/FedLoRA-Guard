# E11 P0 仓库与数据审计

- 分支: experiment/e11-matched-warehouse-conditional-20261123 @ af5a6888ab749fc5456261a7b5e590c0b49d7e0c
- 工作区: ?? configs/e11/
?? reports/e11_p0_formal_20261123/
?? src/e11/
?? tests/test_e11_algebra.py
- V4 商品: {('train', 'defective'): 20, ('dev', 'defective'): 256, ('dev', 'normal'): 128, ('test', 'defective'): 256, ('test', 'normal'): 128, ('spare', 'defective'): 64, ('spare', 'normal'): 48}
- spare 缺陷类型分布: {'D01': 7, 'D02': 7, 'D03': 7, 'D04': 7, 'D05': 6, 'D06': 6, 'D07': 6, 'D08': 6, 'D09': 6, 'D10': 6}
- Dolly: 候选 1500，有效 1483（跳过 17），池 283
- PKU: 候选 1500，有效 1487（跳过 13），池 287
- 训练商品 30 = 20 train + spare ['WL-A1519', 'WL-B1936', 'WL-C9969', 'WL-D2081', 'WL-D4232', 'WL-D4386', 'WL-D8334', 'WL-F2701', 'WL-G6642', 'WL-K1361']
- 题对仓库分配: {'chengdong': 15, 'beigang': 15}
- 评价集: {'dev': {'defective:huayu': 256, 'defective:chengdong': 256, 'defective:beigang': 256, 'normal:huayu': 128, 'normal:chengdong': 128, 'normal:beigang': 128}, 'test': {'defective:huayu': 256, 'defective:chengdong': 256, 'defective:beigang': 256, 'defective:nanzhan': 256, 'normal:huayu': 128, 'normal:chengdong': 128, 'normal:beigang': 128, 'normal:nanzhan': 128}}
- GPU: 0, NVIDIA GeForce GTX 1080 Ti, 11264 MiB
1, NVIDIA GeForce GTX 1080 Ti, 11264 MiB
2, NVIDIA GeForce GTX 1080 Ti, 11264 MiB