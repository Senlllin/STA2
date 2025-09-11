CUDA                            10.2    ~   11.1
python                          3.7
torch                           1.8.0   ~   1.9.0
numpy
lmdb
msgpack-numpy
ninja  
termcolor
tqdm
open3d  
h5py



编译 PointNet++ 算子

cd code/util/pointnet2\_ops\_lib

python setup.py install

```​:codex-file-citation\[codex-file-citation]{line\_range\_start=25 line\_range\_end=28 path=README.md git\_url="https://github.com/Senlllin/STA\_dg/blob/main/README.md#L25-L28"}​  



代码结构

对称几何工具：normalize\_plane、reflect\_points 等函数位于 code/models/sta/sta\_math.py，负责平面归一化、点反射和镜像距离计算

对称参数头：STAParamHead 通过多层感知机预测平面法向量、偏移量及置信度

对称注意力：STAAttentionBias 根据镜像邻近度构建注意力偏置，SymAwareDecoder 在解码器中注入该偏置

损失函数：STALosses 提供 Chamfer 距离、平面正则与几何对称损失的加权组合

配置读取：STAConfig.from\_yaml 可从 YAML 加载 use\_sta、k\_mirror 等参数



配置文件

默认配置位于 configs/sta.yaml，可调节是否启用 STA 及镜像注意力、镜像邻居数、损失权重、日志级别等



训练

使用 code/network/train.py 进行端到端训练，需要提供 LMDB 数据集路径、类别名等参数：

python code/network/train.py \\

&nbsp; --lmdb\_train <path\_train.lmdb> \\

&nbsp; --lmdb\_valid <path\_valid.lmdb> \\

&nbsp; --lmdb\_sn <path\_shapenet.lmdb> \\

&nbsp; --class\_name all \\

&nbsp; --batch\_size 4 \\

&nbsp; --max\_epoch 480



推理/测试

使用 code/network/test.py 载入训练好的权重并导出点云结果：

python code/network/test.py \\

&nbsp; --lmdb\_valid <path\_valid.lmdb> \\

&nbsp; --lmdb\_sn <path\_shapenet.lmdb> \\

&nbsp; --class\_name all \\

&nbsp; --log\_dir weights/STA \\

&nbsp; --last\_epoch 120



对称性评估

提供 tools/eval\_sta.py 对单个点云进行对称性打分：

python tools/eval\_sta.py \\

&nbsp; --input sample.txt \\

&nbsp; --n 0 1 0 \\

&nbsp; --d 0.0 \\

&nbsp; --log-level INFO


数据集组织与调用方式
必需的 LMDB 文件

RealCom 训练集 realcom_data_train.lmdb

RealCom 验证/测试集 realcom_data_test.lmdb

ShapeNet 完整点云库 shapenet_data.lmdb
每个 LMDB 记录包含：

RealCom：[data_id, incomplete_pcd, complete_pcd, model_T, model_R, model_S]，在读取时会裁剪到 
[
−
1
,
1
]
3
[−1,1] 
3
  并按 input_pn/gt_pn 重采样

ShapeNet：[data_id, points]，同样重采样至 gt_pn

文件存放位置

推荐在仓库内新建目录 data/：

STA_dg/
  data/
    RealComData/
      realcom_data_train.lmdb
      realcom_data_test.lmdb
    RealComShapeNetData/
      shapenet_data.lmdb
只要路径正确，LMDB 可放在任意位置；通过命令行参数指定即可

训练与验证调用

python code/network/train.py \
  --lmdb_train data/RealComData/realcom_data_train.lmdb \
  --lmdb_valid data/RealComData/realcom_data_test.lmdb \
  --lmdb_sn data/RealComShapeNetData/shapenet_data.lmdb \
  --class_name all \
  --batch_size 4
脚本内部会调用 RealComGANDataset 将 RealCom 与 ShapeNet 数据混合用于训练/验证

测试/推理调用

python code/network/test.py \
  --lmdb_valid data/RealComData/realcom_data_test.lmdb \
  --lmdb_sn data/RealComShapeNetData/shapenet_data.lmdb \
  --class_name all \
  --log_dir weights/STA
PCN 数据集（可选）

若使用 PcnDataset，需按如下目录组织：

path/
  complete/<class_id>/<model>.pcd
  partial/<class_id>/<model>/00.pcd
通过 PCNGANDataset 或 PcnDataset 调用，用于 PCN 结构的训练/评估

# STA
