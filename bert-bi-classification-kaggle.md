# Kaggle入门竞赛-对推特灾难文本二分类

[Natural Language Processing with Disaster Tweets | Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data)

使用BERT（transformers库）对推特灾难文本二分类

xxx着火了（灾难）

火烧云像是燃烧的火焰（非灾难）



```python
import os
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# 用于加载bert模型的分词器
from transformers import AutoTokenizer
# 用于加载bert模型
from transformers import AutoModel
from pathlib import Path
from tqdm.notebook import tqdm



```


```python
batch_size = 16
# 文本的最大长度
text_max_length = 128
epochs = 100
# 取多少训练集的数据作为验证集
validation_ratio = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 每多少步，打印一次loss
log_per_step = 50

# 数据集所在位置
dataset_dir = Path("/kaggle/input/nlp-getting-started/")
os.makedirs(dataset_dir) if not os.path.exists(dataset_dir) else ''

# 模型存储路径
model_dir = Path("/kaggle/working/")
# 如果模型目录不存在，则创建一个
os.makedirs(model_dir) if not os.path.exists(model_dir) else ''

print("Device:", device)



```

    Device: cuda


# 数据处理

加载数据集，查看文本最大长度


```python
pd_data = pandas.read_csv(dataset_dir / 'train.csv')
pd_data
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7608</th>
      <td>10869</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Two giant cranes holding a bridge collapse int...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7609</th>
      <td>10870</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>@aria_ahrary @TheTawniest The out of control w...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7610</th>
      <td>10871</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M1.94 [01:04 UTC]?5km S of Volcano Hawaii. htt...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7611</th>
      <td>10872</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Police investigating after an e-bike collided ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7612</th>
      <td>10873</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>The Latest: More Homes Razed by Northern Calif...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>7613 rows × 5 columns</p>

</div>


```python
pd_data = pandas.read_csv(dataset_dir / 'train.csv')[['text', 'target']]
pd_data
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7608</th>
      <td>Two giant cranes holding a bridge collapse int...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7609</th>
      <td>@aria_ahrary @TheTawniest The out of control w...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7610</th>
      <td>M1.94 [01:04 UTC]?5km S of Volcano Hawaii. htt...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7611</th>
      <td>Police investigating after an e-bike collided ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7612</th>
      <td>The Latest: More Homes Razed by Northern Calif...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>7613 rows × 2 columns</p>

</div>

在使用BERT进行文本分类时，文本序列会被分段成较小的片段，每个片段的长度不能超过BERT模型的最大输入长度。BERT-base模型的最大输入长度为512个token。但是，实际上，通常不会使用整个512个token的长度，因为这会导致模型的计算和内存消耗过高，尤其是在GPU内存有限的情况下。

因此，为了在保持模型性能的同时有效利用计算资源，常见的做法是将文本序列截断或填充到一个较小的长度，通常是128或者256。在这个设置下，大多数文本序列都可以被完整地处理，而且不会导致过多的计算资源消耗。

选择128作为文本最大长度的原因可能是出于以下考虑：

大多数文本序列可以在128个token的长度内完整表示，因此不会丢失太多信息。
128是一个相对合理的长度，可以平衡模型性能和计算资源的消耗。
在实际应用中，较长的文本序列很少出现，因此选择128不会对大多数样本产生太大影响。


```python
max_length = pd_data['text'].str.len().max()
print(max_length)
# 按ratio随机划分训练集和验证集
pd_validation_data = pd_data.sample(frac = validation_ratio)
pd_train_data = pd_data[~pd_data.index.isin(pd_validation_data.index)]
pd_train_data
"""
输出：157
"""
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>#RockyFire Update =&gt; California Hwy. 20 closed...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7607</th>
      <td>#stormchase Violent Record Breaking EF-5 El Re...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7608</th>
      <td>Two giant cranes holding a bridge collapse int...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7610</th>
      <td>M1.94 [01:04 UTC]?5km S of Volcano Hawaii. htt...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7611</th>
      <td>Police investigating after an e-bike collided ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7612</th>
      <td>The Latest: More Homes Razed by Northern Calif...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>6852 rows × 2 columns</p>

</div>


```python
#定义数据类
class MyDataset(Dataset):
    def __init__(self,mode = 'train'):
        super(MyDataset,self).__init__()
        self.mode = mode
        if mode == 'train':
            self.dataset = pd_train_data
        elif mode == 'validation':
            self.dataset = pd_validation_data
        elif mode == 'test':
            # 如果是测试模式，则返回推文和id。拿id做target主要是方便后面写入结果。
            self.dataset = pandas.read_csv(dataset_dir / 'test.csv')[['text', 'id']]
        else:
            raise Exception("Unknown mode {}".format(mode))
    
    def __getitem__(self, index):
        # 取第index条
        data = self.dataset.iloc[index]
        # 取其推文，做个简单的数据清理
        source = data['text'].replace("#", "").replace("@", "")
        # 取对应的推文
        if self.mode == 'test':
            # 如果是test，将id做为target
            target = data['id']
        else:
            target = data['target']
        # 返回推文和target
        return source, target

    def __len__(self):
        return len(self.dataset)
train_dataset = MyDataset('train')
validation_dataset = MyDataset('validation')
train_dataset.__getitem__(0)
"""
('Our Deeds are the Reason of this earthquake May ALLAH Forgive us all', 1)
"""
```


```python
#使用分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer("I'm learning deep learning", return_tensors='pt')


"""
tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]
config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]
vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]
tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]

{'input_ids': tensor([[ 101, 1045, 1005, 1049, 4083, 2784, 4083,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
"""
```

下面这个collate_fn函数用于对一个batch的文本数据进行处理，将文本句子转换为tensor，并组成一个batch。下面是函数的具体功能和输入输出：

输入参数 batch：一个batch的句子，每个句子是一个元组，包含文本和目标标签，例如：[('推文1', 目标1), ('推文2', 目标2), ...]

输出：处理后的结果，包含两部分：

src：是要送给BERT模型的输入，包含两个tensor：
input_ids：经过分词和映射后的输入文本的token id序列。
attention_mask：用于指示BERT模型在进行自注意力机制时哪些部分是padding的，需要被忽略。1表示有效token，0表示padding。
target：目标标签的tensor序列，即对应每个文本的标签。
这个函数首先将输入的batch分成两个列表，一个是文本列表 text，一个是目标标签列表 target。然后使用 tokenizer 对文本进行分词、映射、padding和裁剪等预处理操作，得到模型的输入 src。最后将处理后的输入 src 和目标标签 target 组合成输出。


collate_fn函数在数据加载器每次取出一个batch的样本时被调用，用于对这个batch的样本进行预处理、转换成模型所需的格式。


```python
def collate_fn(batch):
    """
    将一个batch的文本句子转成tensor，并组成batch。
    :param batch: 一个batch的句子，例如: [('推文', target), ('推文', target), ...]
    :return: 处理后的结果，例如：
             src: {'input_ids': tensor([[ 101, ..., 102, 0, 0, ...], ...]), 'attention_mask': tensor([[1, ..., 1, 0, ...], ...])}
             target：[1, 1, 0, ...]
    """
    text, target = zip(*batch)
    text, target = list(text), list(target)

    # src是要送给bert的，所以不需要特殊处理，直接用tokenizer的结果即可
    # padding='max_length' 不够长度的进行填充
    # truncation=True 长度过长的进行裁剪
    src = tokenizer(text, padding='max_length', max_length=text_max_length, return_tensors='pt', truncation=True)

    return src, torch.LongTensor(target)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
inputs, targets = next(iter(train_loader))
print("inputs:", inputs)
print(inputs['input_ids'].shape)
print("targets:", targets)
#batch_size = 16

"""
inputs: {'input_ids': tensor([[  101, 10482,  6591,  ...,     0,     0,     0],
        [  101,  4911,  2474,  ...,     0,     0,     0],
        [  101,  5916,  6340,  ...,     0,     0,     0],
        ...,
        [  101, 21318,  2571,  ...,     0,     0,     0],
        [  101, 20010, 21149,  ...,     0,     0,     0],
        [  101, 26934,  5315,  ...,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]])}
torch.Size([16, 128])
targets: tensor([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1])

"""
```


768是BERT模型中隐藏层的维度大小。BERT模型使用了12层或者24层的Transformer编码器，每一层的隐藏层输出的维度大小为768

nn.Linear(768, 256)：将输入的维度从768降到256，这是一个线性变换（全连接层），将BERT模型输出的768维隐藏表示转换为更低维度的表示。

nn.ReLU()：ReLU激活函数，用于引入非线性。在降维后的表示上应用ReLU激活函数，以增加模型的非线性能力。

nn.Linear(256, 1)：将256维的表示进一步映射到一个单一的值，用于二分类问题中的概率预测。

nn.Sigmoid()：Sigmoid激活函数，将输出值压缩到0到1之间，表示概率值。

因此，整个self.predictor模块的作用是将BERT模型的输出映射到一个单一的概率值，用于二分类问题的预测。


```python
#构建模型
class TextClassificationModel(nn.Module):
    def __init__(self):
        super(TextClassificationModel, self).__init__()

        # 加载bert模型
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        # 最后的预测层
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, src):
        """
        :param src: 分词后的推文数据
        """

        # 将src直接序列解包传入bert，因为bert和tokenizer是一套的，所以可以这么做。
        # 得到encoder的输出，用最前面[CLS]的输出作为最终线性层的输入
        outputs = self.bert(**src).last_hidden_state[:, 0, :]

        # 使用线性层来做最终的预测
        return self.predictor(outputs)

    
"""

last_hidden_state 的形状是 (batch_size, sequence_length, hidden_size)，其中：

batch_size 是当前批次中样本的数量。
sequence_length 是输入序列的长度。
hidden_size 是隐藏状态的维度，通常等于BERT模型的隐藏层大小，例如在BERT-base中是768。
"""
```




```python
model = TextClassificationModel()
model = model.to(device)
model(inputs.to(device))
criteria = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
# 由于inputs是字典类型的，定义一个辅助函数帮助to(device)
def to_device(dict_tensors):
    result_tensors = {}
    for key, value in dict_tensors.items():
        result_tensors[key] = value.to(device)
    return result_tensors
"""
将字典中的张量转移到指定的设备（如GPU）。它接受一个字典，其中键是张量的名称，值是张量本身。
然后，它迭代字典中的每个键值对，并将值转移到设备上，最后返回一个具有相同键但值位于指定设备上的新字典
"""



def validate():
    model.eval()
    total_loss = 0.
    total_correct = 0
    for inputs, targets in validation_loader:
        inputs, targets = to_device(inputs), targets.to(device)
        outputs = model(inputs)
        loss = criteria(outputs.view(-1), targets.float())
        total_loss += float(loss)

        correct_num = (((outputs >= 0.5).float() * 1).flatten() == targets).sum()
        total_correct += correct_num

    return total_correct / len(validation_dataset), total_loss / len(validation_dataset)
"""
model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]
"""
```

```python
# 首先将模型调成训练模式
model.train()

# 清空一下cuda缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 定义几个变量，帮助打印loss
total_loss = 0.
# 记录步数
step = 0

# 记录在验证集上最好的准确率
best_accuracy = 0

# 开始训练
for epoch in range(epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        # 从batch中拿到训练数据
        inputs, targets = to_device(inputs), targets.to(device)
        # 传入模型进行前向传递
        outputs = model(inputs)
        # 计算损失
        loss = criteria(outputs.view(-1), targets.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += float(loss)
        step += 1

        if step % log_per_step == 0:
            print("Epoch {}/{}, Step: {}/{}, total loss:{:.4f}".format(epoch+1, epochs, i, len(train_loader), total_loss))
            total_loss = 0

        del inputs, targets

    # 一个epoch后，使用过验证集进行验证
    accuracy, validation_loss = validate()
    print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch+1, accuracy, validation_loss))
    torch.save(model, model_dir / f"model_{epoch}.pt")

    # 保存最好的模型
    if accuracy > best_accuracy:
        torch.save(model, model_dir / f"model_best.pt")
        best_accuracy = accuracy

```

    Epoch 1/100, Step: 49/429, total loss:27.0852
    Epoch 1/100, Step: 99/429, total loss:21.9039
    Epoch 1/100, Step: 149/429, total loss:22.6578
    Epoch 1/100, Step: 199/429, total loss:21.1815
    Epoch 1/100, Step: 249/429, total loss:20.3617
    Epoch 1/100, Step: 299/429, total loss:18.9497
    Epoch 1/100, Step: 349/429, total loss:20.8270
    Epoch 1/100, Step: 399/429, total loss:20.0272
    Epoch 1, accuracy: 0.8279, validation loss: 0.0247
    Epoch 2/100, Step: 20/429, total loss:18.0542
    Epoch 2/100, Step: 70/429, total loss:14.7096
    Epoch 2/100, Step: 120/429, total loss:15.0193
    Epoch 2/100, Step: 170/429, total loss:14.2937
    Epoch 2/100, Step: 220/429, total loss:14.1752
    Epoch 2/100, Step: 270/429, total loss:14.2685
    Epoch 2/100, Step: 320/429, total loss:14.0682
    Epoch 2/100, Step: 370/429, total loss:16.1425
    Epoch 2/100, Step: 420/429, total loss:17.1818
    Epoch 2, accuracy: 0.8397, validation loss: 0.0279
    Epoch 3/100, Step: 41/429, total loss:8.0204
    Epoch 3/100, Step: 91/429, total loss:9.5614
    Epoch 3/100, Step: 141/429, total loss:9.2036
    Epoch 3/100, Step: 191/429, total loss:8.9964
    Epoch 3/100, Step: 241/429, total loss:10.7305
    Epoch 3/100, Step: 291/429, total loss:10.5000
    Epoch 3/100, Step: 341/429, total loss:11.3632
    Epoch 3/100, Step: 391/429, total loss:10.3103
    Epoch 3, accuracy: 0.8252, validation loss: 0.0339
    Epoch 4/100, Step: 12/429, total loss:8.1302
    Epoch 4/100, Step: 62/429, total loss:5.9590
    Epoch 4/100, Step: 112/429, total loss:6.9333
    Epoch 4/100, Step: 162/429, total loss:6.4659
    Epoch 4/100, Step: 212/429, total loss:6.3636
    Epoch 4/100, Step: 262/429, total loss:6.6609
    Epoch 4/100, Step: 312/429, total loss:6.3064
    Epoch 4/100, Step: 362/429, total loss:5.7218
    Epoch 4/100, Step: 412/429, total loss:6.8676
    Epoch 4, accuracy: 0.8042, validation loss: 0.0370
    Epoch 5/100, Step: 33/429, total loss:4.4049
    Epoch 5/100, Step: 83/429, total loss:3.0673
    Epoch 5/100, Step: 133/429, total loss:4.1351
    Epoch 5/100, Step: 183/429, total loss:3.8803
    Epoch 5/100, Step: 233/429, total loss:3.2633
    Epoch 5/100, Step: 283/429, total loss:4.6513
    Epoch 5/100, Step: 333/429, total loss:4.3888
    Epoch 5/100, Step: 383/429, total loss:5.1710
    Epoch 5, accuracy: 0.8055, validation loss: 0.0484
    Epoch 6/100, Step: 4/429, total loss:4.7682
    Epoch 6/100, Step: 54/429, total loss:3.6112
    Epoch 6/100, Step: 104/429, total loss:4.2054
    Epoch 6/100, Step: 154/429, total loss:3.0118
    Epoch 6/100, Step: 204/429, total loss:3.7317
    Epoch 6/100, Step: 254/429, total loss:4.3987
    Epoch 6/100, Step: 304/429, total loss:4.3260
    Epoch 6/100, Step: 354/429, total loss:4.8860
    Epoch 6/100, Step: 404/429, total loss:4.0680
    Epoch 6, accuracy: 0.8200, validation loss: 0.0394
    Epoch 7/100, Step: 25/429, total loss:4.3479
    Epoch 7/100, Step: 75/429, total loss:3.1758
    Epoch 7/100, Step: 125/429, total loss:3.0595
    Epoch 7/100, Step: 175/429, total loss:3.2737
    Epoch 7/100, Step: 225/429, total loss:3.4793
    Epoch 7/100, Step: 275/429, total loss:2.8818
    Epoch 7/100, Step: 325/429, total loss:4.4013
    Epoch 7/100, Step: 375/429, total loss:4.0712
    Epoch 7/100, Step: 425/429, total loss:3.6233
    Epoch 7, accuracy: 0.8134, validation loss: 0.0505
    Epoch 8/100, Step: 46/429, total loss:2.4538
    Epoch 8/100, Step: 96/429, total loss:2.9408
    Epoch 8/100, Step: 146/429, total loss:2.0388
    Epoch 8/100, Step: 196/429, total loss:2.2719
    Epoch 8/100, Step: 246/429, total loss:3.0254
    Epoch 8/100, Step: 296/429, total loss:3.1964
    Epoch 8/100, Step: 346/429, total loss:4.5933
    Epoch 8/100, Step: 396/429, total loss:2.3120
    Epoch 8, accuracy: 0.8121, validation loss: 0.0532
    Epoch 9/100, Step: 17/429, total loss:2.1302
    Epoch 9/100, Step: 67/429, total loss:2.8609
    Epoch 9/100, Step: 117/429, total loss:2.5762
    Epoch 9/100, Step: 167/429, total loss:2.5297
    Epoch 9/100, Step: 217/429, total loss:3.0031
    Epoch 9/100, Step: 267/429, total loss:2.8550
    Epoch 9/100, Step: 317/429, total loss:2.7038
    Epoch 9/100, Step: 367/429, total loss:2.0410
    Epoch 9/100, Step: 417/429, total loss:1.6495
    Epoch 9, accuracy: 0.8147, validation loss: 0.0584
    Epoch 10/100, Step: 38/429, total loss:2.4684
    Epoch 10/100, Step: 88/429, total loss:2.5976
    Epoch 10/100, Step: 138/429, total loss:2.2212
    Epoch 10/100, Step: 188/429, total loss:2.0417
    Epoch 10/100, Step: 238/429, total loss:1.8892
    Epoch 10/100, Step: 288/429, total loss:2.2668
    Epoch 10/100, Step: 338/429, total loss:2.4390
    Epoch 10/100, Step: 388/429, total loss:2.4793
    Epoch 10, accuracy: 0.8068, validation loss: 0.0606
    Epoch 11/100, Step: 9/429, total loss:2.3416
    Epoch 11/100, Step: 59/429, total loss:2.9766
    Epoch 11/100, Step: 109/429, total loss:2.4573
    Epoch 11/100, Step: 159/429, total loss:2.1148
    Epoch 11/100, Step: 209/429, total loss:1.9889
    Epoch 11/100, Step: 259/429, total loss:2.2313
    Epoch 11/100, Step: 309/429, total loss:1.9831
    Epoch 11/100, Step: 359/429, total loss:1.7014
    Epoch 11/100, Step: 409/429, total loss:2.2243
    Epoch 11, accuracy: 0.8173, validation loss: 0.0610
    Epoch 12/100, Step: 30/429, total loss:1.2329
    Epoch 12/100, Step: 80/429, total loss:2.0102
    Epoch 12/100, Step: 130/429, total loss:1.2435
    Epoch 12/100, Step: 180/429, total loss:1.9317



```python
model = torch.load(model_dir / f"model_best.pt")
model = model.eval()
test_dataset = MyDataset('test')
#构造测试集的dataloader。测试集是不包含target的
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
results = []
for inputs, ids in tqdm(test_loader):
    outputs = model(inputs.to(device))
    outputs = (outputs >= 0.5).int().flatten().tolist()
    ids = ids.tolist()
    results = results + [(id, result) for result, id in zip(outputs, ids)]
with open('/kaggle/working/results.csv', 'w', encoding='utf-8') as f:
    f.write('id,target\n')
    for id, result in results:
        f.write(f"{id},{result}\n")
print("Finished!")

"""
  0%|          | 0/204 [00:00<?, ?it/s]
  Finished!
"""
```

提交后的结果：

![image-20240321135333231](image-20240321135333231.png)