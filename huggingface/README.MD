
# BERT 资源准备

预训练权重不随代码仓库分发。

从 [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main)
下载 `config.json`、`pytorch_model.bin`、`tokenizer_config.json`、`tokenizer.json`、`vocab.txt`，
放入本目录下的 `google_bert/`。请下载实际权重，而非 Git LFS 指针或 HTML 错误页面。
无需下载 TensorFlow/Flax 格式或其他 LLM。

当前入口默认查找 `huggingface/google_bert/`；如使用其他目录，创建实验时传入
`--bert-path "实际BERT目录"`。记录模型版本，开始实验后不要替换资源或改变续训配置。
具体环境、运行命令与输入信息量差异见根目录 README 的“快速使用”。
