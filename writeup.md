## 2 Byte-Pair Encoding (BPE) Tokenizer
### 为什么用通常用utf-8格式构造分词器：
1. 如果直接使用Unicode 码点，需要覆盖非常多的 Unicode 字符，导致词表会大且稀疏；如果编码为字节序列（byte list），则固定为 256 个可能的字节值，且编码保证可以转换回原始字符，不会出现“词表外（OOV）”问题
2. 通常选择UTF‑8格式，因为占用长度更短。 
   1. 对于英文字符： UTF‑8 通常 1 字节；UTF‑16 通常 2 字节；UTF‑32 固定为 4 字节
   2. UTF‑16/UTF‑32 还需要考虑端序和BOM标记

###  Byte-Pair Encoding （BPE）
首先预分词（Pre-tokenization），将原始文档粗粒度划分为tokens，然后合并高频tokens
BPE 通过迭代合并最高频的相邻符号对来构建词汇表
规则1：不跨越预分词边界
规则2：频率相同时，选择字典序更大的对

Problem(train_bpe):BPE Tokenizer Training
1. 对原始文档按分块，方便并行处理，划分边界时注意special_token
2. 再按special_token获得所有子文档，保证原始边界不干扰合并
3. 对每块文档进行预分词，使用GPT-2 风格的预分词规则
4. 迭代合并最高频的相邻tokens。记录每个pair的出现频率，因为合并时只改变被合并pair的频率，从而优化性能