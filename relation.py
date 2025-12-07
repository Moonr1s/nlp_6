import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

def load_data(file_path):
    """
    解析数据文件。
    格式: ID * Entity1 * Entity2 * Rest
    Rest 部分: Type1 Type2 SubType1 SubType2 Head1 Head2 Sentence... Relation
    """
    sentences = []
    features = []  # 存放类型等结构化特征
    labels = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # 按照 ' * ' 分割，通常应该有4部分
            parts = line.split(' * ')
            
            if len(parts) != 4:
                # 处理部分格式不一致的行（少量数据可能格式特殊，这里选择跳过或仅记录日志）
                # print(f"Skipping malformed line {line_num} in {file_path}")
                continue
                
            entity1_text = parts[1]
            entity2_text = parts[2]
            rest_part = parts[3]
            
            # 解析剩余部分，以空格分隔
            rest_tokens = rest_part.split()
            
            # 确保剩余部分至少包含类型信息、中心语、句子和标签
            # 结构: T1 T2 ST1 ST2 H1 H2 ...Sentence... Relation
            if len(rest_tokens) < 8:
                continue
            
            # 提取元数据特征
            type1 = rest_tokens[0]
            type2 = rest_tokens[1]
            subtype1 = rest_tokens[2]
            subtype2 = rest_tokens[3]
            head1 = rest_tokens[4]
            head2 = rest_tokens[5]
            
            # 标签通常是最后一个 token
            relation_label = rest_tokens[-1]
            
            # 句子是中间的部分
            sentence_tokens = rest_tokens[6:-1]
            sentence_text = " ".join(sentence_tokens)
            
            # 构建样本
            # 文本特征输入
            sentences.append(sentence_text)
            
            # 结构化特征输入 (Type1, Type2, SubType1, SubType2)
            # 也可以加入 Entity1 和 Entity2 的文本作为特征，或者 Head1, Head2
            features.append([type1, type2, subtype1, subtype2])
            
            labels.append(relation_label)
            
    return sentences, np.array(features), np.array(labels)

def train_and_evaluate(train_file, test_file):
    print("正在加载数据...")
    # 1. 加载数据
    train_sentences, train_features, y_train = load_data(train_file)
    test_sentences, test_features, y_test = load_data(test_file)
    
    print(f"训练集大小: {len(train_sentences)}")
    print(f"测试集大小: {len(test_sentences)}")
    
    # 2. 特征工程与管道构建
    # 我们将结合 文本特征 (TF-IDF) 和 类别特征 (One-Hot Encoding)
    
    # 文本特征提取器
    text_transformer = TfidfVectorizer(
        tokenizer=lambda x: x.split(), # 假设中文已经分词，如果没有分词可以使用 list(x) 按字处理
        ngram_range=(1, 2), 
        max_features=10000
    )
    
    # 类别特征编码器 (处理 Type1, Type2, SubType1, SubType2)
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # 组合预处理器
    # ColumnTransformer 无法直接混合 List 和 Array，我们需要自定义处理或分开处理
    # 这里为了简单，我们先分别提取特征，然后水平拼接 (hstack)
    
    print("正在提取特征...")
    # 训练集特征
    X_train_text = text_transformer.fit_transform(train_sentences)
    X_train_cat = cat_transformer.fit_transform(train_features)
    
    # 将稀疏矩阵拼接
    from scipy.sparse import hstack
    X_train = hstack([X_train_text, X_train_cat])
    
    # 测试集特征 (使用 transform 而不是 fit_transform)
    X_test_text = text_transformer.transform(test_sentences)
    X_test_cat = cat_transformer.transform(test_features)
    X_test = hstack([X_test_text, X_test_cat])
    
    # 3. 模型训练
    print("正在训练模型 (Linear SVC)...")
    # 使用线性支持向量机，适合高维稀疏特征
    clf = LinearSVC(class_weight='balanced', random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 4. 预测与评估
    print("正在评估模型...")
    y_pred = clf.predict(X_test)
    
    print("\n=== 模型分类报告 ===")
    # 这里 zero_division=0 是为了处理测试集中可能出现的未见类别警告
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print(f"总体准确率: {accuracy_score(y_test, y_pred):.4f}")
    
    return clf

# --- 主程序入口 ---
if __name__ == "__main__":
    # 请根据您实际的数据路径修改这里
    # 注意：All_8_features_train.txt 包含了正例和负例(Negative)
    base_dir = "data" # 假设数据在当前目录的 data 文件夹下
    train_path = os.path.join(base_dir, "All_8_features_train.txt")
    test_path = os.path.join(base_dir, "All_8_features_test.txt")
    
    # 检查文件是否存在
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_and_evaluate(train_path, test_path)
    else:
        print("未找到数据文件，请检查路径。")
