import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

def load_data(file_path):
    """
    解析数据文件。
    格式: ID * Entity1 * Entity2 * Rest
    Rest 部分: Type1 Type2 SubType1 SubType2 Head1 Head2 Sentence... Relation
    """
    sentences = []
    features = []  # 存放类型等结构化特征
    labels = []
    
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return [], np.array([]), np.array([])

    print(f"正在加载文件: {os.path.basename(file_path)} ...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # 按照 ' * ' 分割，通常应该有4部分
            parts = line.split(' * ')
            
            if len(parts) != 4:
                continue
                
            # entity1_text = parts[1] # 未使用
            # entity2_text = parts[2] # 未使用
            rest_part = parts[3]
            
            # 解析剩余部分，以空格分隔
            rest_tokens = rest_part.split()
            
            # 确保剩余部分至少包含类型信息、中心语、句子和标签
            if len(rest_tokens) < 8:
                continue
            
            # 提取元数据特征
            type1 = rest_tokens[0]
            type2 = rest_tokens[1]
            subtype1 = rest_tokens[2]
            subtype2 = rest_tokens[3]
            # head1 = rest_tokens[4] # 可选特征
            # head2 = rest_tokens[5] # 可选特征
            
            # 标签通常是最后一个 token
            relation_label = rest_tokens[-1]
            
            # 句子是中间的部分
            sentence_tokens = rest_tokens[6:-1]
            sentence_text = " ".join(sentence_tokens)
            
            # 构建样本
            sentences.append(sentence_text)
            features.append([type1, type2, subtype1, subtype2])
            labels.append(relation_label)
            
    return sentences, np.array(features), np.array(labels)

def run_experiment(train_files, eval_files):
    """
    使用指定的训练文件列表进行训练，并在评估文件字典上进行测试
    """
    
    # --- 1. 加载并合并训练数据 ---
    print("="*30)
    print("正在准备训练数据...")
    all_train_sentences = []
    all_train_features = []
    all_train_labels = []

    for t_file in train_files:
        s, f, l = load_data(t_file)
        if len(s) > 0:
            all_train_sentences.extend(s)
            # 处理 features 的合并 (第一次直接赋值，后面vstack)
            if len(all_train_features) == 0:
                all_train_features = f
            else:
                all_train_features = np.vstack((all_train_features, f))
            
            # 处理 labels 的合并
            if len(all_train_labels) == 0:
                all_train_labels = l
            else:
                all_train_labels = np.concatenate((all_train_labels, l))

    print(f"训练集总大小: {len(all_train_sentences)}")

    # --- 2. 特征工程 ---
    print("\n正在构建特征管道...")
    # 文本特征 (TF-IDF)
    text_transformer = TfidfVectorizer(
        tokenizer=lambda x: x.split(), 
        ngram_range=(1, 2), 
        max_features=10000
    )
    
    # 类别特征 (One-Hot)
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # 拟合训练数据
    print("正在提取训练集特征...")
    X_train_text = text_transformer.fit_transform(all_train_sentences)
    X_train_cat = cat_transformer.fit_transform(all_train_features)
    
    # 合并特征
    X_train = hstack([X_train_text, X_train_cat])
    
    # --- 3. 模型训练 ---
    print(f"\n正在训练模型 (Linear SVC) ...")
    # class_weight='balanced' 会自动处理因合并 Positive 数据可能带来的类别不平衡变化
    clf = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    clf.fit(X_train, all_train_labels)
    print("模型训练完成。")
    
    # --- 4. 多数据集评估 ---
    print("\n" + "="*30)
    print("开始评估...")
    
    results = {}
    
    for name, path in eval_files.items():
        print(f"\n>>> 正在评估数据集: {name}")
        if not os.path.exists(path):
            print(f"文件跳过 (不存在): {path}")
            continue
            
        test_sentences, test_features, y_test = load_data(path)
        
        if len(test_sentences) == 0:
            print("数据为空，跳过。")
            continue
            
        # 提取特征 (Transform only)
        X_test_text = text_transformer.transform(test_sentences)
        X_test_cat = cat_transformer.transform(test_features)
        X_test = hstack([X_test_text, X_test_cat])
        
        # 预测
        y_pred = clf.predict(X_test)
        
        # 输出报告
        acc = accuracy_score(y_test, y_pred)
        print(f"[{name}] 准确率: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        results[name] = acc

    return clf, results

# --- 主程序入口 ---
if __name__ == "__main__":
    base_dir = "data" # 假设数据在 data 文件夹下 (根据您的文件路径结构，可能需要调整)
    # 如果您的代码与data目录同级，这通常是正确的。
    # 根据您提供的文件名，这些文件似乎在一个很深的目录里，请确保 base_dir 指向包含 .txt 文件的目录
    # 或者直接使用绝对路径/相对路径列表。
    
    # 定义文件列表
    # 我们将 All 和 Positive 的训练集都用于训练（Positive 作为过采样增强）
    train_files = [
        os.path.join(base_dir, "All_8_features_train.txt"),
        os.path.join(base_dir, "Positive_8_features_train.txt")
    ]
    
    # 定义需要评估的文件字典
    eval_files = {
        "Dev (All)": os.path.join(base_dir, "All_8_features_dev.txt"),
        "Test (All)": os.path.join(base_dir, "All_8_features_test.txt"),
        "Dev (Positive)": os.path.join(base_dir, "Positive_8_features_dev.txt"),
        "Test (Positive)": os.path.join(base_dir, "Positive_8_features_test.txt")
    }
    
    # 检查至少主训练文件存在
    if os.path.exists(train_files[0]):
        run_experiment(train_files, eval_files)
    else:
        print(f"错误: 找不到主训练文件 {train_files[0]}")
        print("请检查 base_dir 路径设置。")