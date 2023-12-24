import json

# 文件路径
input_file = 'belle_chat_ramdon_10k.json'
test_output_file = 'test.json'
train_output_file = 'train.json'

# 读取数据
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 确保数据是一个列表并且有足够的元素
if isinstance(data, list) and len(data) > 1000:
    # 分割数据
    test_set = data[:1000]
    train_set = data[1000:]

    # 保存测试集
    with open(test_output_file, 'w', encoding='utf-8') as file:
        json.dump(test_set, file, ensure_ascii=False, indent=4)
    
    # 保存训练集
    with open(train_output_file, 'w', encoding='utf-8') as file:
        json.dump(train_set, file, ensure_ascii=False, indent=4)
else:
    print("数据不是一个列表或者列表中的元素少于1000个。")
