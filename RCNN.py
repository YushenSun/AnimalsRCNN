# 定义目标类别标签列表
classes = ['elephant', 'lion', 'giraffe', 'zebra', 'rhino', 'non-animal']

# 为每个目标类别分配唯一的整数编码
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# 你可以根据需要定义更多的目标类别和编码

# 打印类别标签及其对应的整数编码
for cls, idx in class_to_idx.items():
    print(f'{cls}: {idx}')
