from datasets import load_dataset
import pandas as pd
import os


from datasets import load_dataset

train_ds = load_dataset(
    "parquet",
    data_files="/root/datasets/DeepEyes-Datasets-47k/train.parquet",
    split="train"
)
test_ds = load_dataset(
    "parquet",
    data_files="/root/datasets/DeepEyes-Datasets-47k/test.parquet",
    split="train"
)
def extract_answer(example):
    example["answer"] = example["extra_info"]["answer"]
    return example

train_ds = train_ds.map(extract_answer)
test_ds = test_ds.map(extract_answer)
output_dir = "/root/datasets/DeepEyes-Datasets-47k/"
print(train_ds[0])
train_path = os.path.join(output_dir, "train.parquet")
test_path = os.path.join(output_dir, "test.parquet")
train_ds.to_parquet(train_path)
test_ds.to_parquet(test_path)
# print(train_ds[0])
# print(f"Saved train ({n_train}) -> {train_path}")
# print(f"Saved test ({n_test}) -> {test_path}")

# import os
# import io
# from PIL import Image
# from datasets import load_dataset, Dataset, DatasetDict
# import pandas as pd

# def preprocess_sample(sample):
#     """
#     将原始 VLM/Tool 样本整理成兼容老框架的格式
#     """
#     # 1. 图片处理：bytes -> PIL
#     if "images" in sample and sample["images"] is not None:
#         preprocessed_images = []
#         for img in sample["images"]:
#             if isinstance(img, dict) and "bytes" in img:
#                 try:
#                     pil_img = Image.open(io.BytesIO(img["bytes"]))
#                     preprocessed_images.append(pil_img)
#                 except Exception as e:
#                     print(f"[WARN] Failed to load image: {e}")
#         sample["preprocessed_images"] = preprocessed_images
#     else:
#         sample["preprocessed_images"] = []

#     # 2. 问题和答案
#     if "extra_info" in sample:
#         sample["problem"] = sample["extra_info"].get("question", "")
#         sample["answer"] = sample["extra_info"].get("answer", "")
#     elif "reward_model" in sample:
#         sample["problem"] = sample.get("problem", "")
#         sample["answer"] = sample.get("reward_model", {}).get("ground_truth", "")
#     else:
#         sample["problem"] = sample.get("problem", "")
#         sample["answer"] = sample.get("answer", "")

#     # 3. messages 处理
#     if "messages" in sample:
#         # 保留原 messages
#         messages = sample["messages"]
#         # 如果原 messages 没有 assistant 内容，自动生成
#         if len(messages) == 1 or all(m.get("role") != "assistant" for m in messages):
#             messages.append({
#                 "role": "assistant",
#                 "content": f"Answer: \\boxed{{{sample['answer']}}}"
#             })
#         sample["messages"] = messages
#     else:
#         # 原始 messages 不存在，构造
#         sample["messages"] = [
#             {"role": "user", "content": sample["problem"]},
#             {"role": "assistant", "content": f"{sample['answer']}"}
#         ]

#     return sample

# def split_and_save_parquet(input_parquet, output_dir, test_ratio=0.1):
#     """
#     1. 读取 parquet
#     2. 整理字段
#     3. 顺序切分 train/test
#     4. 保存到指定目录
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # 1. 加载数据
#     ds = load_dataset("parquet", data_files=input_parquet, split="train")

#     # 2. 预处理
#     ds = ds.map(preprocess_sample)

#     # 3. 顺序切分
#     n_total = len(ds)
#     n_test = int(n_total * test_ratio)
#     n_train = n_total - n_test

#     train_ds = ds.select(range(n_train))
#     test_ds = ds.select(range(n_train, n_total))

#     # 4. 保存 parquet
#     train_path = os.path.join(output_dir, "train.parquet")
#     test_path = os.path.join(output_dir, "test.parquet")
#     train_ds.to_parquet(train_path)
#     test_ds.to_parquet(test_path)
#     print(train_ds[0])
#     print(f"Saved train ({n_train}) -> {train_path}")
#     print(f"Saved test ({n_test}) -> {test_path}")

# # ===== 使用示例 =====
# if __name__ == "__main__":
#     input_parquet = "/data/DeepEyes-Datasets-47k/data_0.1.2_visual_toolbox_v2.parquet"
#     output_dir = "/root/datasets/geo3k_imgurl/"
#     split_and_save_parquet(input_parquet, output_dir, test_ratio=0.1)
