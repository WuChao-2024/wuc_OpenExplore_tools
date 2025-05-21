
import numpy as np

datas = []

datas.append(np.load("origin_actions.npy"))
datas.append(np.load("new_actions.npy"))

# datas.append(np.load("can0feature.npy"))
# datas.append(np.load("new_cv2_can0feature.npy"))

# datas.append(np.load("origin_observation.images.laptop_meanstd.npy"))
# datas.append(np.load("new_cv2_observation.images.laptop_meanstd.npy"))

# datas.append(np.load("new_can0feature.npy"))
# datas.append(np.load("new_cv2_can0feature.npy"))

for data in datas:
    print(f"{data.shape = }")
    print(f"max: {np.max(data)}, min: {np.min(data)}, mean: {np.mean(data)}")

print()

diff = datas[0] - datas[1]
print(f"diff: {np.max(diff)=}  {np.min(diff)=}  {np.mean(diff)=}")

# 计算余弦相似度
def cosine_similarity(A, B):
    # 将张量展平为一维向量
    A_flat = A.flatten()
    B_flat = B.flatten()
    # 计算点积和范数
    dot_product = np.dot(A_flat, B_flat)
    norm_A = np.linalg.norm(A_flat)
    norm_B = np.linalg.norm(B_flat)
    # 避免除以零
    if norm_A == 0 or norm_B == 0:
        return 0
    return dot_product / (norm_A * norm_B)

# 计算并打印余弦相似度
cos_sim = cosine_similarity(datas[0], datas[1])
print(f"\nCosine Similarity: {cos_sim:.4f}")
