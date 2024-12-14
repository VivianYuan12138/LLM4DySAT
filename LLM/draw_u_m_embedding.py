from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

# 创建输出目录
os.makedirs('pic', exist_ok=True)

# 加载用户嵌入数据
user_embeddings1 = np.loadtxt('kh/_u_embed_0.txt', delimiter=',')
user_embeddings2 = np.loadtxt('kh/_u_embed_1.txt', delimiter=',')
user_embeddings3 = np.loadtxt('kh/_u_embed_2.txt', delimiter=',')

# 加载电影嵌入数据
movie_embeddings1 = np.loadtxt('kh/_m_embed_0.txt', delimiter=',')
movie_embeddings2 = np.loadtxt('kh/_m_embed_1.txt', delimiter=',')
movie_embeddings3 = np.loadtxt('kh/_m_embed_2.txt', delimiter=',')

# 合并用户嵌入和电影嵌入
user_embeddings = np.vstack((user_embeddings1, user_embeddings2, user_embeddings3))
movie_embeddings = np.vstack((movie_embeddings1, movie_embeddings2, movie_embeddings3))

# PCA 降维到2维
pca = PCA(n_components=2)
user_2d = pca.fit_transform(user_embeddings)
movie_2d = pca.fit_transform(movie_embeddings)

# 创建颜色映射
colors_user = ['blue', 'cyan', 'navy']  # 用户不同时间片段的颜色
colors_movie = ['green', 'lime', 'darkgreen']  # 电影不同时间片段的颜色

# 设置美化样式
plt.style.use('ggplot')

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 用户嵌入可视化
axes[0].scatter(user_2d[:len(user_embeddings1), 0], user_2d[:len(user_embeddings1), 1], 
                 c=colors_user[0], label='User (Time 1)', alpha=0.7)
axes[0].scatter(user_2d[len(user_embeddings1):len(user_embeddings1) + len(user_embeddings2), 0],
                 user_2d[len(user_embeddings1):len(user_embeddings1) + len(user_embeddings2), 1], 
                 c=colors_user[1], label='User (Time 2)', alpha=0.7)
axes[0].scatter(user_2d[len(user_embeddings1) + len(user_embeddings2):, 0],
                 user_2d[len(user_embeddings1) + len(user_embeddings2):, 1], 
                 c=colors_user[2], label='User (Time 3)', alpha=0.7)
axes[0].set_title('User Embeddings Visualization (2D)', fontsize=14)
axes[0].set_xlabel('PCA Component 1', fontsize=12)
axes[0].set_ylabel('PCA Component 2', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)

# 电影嵌入可视化
axes[1].scatter(movie_2d[:len(movie_embeddings1), 0], movie_2d[:len(movie_embeddings1), 1], 
                 c=colors_movie[0], label='Movie (Time 1)', alpha=0.7)
axes[1].scatter(movie_2d[len(movie_embeddings1):len(movie_embeddings1) + len(movie_embeddings2), 0],
                 movie_2d[len(movie_embeddings1):len(movie_embeddings1) + len(movie_embeddings2), 1], 
                 c=colors_movie[1], label='Movie (Time 2)', alpha=0.7)
axes[1].scatter(movie_2d[len(movie_embeddings1) + len(movie_embeddings2):, 0],
                 movie_2d[len(movie_embeddings1) + len(movie_embeddings2):, 1], 
                 c=colors_movie[2], label='Movie (Time 3)', alpha=0.7)
axes[1].set_title('Movie Embeddings Visualization (2D)', fontsize=14)
axes[1].set_xlabel('PCA Component 1', fontsize=12)
axes[1].set_ylabel('PCA Component 2', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)

# 调整布局并保存
plt.tight_layout()
plt.savefig('pic/user_movie_embeddings.png')
plt.show()
