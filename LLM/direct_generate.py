import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# 初始化BERT的分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def loadusers():
    file_path = 'u.user'
    table = ['UserID', 'Age', 'Gender', 'Occupation']

    data = np.loadtxt(file_path, dtype=str)
    dat_set = []
    for dat in data:
        u_dat = dat.split("|")[:-1]
        dat_set.append(u_dat)

    data_set = np.array(dat_set)
    dat_line1 = data_set[:, 0]  # user
    dat_line2 = np.where(data_set[:, 2] == 'F', 0, 1)
    dat_line3 = []

    for age in data_set[:, 1]:
        if int(age) < 10:
            dat_line3.append('0')
        elif int(age) < 20:
            dat_line3.append('1')
        elif int(age) < 30:
            dat_line3.append('2')
        elif int(age) < 40:
            dat_line3.append('3')
        elif int(age) < 50:
            dat_line3.append('4')
        elif int(age) < 60:
            dat_line3.append('5')
        else:
            dat_line3.append('6')

    dat_line3 = np.array(dat_line3)
    dat_line4 = data_set[:, 3]
    dat_line4_map = {j: i for i, j in enumerate(set(dat_line4))}
    idx_dat_line4 = np.array(list(map(dat_line4_map.get, dat_line4)))

    data_all = np.concatenate([np.reshape(dat_line1, [-1, 1]), np.reshape(dat_line2, [-1, 1]),
                               np.reshape(dat_line3, [-1, 1]), np.reshape(idx_dat_line4, [-1, 1])], axis=1).astype(int)

    np.savetxt('kh/users_attributes.txt', data_all, fmt='%d', delimiter=',', encoding='utf-8')
    return data_set

def loadmovies():
    file_path = 'u.item.txt'
    mv_type = ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
               'Fantasy', 'Film - Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci - Fi', 'Thriller', 'War', 'Western']
    mv_att = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            mv_att.append([])
            row = line.replace("\n", "").split('|')
            mv_id = row[0]
            mv_att[-1].append(str(mv_id))
            ttype = row[-19:]
            for iddx, ty in enumerate(ttype):
                if ty == '1':
                    mv_att[-1].append(str(iddx))

    with open('kh/movies_attributes.txt', 'w', encoding='utf-8') as file:
        for r in mv_att:
            file.write(','.join(r) + '\n')
    return mv_att

def loadratings(user_att, movie_att):
    file_path = 'u.data'
    dat_set = np.loadtxt(file_path, dtype=int, delimiter='\t')
    indices = np.argsort(dat_set[:, 3])
    sorted_dat_set = dat_set[indices]
    
    partition_points = np.linspace(0, len(sorted_dat_set), 11).astype(int)
    partitioned_data = [sorted_dat_set[partition_points[i]:partition_points[i + 1]] for i in range(10)]

    u_dict_list = []
    for partition in partitioned_data:
        u_m_dict = {}
        for edge in partition:
            if edge[0] not in u_m_dict.keys():
                u_m_dict[edge[0]] = [movie_att[edge[1] - 1][1:]]
            else:
                u_m_dict[edge[0]].append(movie_att[edge[1] - 1][1:])
        u_dict_list.append(u_m_dict)

    m_dict_list = []
    for partition in partitioned_data:
        m_u_dict = {}
        for edge in partition:
            if edge[1] not in m_u_dict.keys():
                m_u_dict[edge[1]] = [user_att[edge[0] - 1][1:]]
            else:
                m_u_dict[edge[1]].append(user_att[edge[0] - 1][1:])
        m_dict_list.append(m_u_dict)

    return u_dict_list, m_dict_list

def LLM_to_txt(num, savefilename, dataset_):
    for idxx, dataset in enumerate(dataset_):
        feature = []
        print(idxx, savefilename.format(idxx))
        with open(savefilename.format(idxx), 'w', encoding='utf-8') as savefile:
            for i in range(num):
                if i in dataset.keys():
                    try:
                        text_data = " ".join([" ".join(item) for item in dataset[i]])
                        encoded_inputs = tokenizer.encode_plus(text_data, truncation=True, max_length=512,
                                                               padding='max_length', return_tensors="pt")
                        with torch.no_grad():
                            outputs = bert_model(**encoded_inputs)
                        last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                        feature.append(last_hidden_states)
                    except Exception as e:
                        feature.append(np.zeros(768))
                        print(f"Error processing dataset {i}: {e}")
                else:
                    feature.append(np.zeros(768))
        feature_arr = np.array(feature)
        np.savetxt(savefilename.format(idxx), feature_arr, fmt='%.3f', delimiter=',', encoding='utf-8')

if __name__ == '__main__':
    user_att = loadusers()
    movie_att = loadmovies()
    u_dict_list, m_dict_list = loadratings(user_att, movie_att)
    LLM_to_txt(num=943, savefilename='kh/_u_embed_{}.txt', dataset_=u_dict_list)
    print('User embeddings finished')
    LLM_to_txt(num=1682, savefilename='kh/_m_embed_{}.txt', dataset_=m_dict_list)
    print('Movie embeddings finished')
