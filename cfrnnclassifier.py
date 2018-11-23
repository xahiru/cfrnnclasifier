#In the one we try to find rating as classification task
# therre five buckets 1-5 for each movie
# Each user is modelled as a sequence of movies and its classes are the ratings
# For example User1 U1, rated Movies M1, M2, M3 as {2,4,5} respectively
# Given the User U2, we try to find M2,M3 {?,?}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random

torch.manual_seed(1)
testuserange = 7

# df_t = pd.read_csv("/Users/xahiru/Code/paper/mf/pytorch-tutorials/images/tiny_training2.csv")
# df_t = pd.read_csv("/Users/xahiru/Code/paper/agree/agreerecom/data/ml-100k/u.data").head(500)

df_t = pd.read_table('/Users/xahiru/Code/paper/agree/agreerecom/data/ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp']).head(2000)

model_type_basic_lstm = 1
model_type_basic_lstm_plus_trust = 2
model_type_random = 3

model_type = model_type_basic_lstm


def prepare_sequence(user_movies, to_ix):
    movie_index = []
    for movie in user_movies:
        # print(to_ix[str(movie)])
        # movie_index.append(to_ix[str(movie)])
        movie_index.append(to_ix[movie])
    return torch.tensor(movie_index, dtype=torch.long)

def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

training_data_df = encode_data(df_t)
print("training_data_df")
print(training_data_df)


def agreement(ratings, alpha =2.5, ptype="user"):
    
    if ptype=='item':
        ratings = ratings.T
    #for each unique user iterate
    for user_a in range(ratings.shape[0]):
        for user_b in range(ratings.shape[0]):
            if user_a != user_b:
                a_ratings = ratings[user_a]
                b_ratings = ratings[user_b]
                commonset = np.intersect1d(np.nonzero(a_ratings), np.nonzero(b_ratings))
                common_set_length = len(commonset)
                trust = 0
                if(common_set_length > 0):
                    a_positive = a_ratings[commonset] > alpha
                    b_positive = b_ratings[commonset] > alpha

                    agreement = np.sum(np.logical_not(np.logical_xor(a_positive, b_positive)))

                    trust = agreement/common_set_length
                trust_matrix[user_a,user_b] = trust
    return trust_matrix

# item_to_ix = {}
# training_data = []
# for index, item in training_data_df.iterrows():
    # print(item)
    # training_data.append(item)

movies_to_ix = {}
movies_to_ix = {i:i for i in training_data_df.movieId.unique()}
# user_movies = [training_data_df.movieId[i] for i in training_data_df.userId]


# tag_to_ix = {i:i-1 for i in training_data_df.rating.unique()}
# tag_to_ix = {2.5: 2.5, 3.0: 3.0, 2.0: 2.0, 4.0: 4.0, 3.5: 3.5, 1.0: 1.0, 5.0: 5.0, 4.5: 4.5, 1.5: 1.5, 0.5: 0.5}
# tag_to_ix = {i+1:i for i in range(5)}
# tag_to_ix = {"5": 4, "4": 3, "3": 2, "2": 1, "1": 0}
# tag_to_ix = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
tag_to_ix = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

# tag_to_ix = {0.5: 0, 1.0: 1, 1.5: 2, 2.0: 3, 2.5: 4, 3.0: 5, 3.5: 6, 4.0: 7, 4.5: 8, 5.0: 9}


# print("unique movies")
# print(len(training_data_df.movieId.unique()))
# movies_to_ix = {i:i for i in range(len(training_data_df.movieId.unique()))}
# print("movies_to_ix")
# print(movies_to_ix)
# print(training_data)
# training_data = []

u_index = {}
u_it_index = []
n_users = training_data_df.userId.nunique()
n_items = training_data_df.movieId.nunique()
ratings = np.zeros((n_users, n_items))
# print('ratings.shape')
# print(ratings.shape)
# print('>>>>>>training_data_df.userId')
# print(training_data_df.userId.nunique())
# print('>>>>>>training_data_df.movieId')
# print(training_data_df.movieId.nunique())
for index, row in training_data_df.iterrows():
    # print(row.userId)
    ratings[int(row.userId),int(row.movieId)] = int(row.rating)#TODO rating could be float
    if row.userId not in u_index:
        u_index[int(row.userId)] = int(row.userId) #just to track the unique users
        u_it_index.append([[int(row.movieId)],[row.rating]])
        # u_it_index.append([1][row.movieId])
    else:
        # u_it_index[row.userId].append([[row.movieId],[row.rating]])
        temp = u_it_index[int(row.userId)]
        temp[0].append(int(row.movieId))
        temp[1].append(row.rating)
        # print(u_it_index[row.userId])

print('u_it_index')  
print(u_it_index)


print("movies_to_ix")
print(movies_to_ix)
print("ratings_to_ix")
print(tag_to_ix)

print("ratings")
print(ratings)

trust_matrix = np.zeros((n_items, n_items))
trust_matrix = agreement(ratings, 2.5,'item')
#TODO sklearn.metrics.pairwise.pairwise_distances
#use simtrust sim+trust/2

print("trust_matrix")
print(trust_matrix)
trust_indices = np.argmax(trust_matrix, axis=0)
trust_values = trust_matrix.max(0)
print('trust_indices')
print(trust_indices)
print('trust_values')
print(trust_values)

tagset_size = 5 #Rating ranges from 1-5, TODO test including 0 in the range
EMBEDDING_DIM = 64
HIDDEN_DIM = 32




class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, unique_items_size, tagset_size, trust_matrix, ratings, tag_to_ix):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

    
        self.word_embeddings = nn.Embedding(unique_items_size, embedding_dim)
        # self.trust_embeddings = nn.Embedding(tagset_size, tagset_size)
        

        self.trust_matrix = trust_matrix
        self.trust_indices = np.argmax(trust_matrix, axis=0)
        self.trust_values = trust_matrix.max(0)
        self.ratings = ratings
        self.tag_to_ix = tag_to_ix

        if model_type == model_type_basic_lstm_plus_trust:
            self.trust_embeddings = nn.Embedding(tagset_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim+embedding_dim, hidden_dim)
        else if model_type == model_type_basic_lstm:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)


        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def get_trusted_ratings(self, user_movie_list, user):
        trusted_rating_list = []
        for x in user_movie_list:
            checkzero = self.trust_values[x]
            # if(checkzero==0):
            #     non_zero_items = np.nonzero(self.ratings)
            #     r = random.choice(self.ratings[non_zero_items[0],non_zero_items[1]])
            # else:
            r = self.ratings[user,x]
            if(r==0):
                item = self.ratings[:,x]
                item_r = item[np.nonzero(item)]
                r = round(np.sum(item_r)/len(item_r))

            r = self.tag_to_ix[r]
            # print(r)        
            trusted_rating_list.append(r)
        return torch.tensor(trusted_rating_list, dtype=torch.long), trusted_rating_list
        

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence, items, user):
    # def forward(self, sentence, trusted_tags):
    # def forward(self, sentence):
        # print('items')
        # print(items)
        # print('top_trusted_item')
        # print(self.trust_indices[items])
        # it = self.trust_indices[items]
        # print('get_trusted_ratings')
        trusted_tags, trusted_rating_list = self.get_trusted_ratings(self.trust_indices[items], user)

        # print('sentence')
        # print(sentence)
        # print('trusted_rating_list')
        # print(trusted_rating_list)
        embeds = self.word_embeddings(sentence)
        # with torch.no_grad():
        embeds2 = self.trust_embeddings(trusted_tags)
            # y = torch.LongTensor(5,1).random_() % 5
            # one_hot = torch.FloatTensor(5, 5).zero_()
            # target = one_hot.scatter_(1, y,1)
            # print('one hot target')
            # print(one_hot)
            # print(torch.exp(F.log_softmax(target, dim=1)))
        lstm_in = torch.cat((embeds, embeds2), 0)
        # print('embeds')
        # print(embeds.size())
        # print('embeds2')
        # print(embeds2.size())
        # print('lstm_in')
        # print(lstm_in.size())

        # print("embeds size()")
        # print(embeds)
        # print('embeds2')
        # print(embeds2)


        lstm_out, self.hidden = self.lstm(
            lstm_in.view(len(sentence), 1, -1), self.hidden)
       

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # print("tag_space")
        # # print(tag_space.size())
        # print(tag_space)

        # t_tag_space = F.log_softmax(embeds2,dim=1)

        # tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = F.log_softmax(tag_space, dim=1)
        # values, indices = torch.max(tag_scores,1)
        # print(torch.exp(t_tag_space))

        # embeds = self.word_embeddings(indices)
        # lstm_out2, self.hidden2 = self.lstm2(
        #     embeds.view(len(sentence), 1, -1), self.hidden2)

        # tag_space = self.hidden2tag(lstm_out2.view(len(sentence), -1))
        # print("tag_space")
        # print(tag_space)
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # values, indices = torch.max(tag_scores,1)

        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(movies_to_ix), len(tag_to_ix), trust_matrix, ratings, tag_to_ix)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# with torch.no_grad():
#     print(u_it_index[testuser][0])
#     print(u_it_index[testuser][1])
#     inputs = prepare_sequence_new(u_it_index[testuser][0], movies_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)

for epoch in range(30):  # again, normally you would NOT do 300 epochs, it is toy data
    u_index_count = 0
    for user_movie_list, user_rating_list in u_it_index:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        # print("k")
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        # sentence_in = prepare_sequence(user_movie_list, movies_to_ix)
        sentence_in = torch.tensor(user_movie_list, dtype=torch.long)
        targets = prepare_sequence(user_rating_list, tag_to_ix)
        # trusted_tags = get_trusted_ratings(trust_indices[user_movie_list], u_index_count)
        # print(user_rating_list)
        # targets = torch.tensor(user_rating_list, dtype=torch.long)
        # targets = prepare_sequence(user_rating_list, tag_to_ix)

        # print("sentence_in")
        # print(sentence_in)


        # Step 3. Run our forward pass.
        # tag_scores = model(sentence_in, trusted_tags)
        tag_scores = model(sentence_in, user_movie_list, u_index_count)
        # print("tag_scores")
        # print(tag_scores)
        # values, indices = torch.max(tag_scores,1)
        # print("targets n indices")
        # print(targets)
        # print(indices)


        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores,targets)
        # print('loss')
        # print(loss)
        loss.backward()
        optimizer.step()
        u_index_count = u_index_count + 1
    print(loss)

