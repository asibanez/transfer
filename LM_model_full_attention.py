# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# DataClass definition
class LM_dataset(Dataset):
    def __init__(self, data_df):
        self.str_data_tensor = torch.FloatTensor(data_df.drop(columns = ['LOSS_DESCRIPTION',
                                                                         'LEAF_NODE_TEXTS',
                                                                         'full_denial']).values)
        self.loss_desc_tensor = torch.LongTensor(data_df['LOSS_DESCRIPTION'].to_list())
        self.leaf_node_texts_tensor = torch.LongTensor(data_df['LEAF_NODE_TEXTS'].to_list())
        self.full_denial_tensor = torch.Tensor(data_df['full_denial'].to_list())
        
    def __len__(self):
        return self.full_denial_tensor.size()[0]
        
    def __getitem__(self, idx):
        X_str = self.str_data_tensor[idx, :]
        X_loss_desc = self.loss_desc_tensor[idx, :]
        X_leaf_node_texts = self.leaf_node_texts_tensor[idx, :]
        Y = self.full_denial_tensor[idx]
        
        return X_str, X_loss_desc, X_leaf_node_texts, Y

# Model definition
class LM_model(nn.Module):
    def __init__(self, args, pretrained_embeddings_leaf_nodes,
                 pretrained_embeddings_loss_desc, num_str_vars):
        #super(ECHR_model, self).__init__()
        super().__init__()

        self.num_layers = 1
        self.output_size = 1
        self.num_str_vars = num_str_vars
        self.dropout = args.dropout
        self.input_size = args.embed_dim
        self.h_dim = args.hidden_dim
        self.att_dim = args.att_dim
        self.num_leaf_nodes = args.num_leaf_nodes
        self.seq_len = args.seq_len
        self.query_v_att = nn.Parameter(torch.randn((self.att_dim, 1), requires_grad = True))

        # Embedding
        self.embed_leaf_nodes = nn.Embedding.from_pretrained(pretrained_embeddings_leaf_nodes)
        self.embed_loss_desc = nn.Embedding.from_pretrained(pretrained_embeddings_loss_desc)
        
        # Dropout
        self.drops = nn.Dropout(self.dropout)
        
        # Encode loss description
        self.lstm_loss_desc = nn.LSTM(input_size = self.input_size,
                                      hidden_size = self.h_dim,
                                      num_layers = self.num_layers,
                                      bidirectional = True,
                                      batch_first = True)      
        
        # Encode leaf nodes
        self.lstm_leaf_node = nn.LSTM(input_size = self.input_size,
                                      hidden_size = self.h_dim,
                                      num_layers = self.num_layers,
                                      bidirectional = True,
                                      batch_first = True)
        
        # Encode tree
        self.lstm_tree = nn.LSTM(input_size = self.h_dim * 2,
                                 hidden_size = self.h_dim,
                                 num_layers = self.num_layers,
                                 bidirectional = True,
                                 batch_first = True)
                       
        # Fully connected query vector
        self.fc_query = nn.Linear(in_features = self.h_dim * 2,
                                  out_features = self.att_dim)

        # Fully connected context node
        self.fc_context_node = nn.Linear(in_features = self.h_dim * 2,
                                         out_features = self.att_dim)

        # Fully connected context tree
        self.fc_context_tree = nn.Linear(in_features = self.h_dim * 2,
                                         out_features = self.att_dim)
        
        # Fully connected structured data
        self.bn_struct = nn.BatchNorm1d(self.num_str_vars)
        self.fc_struct = nn.Linear(in_features = self.num_str_vars,
                                   out_features = int(self.num_str_vars/3))     
        
        # Fully connected concat(loss_desc, tree, structured) -> Sigmoid
        self.bn_output = nn.BatchNorm1d(self.h_dim*2*2 + int(self.num_str_vars/3))
        self.fc_output = nn.Linear(in_features = self.h_dim*2*2 + int(self.num_str_vars/3),
                                   out_features = self.output_size)
        
        # Sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_str, X_loss_desc, X_leaf_node_texts):
        
        bilstm_b = self.h_dim
        bilstm_e = self.h_dim * 2

        # Embedding
        x_loss_desc = self.embed_loss_desc(X_loss_desc)                # batch_size x seq_len x embed_dim
        x_leaf_nodes = self.embed_leaf_nodes(X_leaf_node_texts)        # batch_size x (seq_len x n_leaf nodes) x embed_dim
        
        # Loss description encoding
        self.lstm_loss_desc.flatten_parameters()
        x_loss_desc = self.lstm_loss_desc(x_loss_desc)                 # Tuple (len = 2)
        x_loss_desc_fwd = x_loss_desc[0][:, -1, 0:bilstm_b]            # batch_size x hidden_dim
        x_loss_desc_bkwd = x_loss_desc[0][:, 0, bilstm_b:bilstm_e]     # batch_size x hidden_dim
        x_loss_desc = torch.cat((x_loss_desc_fwd, x_loss_desc_bkwd),
                                dim = 1)                               # batch_size x (hidden_dim x 2)
        x_loss_desc = self.drops(x_loss_desc)                          # batch_size x (hidden_dim x 2) 
        
        # Query vector co-attention
        query_v_coatt = self.fc_query(x_loss_desc).unsqueeze(2)        # batch_size x att_dim x 1
        
        # Leaf nodes encoding
        x_tree_dict = {}       
 
        for idx in range(0, self.num_leaf_nodes):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)
            x_single_leaf_node = x_leaf_nodes[:, span_b:span_e, :]     # batch_size x seq_len x embed_dim
            self.lstm_leaf_node.flatten_parameters()
            x_aux = self.lstm_leaf_node(x_single_leaf_node)[0]         # batch_size x seq_len x (hidden_dim x 2)
            x_aux = self.drops(x_aux)                                  # batch_size x seq_len x (hidden_dim x 2)
            # Co-attention loss description - leaf node
            projection = torch.tanh(self.fc_context_node(x_aux))       # batch_size x seq_len x att_dim
            alpha = torch.bmm(projection, query_v_coatt)               # batch_size x seq_len x 1
            alpha = torch.softmax(alpha, dim = 1)                      # batch_size x seq_len x 1
            att_output = x_aux * alpha                                 # batch_size x seq_len x (hidden_dim x 2)
            att_output = torch.sum(att_output, axis = 1)               # batch_size x (hidden_dim x 2)            
            att_output = att_output.unsqueeze(1)                       # batch_size x 1 x (hidden_dim x 2)
            x_tree_dict[idx] = att_output                              # batch_size x 1 x (hidden_dim x 2)            

        x_tree = torch.cat(list(x_tree_dict.values()), dim = 1)        # batch_size x n_leaf_nodes x (hidden_dim x 2)
        
        # Tree encoding attention
        projection = torch.tanh(self.fc_context_tree(x_tree))          # batch_size x n_leaf nodes x att_dim
        alpha = torch.matmul(projection, self.query_v_att)             # batch_size x n_leaf nodes x 1
        alpha = torch.softmax(alpha, dim = 1)                          # batch_size x n_leaf nodes x 1
        att_output = x_tree * alpha                                    # batch_size x n_leaf nodes x (hidden_dim x 2)
        att_output = torch.sum(att_output, axis = 1)                   # batch_size x (hidden_dim x 2)            
        x_tree = att_output                                            # batch_size x (hidden_dim x 2)

        # Structured data
        x_str = self.bn_struct(X_str)                                  # batch_size x num_str_vars
        x_str = self.fc_struct(x_str)                                  # batch_size x (num_str_vars / 3)
        x_str = F.relu(x_str)                                          # batch_size x (num_str_vars / 3)
        x_str = self.drops(x_str)                                      # batch_size x (num_str_vars / 3)
                                        
        # Concatenate loss description, tree encodings, structured vars     
        x = torch.cat((x_loss_desc, x_tree, x_str), dim = 1)           # batch size x (hidden_dim x 2 x 2 + num_str_vars / 3)
        x = self.bn_output(x)                                          # batch size x (hidden_dim x 2 x 2 + num_str_vars / 3)
        x = self.fc_output(x)                                          # batch size x output_size
        
        # Sigmoid function
        x = self.sigmoid(x)                                            # batch size x output_size
        
        return x
