import os
import tqdm
import torch
import numpy as np
import pandas as pd
import time
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, pre_tokenizers, models, trainers

def bin_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to bin the `Age` feature of the user into implicity age categories.

    :Paramters:
    df: pandas dataframe for which the binning of the age feature needs to take place

    :Returns:
    dataframe with transformed `Age` feature
    """

    choice_list, condition_list = list(), list()
    default_value = 'Age_0'
    min_age_list = [1, 5, 10, 15, 20, 30, 40, 60]
    max_age_list = [5, 10, 15, 20, 30, 40, 60, 101]

    for min_age, max_age in zip(min_age_list, max_age_list):
        condition_list.append(((df['Age'] >= min_age) & (df['Age'] < max_age)))
        choice_list.append(f'{int(min_age)}_{int(max_age) - 1}')
    
    df['Age'] = np.select(condlist=condition_list, choicelist=choice_list, default=default_value)

    return df


def bin_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to bin the `City` feature of the user into implicity city categories.

    :Paramters:
    df: pandas dataframe for which the binning of the city feature needs to take place

    :Returns:
    dataframe with transformed 'City' feature
    """

    choice_list, condition_list = list(), list()
    default_value = 'City_0'

    unique_cities = df['City'].unique().tolist()

    for index, city in enumerate(unique_cities, 1):
        condition_list.append((df['City'] == city))
        choice_list.append(f'City_{index}')
    
    df['City'] = np.select(condlist=condition_list, choicelist=choice_list, default=default_value)

    return df

def bin_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to bin the `State` feature of the user into implicity state categories.

    :Paramters:
    df: pandas dataframe for which the binning of the state feature needs to take place

    :Returns:
    dataframe with transformed 'State' feature
    """

    choice_list, condition_list = list(), list()
    default_value = 'state_0'

    unique_cities = df['State'].unique().tolist()

    for index, city in enumerate(unique_cities, 1):
        condition_list.append((df['State'] == city))
        choice_list.append(f'State_{index}')
    
    df['State'] = np.select(condlist=condition_list, choicelist=choice_list, default=default_value)

    return df


def bin_author(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to bin the `Author` feature of the book into implicity author categories.

    :Paramters:
    df: pandas dataframe for which the binning of the author feature needs to take place

    :Returns:
    dataframe with transformed 'Author' feature
    """

    choice_list, condition_list = list(), list()
    default_value = 'Author_0'

    unique_cities = df['Book_Author'].unique().tolist()

    for index, city in enumerate(unique_cities, 1):
        condition_list.append((df['Book_Author'] == city))
        choice_list.append(f'Book_Author_{index}')
    
    df['Book_Author'] = np.select(condlist=condition_list, choicelist=choice_list, default=default_value)

    return df


def bin_publisher(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to bin the `Publisher` feature of the book into implicity publisher categories.

    :Paramters:
    df: pandas dataframe for which the binning of the publisher feature needs to take place

    :Returns:
    dataframe with transformed 'Publisher' feature
    """

    choice_list, condition_list = list(), list()
    default_value = 'Publisher_0'

    unique_cities = df['Publisher'].unique().tolist()

    for index, city in enumerate(unique_cities, 1):
        condition_list.append((df['Publisher'] == city))
        choice_list.append(f'Publisher_{index}')
    
    df['Publisher'] = np.select(condlist=condition_list, choicelist=choice_list, default=default_value)

    return df

def bin_yop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to bin the `YOP` feature of the book into implicity year of production categories.

    :Paramters:
    df: pandas dataframe for which the binning of the yop feature needs to take place

    :Returns:
    dataframe with transformed 'YOP' feature
    """
    choice_list, condition_list = list(), list()
    default_value = 'yop_0'
    df['YOP'] = np.where(df['YOP'].isnull(), 0, df['YOP'])
    df['YOP'] = df['YOP'].astype(int)

    yop_list = list(range(1000, 1601, 100))+ list(range(1601, 2100, 10))
    min_yop_list = yop_list[:-1]
    max_yop_list = yop_list[1:]

    for min_yop, max_yop in zip(min_yop_list, max_yop_list):
        condition_list.append(((df['YOP'] >= min_yop) & (df['YOP'] < max_yop)))
        choice_list.append(f'{int(min_yop)}_{int(max_yop) - 1}')
    
    df['YOP'] = np.select(condlist=condition_list, choicelist=choice_list, default=default_value)

    return df


class BookDataset(Dataset):

    """Class to create dataset from an excel file"""

    def __init__(self, dataset: pd.DataFrame) -> None:
        """
        Method to instantiate object of :class: BookDataset

        :Parameters:
        dataset: pandas dataframe to be converted into a torch dataset

        :Returns:
        None
        """
        super().__init__()

        self.dataset = dataset

    def __len__(self) -> int:
        """
        Method to return the length of the dataset.

        :Parameters:
        None
        
        :Returns:
        Length of the dataset
        """

        return self.dataset.shape[0]

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        """
        Method to return the records at ith index of the dataframe

        :Parameters:
        index: Index position of the record to be returned

        :Returns:
        Tuple containing user's information string, book's information string and book's rating
        """

        row = self.dataset.iloc[index, :]

        # user string parameters
        age, city, state, country = row['Age'], row['City'], row['State'], row['Country']
        # book string parameters
        author, publisher, yop = row['Book_Author'], row['Publisher'], row['YOP']
        # Rating
        rating = row['Book_Rating']
        # User and book string
        user_str = f'[USER_CLS] [AGE_S] {age} [AGE_E] [CITY_S] {city} [CITY_E] [STATE_S] {state} [STATE_E] [COUNTRY_S] {country} [COUNTRY_E]'
        book_str = f'[BOOK_CLS] [AUTHOR_S] {author} [AUTHOR_E] [PUBLISHER_S] {publisher} [PUBLISHER_E] [YOP_S] {yop} [YOP_E]'

        return user_str, book_str, rating


class NeuralRecommender(torch.nn.Module):
    """
    Class to implement Recommendation algorithm using deep learning
    """

    def __init__(self,
                 embedding_dimension,
                 nheads,
                 dim_feedforward,
                 n_layers,
                 dropout,
                 user_vocab_size,
                 book_vocab_size,
                 user_pad_token_id,
                 book_pad_token_id,
                 device,
                 out_classes,
                 base):
        
        """
        Method to instantiate object of :class: NeuralRecommender

        :Parameters:
        embedding_dimension: Embedding dimension size to be used for both book and user embedding layer.
        nheads: Number of heads to be used during the mutli-self attention heads in the transformer encoder layers.
        dim_feedforward: Feedforward layer dimension to be used in the transformer encoder layers.
        n_layers: Number of encoder layers to be used.
        dropout: Dropout probability to be used in the feedforward layers present in transformer encoder layers.
        user_vocab_size: Number of embeddings to be present in the user embedding layer.
        book_vocab_size: Number of embeddings to be present in the book embedding layer.
        user_pad_token_id: Padding token id to indicate padding index in user embedding layer.
        book_pad_token_id: Padding token id to indicate padding index in book embedding layer.
        device: Device onto which layers are to be shifted to.
        out_classes: Number of classes to be predicted by the model
        base: The 'Base' used for creating fixed positional embeddings 
        """
        super().__init__()
        self.DEVICE = device
        self.base = base
        self.user_embedding_layer = torch.nn.Embedding(num_embeddings=user_vocab_size, embedding_dim=embedding_dimension, padding_idx=user_pad_token_id)
        self.book_embedding_layer = torch.nn.Embedding(num_embeddings=book_vocab_size, embedding_dim=embedding_dimension, padding_idx=book_pad_token_id)

        self.user_encoder = torch.nn.TransformerEncoder(encoder_layer=torch.nn.TransformerEncoderLayer(d_model=embedding_dimension,
                                                                                                       nhead=nheads,
                                                                                                       dim_feedforward=dim_feedforward,
                                                                                                       batch_first=True,
                                                                                                       dropout=dropout),
                                                                                                       num_layers=n_layers)
        
        self.book_encoder = torch.nn.TransformerEncoder(encoder_layer=torch.nn.TransformerEncoderLayer(d_model=embedding_dimension,
                                                                                                       nhead=nheads,
                                                                                                       dim_feedforward=dim_feedforward,
                                                                                                       batch_first=True,
                                                                                                       dropout=dropout),
                                                                                                       num_layers=n_layers)
        
        self.linear_layer = torch.nn.Sequential(torch.nn.Linear(in_features=embedding_dimension, out_features=out_classes))

    def get_positional_embedding(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """
        Method to get positional embedding for input sequence tensor

        :Parameters:
        sequence_tensor: input sequence tensor

        :Returns:
        positional embedding tensor
        """

        dimension = sequence_tensor.dim()
        batch_size, sequence_length, embedding_dimension = 0, 0, 0
        if dimension == 2:
            sequence_length, embedding_dimension = sequence_tensor.shape
            batch_size = 1
        elif dimension == 3:
            batch_size, sequence_length, embedding_dimension = sequence_tensor.shape
        else:
            raise Exception('Please pass in a tensor of 2D or 3D shape')
        
        positional_encoding = torch.zeros(size=(batch_size, sequence_length, embedding_dimension), dtype=torch.int32)
        position = torch.arange(start=0, end=sequence_length)
        position = position.unsqueeze(1)
        i_2 = torch.arange(start=0, end=embedding_dimension, step=2)
        divisor = torch.pow(self.base, i_2 / embedding_dimension)

        positional_encoding[:, :, 0::2] = torch.sin(position / divisor)
        positional_encoding[:, :, 1::2] = torch.cos(position / divisor)
        
        return positional_encoding.detach().to(self.DEVICE)

    def forward(self, user_string: torch.Tensor, book_string: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate user and book information tensor through the model.

        :Parameters:
        user_string: tensor containing user string tokens
        book_string: tensor containing book string tokens

        :Returns:
        tensor containing logits for the classes
        """
        # Process for user encoder
        # Generate encoder token embedding and encoder positional embedding
        user_token_embedding = self.user_embedding_layer.forward(user_string)
        user_positional_embedding = self.get_positional_embedding(sequence_tensor=user_token_embedding)
        # Add the encoder token embedding and positional embedding
        user_embedding = user_token_embedding + user_positional_embedding

        
        # Process for book encoder
        # Generate encoder token embedding and encoder positional embedding
        book_token_embedding = self.book_embedding_layer.forward(book_string)
        book_positional_embedding = self.get_positional_embedding(sequence_tensor=book_token_embedding)
        # Add the encoder token embedding and positional embedding
        book_embedding = book_token_embedding + book_positional_embedding

        # Get output from the encoder
        user_encoder_output = self.user_encoder.forward(src=user_embedding)
        book_encoder_output = self.book_encoder.forward(src=book_embedding)


        user_cls = user_encoder_output[:, 1, :]
        book_cls = book_encoder_output[:, 1, :]

        combined_token = user_cls * book_cls

        output = self.linear_layer.forward(combined_token)

        return output


if __name__ == '__main__':
    # Set script states and parameters to be used later
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    # Device to be used while training the model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = True if str(DEVICE) == "cuda" else False

    # Model training parameters
    USER_VOCAB_SIZE = 3000
    BOOK_VOCAB_SIZE = 26000
    BATCH_SIZE = 512
    VALIDATION_BATCH_SIZE=2024
    LEARNING_RATE = 1e-3
    EPOCHS = 300
    LOSS_THRESHOLD = 0.2

    # Set directory and file paths to be used
    source_dir = os.getcwd()
    users_path = os.path.join(source_dir , 'Users.csv')
    ratings_path = os.path.join(source_dir , 'Ratings.csv')
    books_path = os.path.join(source_dir , 'Books.csv')
    df_path = os.path.join(source_dir, 'Book_Recommender_Final_DF.xlsx')

    #### Data Preprocessing Phase ####
    # Preprocessing Users dataset
    user_df = pd.read_csv(users_path)
    user_df.rename(columns={'User-ID':'User_ID'}, inplace=True)
    # Change the the column type of User ID
    user_df['User_ID'] = user_df['User_ID'].astype(str)
    # Split the Location Column into 4 and drop columns which are not required
    user_df[['City', 'State', 'Country', 'Country_2']] = user_df['Location'].str.split(', ', n=3, expand=True)
    user_df = user_df[user_df['Country_2'].isnull()]
    user_df = user_df[[column for column in user_df.columns if column not in ['Location', 'Country_2']]]
    # lower case city, state and country column
    for column in ['City', 'State', 'Country']:
        user_df[column] = user_df[column].str.lower()

    # Data Filters to increase the size of the data
    country_filter = ['canada']
    # Keep only the required country data and valid data
    user_df = user_df[(user_df['Country'].isin(country_filter)) & ((user_df['Age'] < 101) | (user_df['Age'].notnull()))]
    user_df = bin_age(user_df)
    user_df.reset_index(inplace=True, drop=True)

    # Preprocessing Rating dataset
    ratings_df = pd.read_csv(ratings_path)
    ratings_df.rename(columns={'User-ID':'User_ID', 'Book-Rating': 'Book_Rating'}, inplace=True)
    ratings_df['User_ID'] = ratings_df['User_ID'].astype(str)
    ratings_df['ISBN'] = ratings_df['ISBN'].astype(str)
    ratings_df['Book_Rating'] = ratings_df['Book_Rating'].astype(int)

    # Preprocessing Books dataset
    books_df = pd.read_csv(books_path)
    books_df.drop(columns=['Book-Title', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'], inplace=True)
    books_df.rename(columns={'Book-Author':'Book_Author', 'Year-Of-Publication': 'YOP'}, inplace=True)
    books_df = books_df[~books_df['YOP'].isin(['DK Publishing Inc', 'Gallimard'])]
    books_df['YOP'] = books_df['YOP'].astype(int)
    books_df = books_df[books_df['YOP'] > 0]
    books_df['Book_Author'] = books_df['Book_Author'].str.lower()
    books_df['Publisher'] = books_df['Publisher'].str.lower()
    books_df['YOP'] = books_df['YOP'].astype(str)

    # Merging User, Rating and Books dataset
    new_df = pd.merge(user_df, ratings_df, on=['User_ID'], how='left')
    new_df = new_df[new_df['Book_Rating'].notnull()]

    new_df = pd.merge(new_df, books_df, on=['ISBN'], how='left')

    #### Data Preprocessing Phase ####
    # Feature Engineer exisiting features into implicit textual categorical features
    new_df = bin_city(new_df)
    new_df = bin_author(new_df)
    new_df = bin_state(new_df)
    new_df = bin_publisher(new_df)
    new_df = bin_yop(new_df)

    # Save the final file
    new_df.to_excel('Book_Recommender_Final_DF.xlsx', index=False)

    #### Model Training Phase ####

    # The saved file is used for training the model
    df = pd.read_excel(df_path)
    train_df = df.copy(deep=True)

    # Each implicit feature category needs to be stored as a tokens in the tokenizer's vocabulary without being split
    # They are therefore extracted from the dataset 
    sp_user_token_list = list()
    sp_book_token_list = list()
    for column in ['Age', 'City', 'State', 'Country']:
        unique_words = train_df[column].unique().tolist()
        sp_user_token_list.extend(unique_words)

    for column in ['Book_Author', 'Publisher', 'YOP']:
        unique_words = train_df[column].unique().tolist()
        sp_book_token_list.extend(unique_words)


    # The training dataset is converted into a pytorch dataset and a dataloader for the training dataset is created
    train_df.reset_index(inplace=True, drop=True)

    train_dataset = BookDataset(dataset=train_df.copy(deep=True))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)

    # Store the sentences present in the dataset to train the tokenizer
    user_sentences, book_sentences = list(), list()
    # Sentences to be used to train the tokenizers
    for user_sentence, book_sentence, label in train_dataset:
        user_sentences.append(user_sentence)
        book_sentences.append(book_sentence)

    # Final list of tokens to be used for training user and book tokenizer 
    # Special User and Book Tokens
    special_user_tokens = ['[USER_PAD]', '[USER_CLS]', '[AGE_S]', '[AGE_E]', '[CITY_S]', '[CITY_E]', '[STATE_S]', '[STATE_E]', '[COUNTRY_S]', '[COUNTRY_E]']
    special_user_tokens = special_user_tokens + sp_user_token_list
    special_book_tokens = ['[BOOK_PAD]', '[BOOK_CLS]', '[AUTHOR_S]', '[AUTHOR_E]', '[PUBLISHER_S]', '[PUBLISHER_E]', '[YOP_S]', '[YOP_E]']
    special_book_tokens = special_book_tokens + sp_book_token_list

    # Train user and book tokenizer
    # Build custom tokenizers for user and book
    user_tokenizer = Tokenizer(model=models.BPE())
    book_tokenizer = Tokenizer(model=models.BPE())

    user_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.ByteLevel(add_prefix_space=True)]) # type: ignore
    user_tokenizer.model = models.BPE(unk_token='[UNK_USER]') # type: ignore

    user_trainer = trainers.BpeTrainer(special_tokens=special_user_tokens, show_progress=True, vocab_size=USER_VOCAB_SIZE, min_frequency=1) # type: ignore
    # Train user tokenizer
    user_tokenizer.train_from_iterator(user_sentences, trainer=user_trainer) # type: ignore

    book_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.ByteLevel(add_prefix_space=True)]) # type: ignore
    book_tokenizer.model = models.BPE(unk_token='[UNK_BOOK]') # type: ignore

    book_trainer = trainers.BpeTrainer(special_tokens=special_book_tokens, show_progress=True, vocab_size=BOOK_VOCAB_SIZE, min_frequency=1) # type: ignore
    # Train book tokenizer
    book_tokenizer.train_from_iterator(book_sentences, trainer=book_trainer) # type: ignore

    # Save the tokenizers
    user_tokenizer.save(os.path.join(source_dir, f'user_tokenizer_{USER_VOCAB_SIZE}.json'))
    book_tokenizer.save(os.path.join(source_dir, f'book_tokenizer_{BOOK_VOCAB_SIZE}.json'))

    # Padding Token IDs
    USER_PAD_ID = user_tokenizer.token_to_id('[USER_PAD]')
    BOOK_PAD_ID = book_tokenizer.token_to_id('[BOOK_PAD]')

    # Instantiate model to be trained
    model = NeuralRecommender(embedding_dimension=64,
                              nheads=4,
                              dim_feedforward=512,
                              n_layers=2,
                              dropout=0.2,
                              user_vocab_size=USER_VOCAB_SIZE,
                              book_vocab_size=BOOK_VOCAB_SIZE,
                              user_pad_token_id=USER_PAD_ID,
                              book_pad_token_id=BOOK_PAD_ID,
                              device=DEVICE,
                              out_classes=11,
                              base=10000)

    # Shift the model to the device
    model = model.to(DEVICE)
    # Instantiate optimizer and crossentropyloss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()

    # Train the model
    model.train()
    loss_log_list = list()
    start_time = time.perf_counter()
    with tqdm.tqdm(total=EPOCHS, desc=f'Training...') as main_bar:
        for epoch in range(EPOCHS):
            train_loss_list = []
            for user_string_batch, book_string_batch, label_batch in train_dataloader:
                    user_tokens = [user_tokenizer.encode(u_sentence).ids for u_sentence in user_string_batch]
                    book_tokens = [book_tokenizer.encode(b_sentence).ids for b_sentence in book_string_batch]

                    user_tensor = torch.tensor(data=user_tokens, dtype=torch.long, device=DEVICE)
                    book_tensor = torch.tensor(data=book_tokens, dtype=torch.long, device=DEVICE)
                    label_tensor = torch.tensor(data=label_batch.clone().detach(), dtype=torch.long, device=DEVICE)

                    predicted_labels = model.forward(user_tensor, book_tensor)
                    loss = loss_function.forward(predicted_labels, label_tensor)
                    loss_value = loss.item()

                    train_loss_list.append(loss_value)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            main_bar.update()

            # Keep a track of training loss 
            average_loss = sum(train_loss_list) / len(train_loss_list)
            loss_string = f'EPOCH {epoch + 1} Train Loss: {average_loss}'
            if epoch % 20 == 0:
                print(loss_string)
                # Log the training results per epoch
                loss_log_list.append(loss_string)

            if average_loss < LOSS_THRESHOLD:
                print(loss_string)
                # Log the training results per epoch
                loss_log_list.append(loss_string)
                break
    end_time = time.perf_counter()

    # Move the model to cpu before storing the model
    DEVICE = torch.device('cpu')
    model = model.to(DEVICE)
    torch.save(model, os.path.join(source_dir, 'Book_Recommendation_Model.pt'))

    log_string  = '\n'.join(loss_log_list + [f'Total training time: {end_time - start_time} seconds'])
    # Save the training log file
    with open(os.path.join(source_dir, 'Book_Recommender_Log_File.txt'), mode='w', encoding='utf-8') as log_file:
        log_file.write(log_string)

    #### Model validation phase ####
    # Create validation dataset
    # In our case a sample of the training dataset is itself used 
    validation_dataset = BookDataset(dataset=train_df[:VALIDATION_BATCH_SIZE].copy(deep=True))
    validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=True, pin_memory=pin_memory)
    # Check for model inferencing over CPU
    DEVICE = torch.device('cpu')
    model = model.to(DEVICE)
    model.DEVICE = DEVICE
    pin_memory = True if str(DEVICE) == "cuda" else False
    model.eval()
    with torch.inference_mode():
        start_time = time.perf_counter()
        for user_string_batch, book_string_batch, label_batch in validation_dataloader:
                    user_tokens = [user_tokenizer.encode(u_sentence).ids for u_sentence in user_string_batch]
                    book_tokens = [book_tokenizer.encode(b_sentence).ids for b_sentence in book_string_batch]

                    user_tensor = torch.tensor(data=user_tokens, dtype=torch.long, device=DEVICE)
                    book_tensor = torch.tensor(data=book_tokens, dtype=torch.long, device=DEVICE)
                    label_tensor = torch.tensor(data=label_batch.clone().detach(), dtype=torch.long, device=DEVICE)

                    predicted_labels = model.forward(user_tensor, book_tensor)
                    # Get the top-1 prediction for the user-book pair
                    predicted_labels = torch.argmax(predicted_labels, dim=-1)

                    accuracy = torch.sum(label_tensor == predicted_labels).item() / len(label_tensor)
        end_time = time.perf_counter()
        inference_time = end_time - start_time    
        print(f'Validation accuracy of {accuracy*100:.3f} is achieved after {epoch} training epochs. The run time of inference is {inference_time} seconds')