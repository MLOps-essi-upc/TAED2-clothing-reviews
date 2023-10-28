"""
This module is created to contain the classes
related to the first deep learning model we tried
and all functions related to it.
"""

import torch


class Dictionary:
    """
    This class constructs a dictionary to map
    the tokens that encode the words to an index,
    and viceversa.
    """

    def __init__(self):
        """
        Initialize the dictionary.
        """
        # A dictionary to map tokens to indices
        self.token2idx = {}
        # A list to map indices to tokens
        self.idx2token = []

    def add_token(self, token):
        """
        Add a token to the dictionary and return its index.

        Args:
            token (str):
            - The token to be added to the dictionary.

        Returns:
            int: The index of the added token.
        """
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
        """
        Get the length of the dictionary.

        Returns:
            int: The number of tokens in the dictionary.
        """
        return len(self.idx2token)


class CharRNNClassifier(torch.nn.Module):
    """
    This class defines the RNN Classifier
    itself.
    """

    def __init__(
            self, input_size,
            embedding_size,
            hidden_size,
            output_size,
            model="lstm",
            num_layers=4,
            bidirectional=False,
            pad_idx=0
    ):
        """
        Initialize the Character RNN Classifier model.

        Args:
            input_size (int): The size of the input vocabulary.
            embedding_size (int): The size of word embeddings.
            hidden_size (int): The size of the hidden layers.
            output_size (int): The size of the output
            (number of classes).
            model (str): The RNN model type ("lstm" or "gru").
            num_layers (int): The number of RNN layers.
            bidirectional (bool): Whether to use bidirectional RNN.
            pad_idx (int): The padding index for embeddings.

        Returns:
            None
        """
        # Initialize the parent class.
        super().__init__()
        self.model = model.lower()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embed = torch.nn.Embedding(
            input_size, embedding_size, padding_idx=pad_idx
        )
        # Create a GRU model if specified.
        if self.model == "gru":
            self.rnn = torch.nn.GRU(
                embedding_size, hidden_size,
                num_layers, bidirectional=bidirectional
            )
        # Create an LSTM model if specified.
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(
                embedding_size, 2 * hidden_size,
                num_layers, bidirectional=bidirectional
            )
        # Create a linear layer for output.
        self.h2o = torch.nn.Linear(
            self.num_directions * hidden_size, output_size
        )
        # Apply dropout for regularization.
        self.dropout = torch.nn.Dropout(0.2, inplace=True)

    def forward(self, user_input, input_lengths):
        """
        Forward pass of the model.

        Args:
            user_input (Tensor): Input sequences.
            input_lengths (Tensor): Lengths of
            input sequences.

        Returns:
            Tensor: Model output.
        """
        # Embed the input sequences.
        encoded = self.embed(user_input)
        # Pack the sequences for efficient computation.
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            encoded, input_lengths
        )
        # Pass packed sequences through RNN.
        output, _ = self.rnn(packed)
        # Unpack the sequences.
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, padding_value=float('-inf')
        )
        # Get the maximum values along the time dimension.
        output, _ = padded.max(dim=0)
        # Apply dropout for regularization.
        output = self.dropout(output)
        # Linear transformation to the output classes.
        output = self.h2o(output.view(
            -1, self.num_directions * self.hidden_size
        ))

        return output


if __name__ == '__main__':
    pass
