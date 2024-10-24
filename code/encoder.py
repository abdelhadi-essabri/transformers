import torch.nn as nn
from utils import replicate
from attention import MultiHeadAttention
from embedding import Embedding, PositionalEncoding


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim=768, heads=8, expansion_factor=4, dropout=0.2):
        """
        Bloc de base du transformateur, utilisé dans l'encodeur et le décodeur.
        
        :param embed_dim: la dimension de l'embedding (par défaut 768).
        :param heads: le nombre de têtes dans l'attention multi-têtes (par défaut 8).
        :param expansion_factor: facteur d'expansion pour la couche feed-forward (par défaut 4).
        :param dropout: la probabilité de dropout (entre 0 et 1, par défaut 0.2).
        """
        super(TransformerBlock, self).__init__()

        # Attention multi-têtes
        self.attention = MultiHeadAttention(embed_dim, heads)
        # Normalisation des embeddings
        self.norm = nn.LayerNorm(embed_dim)

        # Réseau feed-forward avec une fonction d'activation ReLU
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),  # ex: 768x(4*768) -> (768, 3072)
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.Linear(embed_dim * expansion_factor, embed_dim),  # ex: 3072x768 -> (3072, 768)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, value, mask=None):
        #################### Attention multi-têtes ####################
        # Passage des clés, requêtes et valeurs dans l'attention multi-têtes
        attention_out = self.attention(key, query, value, mask)  # ex: 32x10x768

        # Ajout de la connexion résiduelle
        attention_out = attention_out + value  # ex: 32x10x768

        # Normalisation et dropout
        attention_norm = self.dropout(self.norm(attention_out))  # ex: 32x10x768

        #################### Réseau Feed-Forward ####################
        fc_out = self.feed_forward(attention_norm)  # ex: 32x10x768 -> 32x10x3072 -> 32x10x768

        # Ajout de la connexion résiduelle
        fc_out = fc_out + attention_norm  # ex: 32x10x768

        # Normalisation et dropout
        fc_norm = self.dropout(self.norm(fc_out))  # ex: 32x10x768

        return fc_norm


class Encoder(nn.Module):

    def __init__(self, seq_len, vocab_size, embed_dim=768, num_blocks=6, expansion_factor=4, heads=8, dropout=0.2):
        """
        L'encodeur de l'architecture Transformer, une pile d'encoders.
        Selon le papier "Attention Is All You Need", il y a généralement 6 blocs d'encoders empilés.
        
        :param seq_len: la longueur des séquences (le nombre de mots dans chaque séquence).
        :param vocab_size: la taille totale du vocabulaire (nombre total de mots uniques).
        :param embed_dim: la dimension de l'embedding (par défaut 768).
        :param num_blocks: le nombre de blocs (ou d'encodeurs), par défaut 6.
        :param expansion_factor: le facteur qui détermine la taille de sortie de la couche feed-forward.
        :param heads: le nombre de têtes dans l'attention multi-têtes de chaque bloc.
        :param dropout: la probabilité de dropout (entre 0 et 1).
        """
        super(Encoder, self).__init__()

        # Embedding des mots (taille du vocabulaire x dimension d'embedding)
        self.embedding = Embedding(vocab_size, embed_dim)

        # Encodage positionnel (embedding dimension x longueur de séquence)
        self.positional_encoder = PositionalEncoding(embed_dim, seq_len)

        # Réplication du bloc de transformateur pour créer une pile d'encoders
        self.blocks = replicate(TransformerBlock(embed_dim, heads, expansion_factor, dropout), num_blocks)

    def forward(self, x):
        # Ajout de l'encodage positionnel aux embeddings des mots
        out = self.positional_encoder(self.embedding(x))

        # Passage à travers chaque bloc de l'encodeur
        for block in self.blocks:
            out = block(out, out, out)

        # La sortie finale est de la forme: batch_size x seq_len x embed_size, ex: 32x10x768
        return out
