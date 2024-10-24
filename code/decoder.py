import torch.nn as nn
import torch.nn.functional as F
from utils import replicate
from attention import MultiHeadAttention
from embedding import PositionalEncoding
from encoder import TransformerBlock


class DecoderBlock(nn.Module):

    def __init__(self, embed_dim=768, heads=8, expansion_factor=4, dropout=0.2):
        """
        Bloc du décodeur dans l'architecture Transformer. Ce bloc comprend un mécanisme d'attention multi-têtes
        suivi par un TransformerBlock, similaire à l'encodeur.
        
        :param embed_dim: la dimension de l'embedding (par défaut 768).
        :param heads: le nombre de têtes dans l'attention multi-têtes (par défaut 8).
        :param expansion_factor: facteur d'expansion pour la couche feed-forward (par défaut 4).
        :param dropout: probabilité de dropout pour éviter le surapprentissage (par défaut 0.2).
        """
        super(DecoderBlock, self).__init__()

        # Attention multi-têtes du décodeur
        self.attention = MultiHeadAttention(embed_dim, heads)
        # Normalisation
        self.norm = nn.LayerNorm(embed_dim)
        # Dropout pour éviter le surapprentissage
        self.dropout = nn.Dropout(dropout)
        # Le bloc Transformer (similaire à celui de l'encodeur)
        self.transformerBlock = TransformerBlock(embed_dim, heads, expansion_factor, dropout)

    def forward(self, key, query, x, mask):
        # Passe les entrées dans l'attention multi-têtes du décodeur
        decoder_attention = self.attention(x, x, x, mask)

        # Ajout de la connexion résiduelle + normalisation
        value = self.dropout(self.norm(decoder_attention + x))

        # Passage dans le bloc Transformer du décodeur (avec attention multi-têtes et feed-forward)
        decoder_attention_output = self.transformerBlock(key, query, value)

        return decoder_attention_output


class Decoder(nn.Module):

    def __init__(self, target_vocab_size, seq_len, embed_dim=768, num_blocks=6, expansion_factor=4, heads=8, dropout=0.2):
        """
        La partie décodeur de l'architecture Transformer. Il est constitué de plusieurs blocs de décodeurs empilés.
        Le papier "Attention Is All You Need" suggère 6 blocs de décodeurs.

        :param target_vocab_size: la taille du vocabulaire cible.
        :param seq_len: la longueur des séquences (nombre de mots dans chaque séquence).
        :param embed_dim: la dimension de l'embedding (par défaut 768).
        :param num_blocks: le nombre de blocs (par défaut 6).
        :param expansion_factor: le facteur d'expansion pour les couches feed-forward (par défaut 4).
        :param heads: le nombre de têtes dans l'attention multi-têtes (par défaut 8).
        :param dropout: la probabilité de dropout pour éviter le surapprentissage (par défaut 0.2).
        """
        super(Decoder, self).__init__()

        # Embedding des mots cibles (taille du vocabulaire x dimension d'embedding)
        self.embedding = nn.Embedding(target_vocab_size, embed_dim)

        # Encodage positionnel pour ajouter des informations sur l'ordre des mots
        self.positional_encoder = PositionalEncoding(embed_dim, seq_len)

        # Réplication des blocs du décodeur (6 par défaut)
        self.blocks = replicate(DecoderBlock(embed_dim, heads, expansion_factor, dropout), num_blocks)

        # Dropout pour la régularisation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask):
        # Ajout de l'encodage positionnel aux embeddings des mots cibles
        x = self.dropout(self.positional_encoder(self.embedding(x)))  # ex: 32x10x768

        # Passage à travers chaque bloc du décodeur
        for block in self.blocks:
            x = block(encoder_output, x, encoder_output, mask)

        # Sortie finale après tous les blocs du décodeur
        return x
