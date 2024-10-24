from math import sin, cos, sqrt, log
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=768):
        """
        Classe Embedding pour convertir des mots en vecteurs d'embedding.
        :param vocab_size: la taille du vocabulaire (nombre total de mots distincts).
        :param embed_dim: la dimension de l'espace d'embedding (par défaut 768).

        Exemple: Si la taille du vocabulaire est de 1000 et que la dimension d'embedding est 768,
        la matrice d'embedding aura une taille de 1000x768.
        Pour un batch de 64 séquences de 15 mots chacune, la sortie aura une forme de 64x15x768.
        """
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim  # Définition de la dimension d'embedding
        # nn.Embedding crée une couche qui mappe des indices de mots en vecteurs d'embedding.
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Passage en avant (forward pass).
        :param x: séquence de mots (exprimée en indices de vocabulaire).
        :return: représentation numérique des mots (embedding).
        """
        # L'embedding est multiplié par la racine carrée de la dimension pour une normalisation.
        output = self.embed(x) * sqrt(self.embed_dim)
        # print(f"Embedding shape: {output.shape}")
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=768, max_seq_len=5000, dropout=0.1):
        """
        Classe PositionalEncoding pour ajouter un encodage positionnel aux embeddings.
        :param embed_dim: la dimension de l'embedding (doit correspondre à celle des embeddings).
        :param max_seq_len: la longueur maximale de la séquence (par défaut 5000).
        :param dropout: la probabilité de dropout pour régulariser les embeddings.

        Le principe de l'encodage positionnel est d'ajouter des informations sur la position
        des mots dans la séquence aux embeddings des mots avant de les envoyer à un modèle
        de type transformateur. Le calcul est basé sur des fonctions sinusoïdales (sin et cos).

        Pour plus de détails, consulter la section "Positional Encoding" dans l'article
        "Attention Is All You Need".
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        # Initialisation de la matrice d'encodage positionnel à zéro.
        positional_encoding = torch.zeros(max_seq_len, self.embed_dim)
        # Calcul des positions (0 à max_seq_len - 1) et redimensionnement.
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        # Calcul du terme de division basé sur l'exponentielle.
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(log(10000.0) / embed_dim)
        )
        # Application de sin et cos sur les positions. Indices pairs pour sin, impairs pour cos.
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Ajout d'une dimension supplémentaire pour correspondre à la forme (batch_size, seq_len, embed_dim).
        pe = positional_encoding.unsqueeze(0)

        # Enregistrement de la matrice "pe" dans le module sans qu'elle soit un paramètre modifiable.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass pour ajouter l'encodage positionnel aux embeddings.
        :param x: les embeddings des mots dans une séquence (batch_size, seq_len, embed_dim).
        :return: la séquence enrichie par l'encodage positionnel.
        """
        # Ajout de l'encodage positionnel à l'embedding.
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        # Application du dropout pour la régularisation.
        return self.dropout(x)
