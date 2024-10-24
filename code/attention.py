from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim=768, heads=8):
        """
        Classe Multi-Head Attention.
        :param embed_dim: la dimension de l'embedding (ici 768).
        :param heads: le nombre de têtes d'attention, par défaut égal à 8.
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  # 768 après modification
        self.heads = heads  # 8 têtes d'attention
        # Calcul de la dimension de chaque tête. Ici, 768 / 8 = 96.
        self.head = int(self.embed_dim / self.heads)
        # Note: embed_dim doit être divisible par le nombre de têtes.

        # Création des couches linéaires pour les matrices Query, Key et Value.
        # Chacune de ces matrices aura une dimension de (batch_size, seq_len, heads, head_dim)
        self.query = nn.Linear(self.head, self.head, bias=False)
        self.value = nn.Linear(self.head, self.head, bias=False)
        self.key = nn.Linear(self.head, self.head, bias=False)

        # Couche fully connected pour combiner les résultats des différentes têtes.
        self.fc_out = nn.Linear(self.head * self.heads, embed_dim)

    def forward(self, key, query, value, mask=None):
        """
        Applique l'attention multi-tête sur les inputs key, query et value.
        :param key: la matrice key (batch_size, seq_len, embed_dim).
        :param query: la matrice query (batch_size, seq_len, embed_dim).
        :param value: la matrice value (batch_size, seq_len, embed_dim).
        :param mask: un masque optionnel pour les positions à ignorer.
        :return: la sortie après l'attention multi-tête.
        """
        # Récupération des dimensions du batch et des séquences.
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)

        # Reshape des inputs de (batch_size, seq_len, embed_dim) à (batch_size, seq_len, heads, head_dim)
        key = key.reshape(batch_size, k_len, self.heads, self.head)
        query = query.reshape(batch_size, q_len, self.heads, self.head)
        value = value.reshape(batch_size, v_len, self.heads, self.head)

        # Application des transformations linéaires pour key, query et value.
        key = self.key(key)  # Shape: (batch_size, seq_len, heads, head_dim)
        query = self.query(query)  # Idem pour query
        value = self.value(value)  # Idem pour value

        ############### Calcul de l'attention (query x key) ###############

        # Produit matriciel entre query et key. On utilise einsum pour manipuler facilement les dimensions.
        # Résultat: shape (batch_size, heads, q_len, k_len)
        product = torch.einsum("bqhd,bkhd->bhqk", [query, key])

        # Si un masque est fourni (par exemple dans le décodeur), on applique le masque.
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        # Normalisation par la racine carrée de la dimension de chaque tête (pour éviter des valeurs trop grandes).
        product = product / sqrt(self.head)

        # Application de la fonction softmax pour obtenir les scores d'attention.
        scores = F.softmax(product, dim=-1)

        ############### Application des scores à value (scores x value) ###############

        # On multiplie les scores d'attention par les valeurs (value).
        # Einsum permet de gérer les dimensions de manière fluide.
        # Sortie : shape (batch_size, heads, q_len, head_dim)
        output = torch.einsum("nhql,nlhd->nqhd", [scores, value])

        # On reshape la sortie pour combiner les têtes: (batch_size, q_len, heads * head_dim) -> (batch_size, q_len, embed_dim)
        output = output.reshape(batch_size, q_len, self.heads * self.head)

        # Passage à travers la couche fully connected pour générer la sortie finale.
        output = self.fc_out(output)  # (batch_size, q_len, embed_dim)

        return output
