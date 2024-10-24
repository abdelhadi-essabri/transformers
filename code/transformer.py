import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, src_vocab_size, target_vocab_size, seq_len, embed_dim=768, num_blocks=6, expansion_factor=4, heads=8, dropout=0.2):
        """
        Le modèle Transformer qui comprend à la fois un encodeur et un décodeur pour traiter des séquences
        dans des tâches telles que la traduction automatique.

        :param embed_dim: la dimension de l'embedding (par défaut 768).
        :param src_vocab_size: la taille du vocabulaire source (entrée).
        :param target_vocab_size: la taille du vocabulaire cible (sortie).
        :param seq_len: la longueur de la séquence (nombre de mots).
        :param num_blocks: le nombre de blocs d'encodeurs et de décodeurs (par défaut 6).
        :param expansion_factor: le facteur d'expansion pour la couche feed-forward (par défaut 4).
        :param heads: le nombre de têtes dans l'attention multi-têtes (par défaut 8).
        :param dropout: la probabilité de dropout pour éviter le surapprentissage (par défaut 0.2).
        """
        super(Transformer, self).__init__()
        self.target_vocab_size = target_vocab_size

        # Encoder de la séquence source
        self.encoder = Encoder(seq_len=seq_len,
                               vocab_size=src_vocab_size,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               heads=heads,
                               dropout=dropout)

        # Décodeur de la séquence cible
        self.decoder = Decoder(target_vocab_size=target_vocab_size,
                               seq_len=seq_len,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               heads=heads,
                               dropout=dropout)

        # Couche Fully Connected pour prédire les mots du vocabulaire cible
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)

    def make_trg_mask(self, trg):
        """
        Crée un masque pour la séquence cible. Le masque est une matrice triangulaire inférieure, où
        seuls les éléments sous ou sur la diagonale sont visibles (utilisé pour la régulation dans le décodeur).

        :param trg: la séquence cible
        :return: le masque triangulaire inférieur
        """
        batch_size, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch_size, 1, trg_len, trg_len)
        return trg_mask

    def forward(self, source, target):
        """
        Passe avant du modèle Transformer. Prend la séquence source et la séquence cible et les passe
        respectivement à travers l'encodeur et le décodeur. La sortie est une distribution de probabilité sur le vocabulaire cible.

        :param source: la séquence source (entrée)
        :param target: la séquence cible (entrée partielle)
        :return: les probabilités de la séquence prédite
        """
        # Création du masque pour la séquence cible
        trg_mask = self.make_trg_mask(target)

        # Passage de la séquence source à travers l'encodeur
        enc_out = self.encoder(source)

        # Passage de la séquence cible et de la sortie de l'encodeur dans le décodeur
        outputs = self.decoder(target, enc_out, trg_mask)

        # Passage des sorties à travers la couche Fully Connected et application de softmax pour obtenir les probabilités
        output = F.softmax(self.fc_out(outputs), dim=-1)
        return output
