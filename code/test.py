# Importer le modèle Transformer depuis le module correspondant
from transformer import Transformer
import torch

# Définir les paramètres du modèle
src_vocab_size = 3200  # Taille du vocabulaire source
target_vocab_size = 3200  # Taille du vocabulaire cible
num_blocks = 6  # Nombre de blocs d'encodeurs/décodeurs
seq_len = 12  # Longueur des séquences

# Imprimer un message pour vérifier que le script démarre correctement
print('okk')

# Définir les séquences source et cible en tant que tenseurs
# Supposons que le token '0' soit le token de début (sos) et '1' soit le token de fin (eos)
src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1],  # Séquence source 1
                    [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])  # Séquence source 2

target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1],  # Séquence cible 1
                       [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])  # Séquence cible 2
# Vérifier les formes des séquences
print(src.shape, target.shape)  # Devrait imprimer (2, 12) pour les deux

# Créer une instance du modèle Transformer avec une dimension d'embedding de 768
model = Transformer(embed_dim=768,  # Changement à 768
                    src_vocab_size=src_vocab_size,
                    target_vocab_size=target_vocab_size,
                    seq_len=seq_len,
                    num_blocks=num_blocks,
                    expansion_factor=4,
                    heads=8)

# Imprimer une représentation du modèle pour vérifier sa construction
print(model)

# Passer les séquences source et cible à travers le modèle
out = model(src, target)

# Afficher la forme de la sortie générée par le modèle
print(f"Output Shape: {out.shape}")  # Devrait être (2, 12, 3200) --->(batch_size,seq_lenght,vocab_size)
