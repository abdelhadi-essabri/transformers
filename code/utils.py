import copy
import torch.nn as nn

def replicate(block, N=6) -> nn.ModuleList:
    """
    Réplique un bloc donné plusieurs fois pour créer une pile de blocs identiques, 
    souvent utilisée dans les architectures Transformer pour empiler plusieurs couches d'encodeur ou de décodeur.
    
    Args:
    - block (nn.Module): Un bloc neural (par exemple, un encodeur ou un décodeur) hérité de nn.Module, 
                         qui sera copié N fois pour créer une pile de couches.
    - N (int, optionnel): Le nombre de copies du bloc à créer. Par défaut, N est 6, comme spécifié dans le papier 
                          original des Transformers (Vaswani et al., 2017).
    
    Returns:
    - nn.ModuleList: Une liste spéciale PyTorch (nn.ModuleList) qui contient N copies du bloc donné. Cette liste permet
                     à PyTorch de gérer correctement les paramètres et les gradients pour chaque bloc dans l'entraînement.
    """
    
    # Créer une liste de N blocs, chacun étant une copie indépendante du bloc d'origine.
    # copy.deepcopy() est utilisé pour s'assurer que chaque copie est complètement séparée, 
    # c'est-à-dire qu'aucun bloc ne partage ses poids avec un autre.
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    
    return block_stack
