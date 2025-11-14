"""
=========================================================================================
IMAGE CAPTIONING AVEC PYTORCH - VERSION OPTIMISÉE
=========================================================================================

Ce script implémente un modèle d'image captioning optimisé pour un entraînement rapide.

OPTIMISATIONS PRINCIPALES :
1. Pré-extraction des features CNN avec cache (gain ~50x)
2. Batch size augmenté pour mieux utiliser le GPU (gain ~1.5x)
3. DataLoader avec num_workers et pin_memory (gain ~1.3x)
4. Mixed Precision Training (AMP) (gain ~2x)
5. Gradient accumulation pour stabilité
6. OneCycleLR scheduler pour convergence rapide

TEMPS ATTENDU : ~2-5 minutes par epoch (vs 1h+ sans optimisations)
=========================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================================================================================
# 0. CONFIGURATION
# =========================================================================================

# Chemins des données
DATA_DIR = "/content/Flicker8k_Dataset"  # Modifier selon votre environnement
CAPTIONS_FILE = "/content/Flickr8k.token.txt"  # Modifier selon votre environnement
FEATURES_CACHE = "flickr8k_features_resnet50.pth"  # Cache des features CNN

# Hyperparamètres optimisés
BATCH_SIZE = 128  # ✅ Augmenté pour mieux utiliser le GPU
NUM_WORKERS = 2   # ✅ Chargement parallèle des données
NUM_EPOCHS = 5
LEARNING_RATE = 5e-4
MAX_SEQ_LENGTH = 50
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
ACCUMULATION_STEPS = 4  # ✅ Gradient accumulation pour stabilité

# Configuration du device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device utilisé : {device}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   Mémoire disponible : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# =========================================================================================
# 1. CHARGEMENT DES DONNÉES
# =========================================================================================

def load_captions(captions_file):
    """
    Charge les légendes depuis le fichier Flickr8k.token.txt
    Format : image.jpg#0  caption text

    Returns:
        pandas.DataFrame avec colonnes ['image', 'caption']
    """
    print("\n📂 Chargement des légendes...")

    data = []
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Séparation image#num et caption
            parts = line.split('\t')
            if len(parts) != 2:
                continue

            image_info, caption = parts
            # Extraction du nom de l'image (sans #0, #1, etc.)
            image_name = image_info.split('#')[0]

            data.append({
                'image': image_name,
                'caption': caption.strip()
            })

    df = pd.DataFrame(data)
    print(f"✅ {len(df)} légendes chargées pour {df['image'].nunique()} images")
    print(f"   Moyenne : {len(df) / df['image'].nunique():.1f} légendes par image")

    return df

# =========================================================================================
# 2. PRÉ-TRAITEMENT DES TEXTES
# =========================================================================================

class TextPreprocessor:
    """
    Préprocesseur pour convertir les légendes en séquences d'entiers.
    Gère le vocabulaire et les tokens spéciaux.
    """

    def __init__(self, min_word_freq=2):
        """
        Args:
            min_word_freq: Fréquence minimale pour qu'un mot soit dans le vocabulaire
        """
        self.min_word_freq = min_word_freq
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

    def build_vocabulary(self, captions):
        """
        Construit le vocabulaire à partir des légendes.

        Args:
            captions: Liste de légendes (strings)
        """
        print("\n📚 Construction du vocabulaire...")

        # Comptage des mots
        word_counts = Counter()
        for caption in captions:
            words = caption.lower().split()
            word_counts.update(words)

        # Filtrage par fréquence
        valid_words = [word for word, count in word_counts.items()
                      if count >= self.min_word_freq]

        # Création des dictionnaires (tokens spéciaux en premier)
        special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        self.word_to_idx = {token: idx for idx, token in enumerate(special_tokens)}

        for word in sorted(valid_words):
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)

        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        print(f"✅ Vocabulaire construit : {self.vocab_size} mots")
        print(f"   Mots filtrés (freq < {self.min_word_freq}) : {len(word_counts) - len(valid_words)}")

    def caption_to_sequence(self, caption):
        """
        Convertit une légende en séquence d'indices.

        Args:
            caption: Légende (string)

        Returns:
            Liste d'indices (avec <START> et <END>)
        """
        words = caption.lower().split()
        sequence = [self.word_to_idx['<START>']]

        for word in words:
            idx = self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
            sequence.append(idx)

        sequence.append(self.word_to_idx['<END>'])
        return sequence

    def sequence_to_caption(self, sequence):
        """
        Convertit une séquence d'indices en légende.

        Args:
            sequence: Liste ou tensor d'indices

        Returns:
            Légende (string)
        """
        if torch.is_tensor(sequence):
            sequence = sequence.cpu().numpy()

        words = []
        for idx in sequence:
            word = self.idx_to_word.get(int(idx), '<UNK>')
            if word in ['<START>', '<PAD>']:
                continue
            if word == '<END>':
                break
            words.append(word)

        return ' '.join(words)

# =========================================================================================
# 3. EXTRACTION DES FEATURES CNN (PRÉ-EXTRACTION AVEC CACHE) ⚡
# =========================================================================================

class FeatureExtractorCNN:
    """
    Extracteur de features basé sur ResNet50 pré-entraîné.
    Les features sont extraites UNE SEULE FOIS et mises en cache.
    """

    def __init__(self, model_name='resnet50', device='cuda'):
        """
        Args:
            model_name: Nom du modèle ('resnet50' ou 'vgg16')
            device: Device PyTorch
        """
        self.device = device
        self.model_name = model_name

        print(f"\n🔧 Initialisation de l'extracteur de features ({model_name})...")

        # Chargement du modèle pré-entraîné
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Suppression de la couche de classification
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 25088
        else:
            raise ValueError(f"Modèle non supporté : {model_name}")

        self.model.eval()
        self.model.to(device)

        # Transformation des images (normalization ImageNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ Extracteur prêt (dimension des features : {self.feature_dim})")

    def extract_single(self, image_path):
        """
        Extrait les features d'une seule image.

        Args:
            image_path: Chemin vers l'image

        Returns:
            Tensor de features (feature_dim,)
        """
        try:
            # Chargement et transformation
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Extraction
            with torch.no_grad():
                features = self.model(img_tensor).view(-1)

            return features.cpu()

        except Exception as e:
            print(f"❌ Erreur sur {image_path}: {e}")
            return torch.zeros(self.feature_dim)

    def extract_batch(self, image_paths, batch_size=64):
        """
        Extrait les features par batch (plus rapide).

        Args:
            image_paths: Liste de chemins d'images
            batch_size: Taille du batch

        Returns:
            Dict {image_name: features_tensor}
        """
        features_cache = {}

        print(f"🔍 Extraction des features en batch (batch_size={batch_size})...")

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            valid_names = []

            # Chargement du batch
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                    valid_names.append(os.path.basename(path))
                except Exception as e:
                    print(f"❌ Erreur sur {path}: {e}")
                    continue

            if len(batch_tensors) == 0:
                continue

            # Extraction batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor).view(batch_tensor.size(0), -1)

            # Stockage dans le cache
            for name, feat in zip(valid_names, features):
                features_cache[name] = feat.cpu()

        print(f"✅ {len(features_cache)} features extraites")
        return features_cache


def load_or_extract_features(df, data_dir, cache_path, device):
    """
    Charge les features depuis le cache ou les extrait si nécessaire.

    Args:
        df: DataFrame avec la colonne 'image'
        data_dir: Répertoire des images
        cache_path: Chemin du fichier cache
        device: Device PyTorch

    Returns:
        Dict {image_name: features_tensor}
    """
    # Tentative de chargement depuis le cache
    if os.path.exists(cache_path):
        print(f"\n💾 Chargement des features depuis le cache : {cache_path}")
        features_cache = torch.load(cache_path)
        print(f"✅ {len(features_cache)} features chargées depuis le cache")
        return features_cache

    # Extraction des features
    print(f"\n⚠️  Cache introuvable. Extraction des features (cela peut prendre quelques minutes)...")

    extractor = FeatureExtractorCNN(model_name='resnet50', device=device)

    # Liste des images uniques
    unique_images = df['image'].unique()
    image_paths = [os.path.join(data_dir, img) for img in unique_images]

    # Extraction par batch (beaucoup plus rapide)
    features_cache = extractor.extract_batch(image_paths, batch_size=64)

    # Sauvegarde dans le cache
    print(f"\n💾 Sauvegarde des features dans le cache : {cache_path}")
    torch.save(features_cache, cache_path)
    print(f"✅ Cache sauvegardé. Prochains lancements seront instantanés !")

    return features_cache

# =========================================================================================
# 4. DIVISION DU DATASET
# =========================================================================================

def split_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Divise le dataset en train/val/test au niveau des images.

    Args:
        df: DataFrame avec colonnes ['image', 'caption']
        train_ratio: Proportion du train set
        val_ratio: Proportion du validation set
        test_ratio: Proportion du test set
        seed: Seed pour reproductibilité

    Returns:
        train_df, val_df, test_df
    """
    print("\n✂️  Division du dataset...")

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Les ratios doivent sommer à 1.0"

    # Images uniques
    unique_images = df['image'].unique()
    total_images = len(unique_images)

    # Shuffle reproductible
    import numpy as np
    np.random.seed(seed)
    shuffled_images = np.random.permutation(unique_images)

    # Calcul des indices
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Séparation
    train_images = set(shuffled_images[:train_end])
    val_images = set(shuffled_images[train_end:val_end])
    test_images = set(shuffled_images[val_end:])

    # Création des DataFrames
    train_df = df[df['image'].isin(train_images)].copy()
    val_df = df[df['image'].isin(val_images)].copy()
    test_df = df[df['image'].isin(test_images)].copy()

    print(f"✅ Division effectuée :")
    print(f"   Train : {len(train_df)} légendes ({len(train_images)} images)")
    print(f"   Val   : {len(val_df)} légendes ({len(val_images)} images)")
    print(f"   Test  : {len(test_df)} légendes ({len(test_images)} images)")

    return train_df, val_df, test_df

# =========================================================================================
# 5. DATASET PYTORCH OPTIMISÉ
# =========================================================================================

class ImageCaptionDataset(Dataset):
    """
    Dataset PyTorch optimisé avec features pré-extraites.
    """

    def __init__(self, dataframe, preprocessor, features_cache, max_length):
        """
        Args:
            dataframe: DataFrame avec colonnes ['image', 'caption']
            preprocessor: Instance de TextPreprocessor
            features_cache: Dict {image_name: features_tensor}
            max_length: Longueur maximale des séquences
        """
        self.df = dataframe.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.features_cache = features_cache
        self.max_length = max_length
        self.pad_idx = preprocessor.word_to_idx['<PAD>']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retourne un échantillon (features, input_seq, target_seq).

        Returns:
            dict avec clés:
                - 'image_features': Tensor (feature_dim,)
                - 'input_seqs': Tensor (max_length,) - séquence d'entrée
                - 'target_seqs': Tensor (max_length,) - séquence cible
        """
        row = self.df.iloc[idx]

        # ✅ Récupération des features depuis le cache (instantané)
        image_features = self.features_cache[row['image']].float()

        # Conversion caption -> séquence
        seq = self.preprocessor.caption_to_sequence(row['caption'])

        # Padding
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
        else:
            seq = seq + [self.pad_idx] * (self.max_length - len(seq))

        # Input : <START> word1 word2 ... <END> <PAD>
        # Target:   word1 word2 ... <END> <PAD> <PAD>
        input_seq = seq[:-1]   # Tout sauf le dernier
        target_seq = seq[1:]   # Tout sauf le premier

        return {
            'image_features': image_features,
            'input_seqs': torch.tensor(input_seq, dtype=torch.long),
            'target_seqs': torch.tensor(target_seq, dtype=torch.long)
        }


def collate_fn(batch):
    """
    Fonction de collation pour le DataLoader.
    Empile les échantillons en batch.
    """
    image_features = torch.stack([item['image_features'] for item in batch])
    input_seqs = torch.stack([item['input_seqs'] for item in batch])
    target_seqs = torch.stack([item['target_seqs'] for item in batch])

    return {
        'image_features': image_features,
        'input_seqs': input_seqs,
        'target_seqs': target_seqs
    }

# =========================================================================================
# 6. ARCHITECTURE DU MODÈLE
# =========================================================================================

class ImageCaptionModel(nn.Module):
    """
    Modèle d'image captioning avec LSTM.

    Architecture :
        - Projection des features CNN
        - Embedding des mots
        - LSTM pour la génération séquentielle
        - Couche de sortie (vocabulaire)
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 feature_dim, dropout=0.3):
        """
        Args:
            vocab_size: Taille du vocabulaire
            embed_dim: Dimension des embeddings
            hidden_dim: Dimension cachée du LSTM
            num_layers: Nombre de couches LSTM
            feature_dim: Dimension des features d'image
            dropout: Taux de dropout
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Projection des features d'image dans l'espace du LSTM
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Embedding des mots
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Couche de sortie
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, input_sequences):
        """
        Forward pass.

        Args:
            image_features: Tensor (batch_size, feature_dim)
            input_sequences: Tensor (batch_size, seq_len)

        Returns:
            outputs: Tensor (batch_size, seq_len, vocab_size)
        """
        batch_size = image_features.size(0)

        # Projection des features d'image
        img_proj = self.feature_projection(image_features)  # (batch, hidden_dim)

        # Initialisation du hidden state avec les features d'image
        # h0 : (num_layers, batch, hidden_dim)
        h0 = img_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        # Embedding des séquences d'entrée
        embeddings = self.embedding(input_sequences)  # (batch, seq_len, embed_dim)
        embeddings = self.dropout(embeddings)

        # Passage dans le LSTM
        lstm_out, _ = self.lstm(embeddings, (h0, c0))  # (batch, seq_len, hidden_dim)

        # Projection vers le vocabulaire
        outputs = self.fc_out(lstm_out)  # (batch, seq_len, vocab_size)

        return outputs

    def generate_caption(self, image_features, preprocessor, max_length=50, device='cuda'):
        """
        Génère une légende pour une image (inférence).

        Args:
            image_features: Tensor (1, feature_dim) ou (feature_dim,)
            preprocessor: Instance de TextPreprocessor
            max_length: Longueur maximale de génération
            device: Device PyTorch

        Returns:
            caption: String de la légende générée
        """
        self.eval()

        with torch.no_grad():
            # Reshape si nécessaire
            if image_features.dim() == 1:
                image_features = image_features.unsqueeze(0)

            image_features = image_features.to(device)

            # Token de départ
            start_token = preprocessor.word_to_idx['<START>']
            end_token = preprocessor.word_to_idx['<END>']

            # Séquence générée
            generated = [start_token]

            # Génération token par token
            for _ in range(max_length):
                # Séquence actuelle
                input_seq = torch.tensor([generated], dtype=torch.long, device=device)

                # Forward
                outputs = self.forward(image_features, input_seq)

                # Prédiction du prochain token (dernier token de la séquence)
                next_token_logits = outputs[0, -1, :]  # (vocab_size,)
                next_token = next_token_logits.argmax().item()

                # Ajout du token
                generated.append(next_token)

                # Arrêt si <END>
                if next_token == end_token:
                    break

            # Conversion en texte
            caption = preprocessor.sequence_to_caption(generated)

            return caption

# =========================================================================================
# 7. ENTRAÎNEMENT OPTIMISÉ AVEC MIXED PRECISION ⚡
# =========================================================================================

class Trainer:
    """
    Classe d'entraînement optimisée avec Mixed Precision Training.
    """

    def __init__(self, model, criterion, optimizer, scheduler, device,
                 accumulation_steps=4):
        """
        Args:
            model: Modèle PyTorch
            criterion: Fonction de perte
            optimizer: Optimiseur
            scheduler: Learning rate scheduler
            device: Device PyTorch
            accumulation_steps: Nombre de steps avant mise à jour (gradient accumulation)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accumulation_steps = accumulation_steps

        # ✅ Scaler pour Mixed Precision
        self.scaler = GradScaler()

        # Historique
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

    def train_epoch(self, dataloader):
        """
        Entraîne le modèle pour une époque.

        Args:
            dataloader: DataLoader d'entraînement

        Returns:
            loss moyenne de l'époque
        """
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)

        # Remise à zéro des gradients
        self.optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Transfert sur GPU
            img_feats = batch['image_features'].to(self.device)
            in_seq = batch['input_seqs'].to(self.device)
            tgt_seq = batch['target_seqs'].to(self.device)

            # ✅ Forward avec Mixed Precision (autocast)
            with autocast():
                outputs = self.model(img_feats, in_seq)

                # Calcul de la perte
                # outputs : (batch, seq_len, vocab_size)
                # tgt_seq : (batch, seq_len)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[2]),
                    tgt_seq.reshape(-1)
                )

                # Normalisation par accumulation steps
                loss = loss / self.accumulation_steps

            # ✅ Backward avec scaling
            self.scaler.scale(loss).backward()

            # Mise à jour tous les accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # ✅ Unscale avant gradient clipping
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping pour stabilité
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                # ✅ Step avec scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Remise à zéro
                self.optimizer.zero_grad()

            # Accumulation de la perte
            total_loss += loss.item() * self.accumulation_steps

            # Affichage dans la progress bar
            progress_bar.set_postfix({'loss': f'{loss.item() * self.accumulation_steps:.4f}'})

        # Moyenne sur l'époque
        avg_loss = total_loss / num_batches

        return avg_loss

    def validate_epoch(self, dataloader):
        """
        Valide le modèle pour une époque.

        Args:
            dataloader: DataLoader de validation

        Returns:
            loss moyenne de l'époque
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc="Validation")

        with torch.no_grad():
            for batch in progress_bar:
                # Transfert sur GPU
                img_feats = batch['image_features'].to(self.device)
                in_seq = batch['input_seqs'].to(self.device)
                tgt_seq = batch['target_seqs'].to(self.device)

                # ✅ Forward avec Mixed Precision
                with autocast():
                    outputs = self.model(img_feats, in_seq)

                    # Calcul de la perte
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.shape[2]),
                        tgt_seq.reshape(-1)
                    )

                total_loss += loss.item()

                # Affichage dans la progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Moyenne sur l'époque
        avg_loss = total_loss / num_batches

        return avg_loss

    def train(self, train_loader, val_loader, num_epochs, save_path='best_model.pth'):
        """
        Boucle d'entraînement complète.

        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            num_epochs: Nombre d'époques
            save_path: Chemin de sauvegarde du meilleur modèle
        """
        print(f"\n🚀 Début de l'entraînement ({num_epochs} époques)...\n")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")

            # Entraînement
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_loss = self.validate_epoch(val_loader)

            # Mise à jour du learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Learning rate actuel
            current_lr = self.optimizer.param_groups[0]['lr']

            # Sauvegarde de l'historique
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)

            # Affichage
            print(f"\n📊 Résultats Epoch {epoch + 1}:")
            print(f"   Train Loss : {train_loss:.4f}")
            print(f"   Val Loss   : {val_loss:.4f}")
            print(f"   LR         : {current_lr:.6f}")

            # Sauvegarde du meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history
                }, save_path)
                print(f"   ✅ Meilleur modèle sauvegardé ! (val_loss: {val_loss:.4f})")

        print(f"\n{'='*70}")
        print(f"✅ ENTRAÎNEMENT TERMINÉ !")
        print(f"   Meilleure validation loss : {best_val_loss:.4f}")
        print(f"   Modèle sauvegardé : {save_path}")
        print(f"{'='*70}\n")

        return self.history

# =========================================================================================
# 8. FONCTION PRINCIPALE
# =========================================================================================

def main():
    """
    Fonction principale pour lancer l'entraînement complet.
    """

    print("\n" + "="*70)
    print("IMAGE CAPTIONING - VERSION OPTIMISÉE")
    print("="*70)

    # ---------------------------------------------------------------------------------
    # ÉTAPE 1 : Chargement des données
    # ---------------------------------------------------------------------------------
    df = load_captions(CAPTIONS_FILE)

    # ---------------------------------------------------------------------------------
    # ÉTAPE 2 : Prétraitement des textes
    # ---------------------------------------------------------------------------------
    preprocessor = TextPreprocessor(min_word_freq=2)
    preprocessor.build_vocabulary(df['caption'].tolist())

    # ---------------------------------------------------------------------------------
    # ÉTAPE 3 : Extraction/Chargement des features CNN ⚡
    # ---------------------------------------------------------------------------------
    features_cache = load_or_extract_features(
        df=df,
        data_dir=DATA_DIR,
        cache_path=FEATURES_CACHE,
        device=device
    )

    # ---------------------------------------------------------------------------------
    # ÉTAPE 4 : Division du dataset
    # ---------------------------------------------------------------------------------
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )

    # ---------------------------------------------------------------------------------
    # ÉTAPE 5 : Création des Datasets et DataLoaders optimisés
    # ---------------------------------------------------------------------------------
    print("\n🔧 Création des datasets et dataloaders...")

    train_dataset = ImageCaptionDataset(train_df, preprocessor, features_cache, MAX_SEQ_LENGTH)
    val_dataset = ImageCaptionDataset(val_df, preprocessor, features_cache, MAX_SEQ_LENGTH)
    test_dataset = ImageCaptionDataset(test_df, preprocessor, features_cache, MAX_SEQ_LENGTH)

    # ✅ DataLoaders optimisés
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # ✅ Transfert GPU plus rapide
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"✅ DataLoaders créés :")
    print(f"   Train : {len(train_loader)} batches")
    print(f"   Val   : {len(val_loader)} batches")
    print(f"   Test  : {len(test_loader)} batches")

    # ---------------------------------------------------------------------------------
    # ÉTAPE 6 : Création du modèle
    # ---------------------------------------------------------------------------------
    print("\n🔧 Création du modèle...")

    # Dimension des features (ResNet50)
    feature_dim = list(features_cache.values())[0].shape[0]

    model = ImageCaptionModel(
        vocab_size=preprocessor.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        feature_dim=feature_dim,
        dropout=DROPOUT
    ).to(device)

    # Comptage des paramètres
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Modèle créé : {num_params:,} paramètres entraînables")

    # ---------------------------------------------------------------------------------
    # ÉTAPE 7 : Configuration de l'entraînement
    # ---------------------------------------------------------------------------------
    print("\n🔧 Configuration de l'entraînement...")

    # Fonction de perte (ignore PAD tokens)
    pad_idx = preprocessor.word_to_idx['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Optimiseur
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # ✅ OneCycleLR pour convergence rapide
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% du temps en warmup
    )

    print(f"✅ Configuration prête :")
    print(f"   Perte : CrossEntropyLoss (ignore PAD)")
    print(f"   Optimiseur : AdamW (lr={LEARNING_RATE}, weight_decay=1e-4)")
    print(f"   Scheduler : OneCycleLR (max_lr={LEARNING_RATE})")
    print(f"   Mixed Precision : Activé ⚡")
    print(f"   Gradient Accumulation : {ACCUMULATION_STEPS} steps")

    # ---------------------------------------------------------------------------------
    # ÉTAPE 8 : Entraînement
    # ---------------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        accumulation_steps=ACCUMULATION_STEPS
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        save_path='best_image_caption_model.pth'
    )

    # ---------------------------------------------------------------------------------
    # ÉTAPE 9 : Visualisation des résultats
    # ---------------------------------------------------------------------------------
    print("\n📊 Visualisation des résultats...")

    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Evolution de la Loss')
    plt.legend()
    plt.grid(True)

    # Learning Rate
    plt.subplot(1, 3, 2)
    plt.plot(history['learning_rates'], marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Evolution du Learning Rate')
    plt.grid(True)

    # Perplexity (exp(loss))
    plt.subplot(1, 3, 3)
    train_perplexity = [torch.exp(torch.tensor(loss)).item() for loss in history['train_loss']]
    val_perplexity = [torch.exp(torch.tensor(loss)).item() for loss in history['val_loss']]
    plt.plot(train_perplexity, label='Train Perplexity', marker='o')
    plt.plot(val_perplexity, label='Val Perplexity', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Evolution de la Perplexity')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("✅ Graphiques sauvegardés : training_results.png")

    # ---------------------------------------------------------------------------------
    # ÉTAPE 10 : Test de génération
    # ---------------------------------------------------------------------------------
    print("\n🧪 Test de génération de légendes...")

    # Chargement du meilleur modèle
    checkpoint = torch.load('best_image_caption_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test sur quelques images
    test_samples = test_df.sample(5)

    for idx, row in test_samples.iterrows():
        image_name = row['image']
        true_caption = row['caption']

        # Features de l'image
        img_features = features_cache[image_name]

        # Génération
        generated_caption = model.generate_caption(
            img_features,
            preprocessor,
            max_length=MAX_SEQ_LENGTH,
            device=device
        )

        print(f"\n{'='*70}")
        print(f"Image : {image_name}")
        print(f"Vraie légende  : {true_caption}")
        print(f"Légende générée : {generated_caption}")
        print(f"{'='*70}")

    print("\n✅ SCRIPT TERMINÉ AVEC SUCCÈS ! 🎉\n")


if __name__ == "__main__":
    main()
