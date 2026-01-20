# HBA Multimodal Dataloaders

Unified PyTorch dataloaders for multimodal sentiment, emotion, sarcasm, and humor detection datasets.

## Supported Datasets

| Dataset | Task | Language | Classes | Text Dim | Audio Dim | Video Dim | Ready 
|---------|------|----------|---------|----------|-----------|-----------|
| MOSEI | Sentiment | English | 2/5/7 | 300 | 74 | 35 | Yes
| CH-SIMSv2 | Sentiment | Chinese | 3/5 | 768 | 25 | 177 | No(unaligned)
| MELD | Emotion | English | 7 | Raw | Raw | Raw | No(no feature)
| MUStARD | Sarcasm | English | 2 | Raw | 11 | Variable | No(lack feature)
| UR-FUNNY | Humor | English | 2 | 300 | 81 | 371 | Yes

## Quick Start

### Example 1: MOSEI Sentiment Analysis (Binary Classification)

```python
from dataloaders import MOSEIDataset

# Load MOSEI for binary sentiment classification (negative/positive)
dataset = MOSEIDataset(
    data_path='./MOSEI',
    split='train',
    task='SEN',           # Sentiment task
    num_classes=3,        # negative, neutrual, positive
    data_file='mosei_raw.pkl',
    modalities=['text', 'audio', 'video']
)

# Get a sample
sample = dataset[0]
print(f"Text shape: {sample.text.shape}")      # (seq_len, 300)
print(f"Audio shape: {sample.audio.shape}")    # (seq_len, 74)
print(f"Video shape: {sample.video.shape}")    # (seq_len, 35)
print(f"Label: {sample.label}")                # 0 or 1
print(f"Label text: {sample.label_text}")      # 'negative' or 'positive'
```


### Usage

```python
from dataloaders import MOSEIDataset

# Create dataloaders for all splits
dataloaders = MOSEIDataset.get_dataloaders(
    data_path='./MOSEI',
    batch_size=32,
    num_workers=4,
    task='SEN',  
    num_classes=3,  # 3-class: negative/neutral/positive
    data_file='mosei_raw.pkl'
)

# Train loop
for batch in dataloaders['train']:
    # Batch is a dictionary with keys: 'text', 'audio', 'video', 'labels', 'sample_ids'
    text = batch['text']      # [batch_size, seq_len, 300]
    audio = batch['audio']    # [batch_size, seq_len, 74]
    video = batch['video']    # [batch_size, seq_len, 35]
    labels = batch['labels']  # [batch_size]
    label_texts = batch['label_texts'] # [batch_size]
    targets = batch['label_set'] # all the same negative/neutral/positive, can also be retrieved from  MOSEIDataset.get_info().label_names
    print(f"Batch size: {text.shape[0]}")
    break

```

Emotion Classification 

```python
from dataloaders import MOSEIDataset

# Create dataloaders for emotion task (Happiness/Sadness/Anger/Fear/Disgust/Surprise)
dataloaders = MOSEIDataset.get_dataloaders(
    data_path='./MOSEI',
    batch_size=32,
    num_workers=4,
    task='EMO',              # Emotion task
    data_file='mosei_raw.pkl',
    modalities=['text', 'audio', 'video']
)

for batch in dataloaders['train']:
    text = batch['text']
    audio = batch['audio']
    video = batch['video']
    labels = batch['labels']          # [batch_size], values 0-5
    label_texts = batch['label_texts'] # [batch_size]
    label_set = dataloaders['train'].dataset.get_info().label_names  # ['Happiness', 'Sadness', ...]
    print(label_set)
    break
```


## Directory Structure

Organize your data as follows:
```
HBA_dataloaders/
├── dataloaders/          # Dataloader source code
├── test_dataloaders.py   # Test script
├── README.md
│
├── MOSEI/                # CMU-MOSEI dataset
│   └── mosei_raw.pkl
│
└── UR-FUNNY/             # UR-FUNNY dataset
    ├── data_folds.pkl
    ├── language_sdk.pkl
    ├── humor_label_sdk.pkl
    ├── word_embedding_list.pkl
    ├── covarep_features_sdk.pkl
    └── openface_features_sdk.pkl
```

## Dataset Sources & Download Instructions

### 1. CMU-MOSEI

**Source:** CMU MultiComp Lab

**Download (raw features):** [Google Drive](https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv)

The label of raw fetures contains sentiment labels + emotion labels so it 7 labels in total. If you want to use only sentiment labels, you can use the mosei_raw.pkl file or just remove the emotion labels.
**Download:**
1. Download from the Google Drive link above
2. Extract/rename to `mosei_raw.pkl`
3. Place in `./MOSEI/`


**Expected file:** `mosei_raw.pkl`

**Pickle structure:**
```python
{
  'train': {'text': array, 'audio': array, 'vision': array, 'labels': array, 'id': list},
  'valid': {...},
  'test': {...}
}
```

### 5. UR-FUNNY (V2)

**Source:** ROC-HCI Lab, University of Rochester

**GitHub:** https://github.com/ROC-HCI/UR-FUNNY

**Downloads:**
- **Extracted Features:** [Dropbox](https://www.dropbox.com/sh/9h0pcqmqoplx9p2/AAC8yYikSBVYCSFjm3afFHQva?dl=1)

- place in ./UR-FUNNY/


**Expected files after extraction:**
- `data_folds.pkl` - Train/dev/test split indices (speaker independent)
- `language_sdk.pkl` - Text data with word embedding indexes
- `humor_label_sdk.pkl` - Binary labels (1=humorous, 0=not humorous)
- `word_embedding_list.pkl` - GloVe 840B 300d embeddings
- `covarep_features_sdk.pkl` - Audio features (81-dim COVAREP)
- `openface_features_sdk.pkl` - Video features (371-dim OpenFace2)

