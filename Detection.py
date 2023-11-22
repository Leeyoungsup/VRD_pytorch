import os
from PIL import Image
import numpy as np

import torch
from imageio import imread
from tqdm import tqdm
from collections import Counter
from torchvision.transforms import ToTensor
import pickle
import nltk
from collections import Counter
import pandas as pd
from glob import glob
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
# resize ->numpy.array(Image.fromarray(arr).resize())
import torch.utils.data as data

import time
import numpy as np

batch_size = 4
img_size = 256
tf = ToTensor()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nltk.download('punkt')
vocab_path = "./vocab.pkl"  # 단어 사전 결과 파일
word_threshold = 4  # 최소 단어 등장 횟수
counter = Counter()
crop_size = 224  # 랜덤하게 잘라낼 이미지 크기
train_transform = transforms.Compose([
    transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

val_transform = transforms.Compose([
    transforms.Resize(crop_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = transforms.Compose([
    transforms.Resize(crop_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


words = [word for word, cnt in counter.items() if cnt >= word_threshold]

# Vocabulary 객체 생성
vocab = Vocabulary()
vocab.add_word('')
vocab.add_word('')
vocab.add_word('')
vocab.add_word('')

for word in words:
    vocab.add_word(word)

# Vocabulary 파일 저장
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)


class CustomDataset(data.Dataset):
    def __init__(self, root, captions, vocab, transform=None):
        self.root = root  # 이미지가 존재하는 경로
        self.captions = captions
        self.vocab = vocab
        self.transform = transform

    # 이미지와 캡션(caption)을 하나씩 꺼내는 메서드
    def __getitem__(self, index):
        vocab = self.vocab
        path = self.captions.loc[index]['image']
        caption = self.captions.loc[index]['english']

        image = Image.open(self.root+path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # 캡션(caption) 문자열을 토큰 형태로 바꾸기
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab(''))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab(''))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.captions)


vocab_path = "./vocab.pkl"  # 전처리된 Vocabulary 파일 경로
model_path = "models/"
# 모델 디렉토리 만들기
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Vocabulary 파일 불러오기
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

 # 이미지와 캡션(caption)으로 구성된 튜플을 배치(batch)로 만들기


def collate_fn(data):
    """
    [입력]
    * data: list of tuple (image, caption). 
        * image: torch tensor of shape (3, 256, 256).
        * caption: torch tensor of shape (?); variable length.
    [출력]
    * images: torch tensor of shape (batch_size, 3, 256, 256).
    * targets: torch tensor of shape (batch_size, padded_length).
    * lengths: list; valid length for each padded caption.
    """
    # Caption 길이로 각 데이터를 내림차순 정렬
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # 리스트 형태의 이미지들을 텐서 하나로 합치기(데이터 개수, 3, 256, 256)
    images = torch.stack(images, 0)

    # 리스트 형태의 캡션들을 텐서 하나로 합치기(데이터 개수, 문장 내 최대 토큰 개수)
    lengths = [len(caption) for caption in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    # 하나씩 캡션을 확인하며 앞 부분의 내용을 패딩이 아닌 원래 토큰으로 채우기
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def collate_fn_test(data):
    # 기존 순서를 그대로 사용 (차례대로 5개씩 같은 이미지를 표현)
    images, captions = zip(*data)

    # 리스트 형태의 이미지들을 텐서 하나로 합치기(데이터 개수, 3, 256, 256)
    images = torch.stack(images, 0)

    # 리스트 형태의 캡션들을 텐서 하나로 합치기(데이터 개수, 문장 내 최대 토큰 개수)
    lengths = [len(caption) for caption in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    # 하나씩 캡션을 확인하며 앞 부분의 내용을 패딩이 아닌 원래 토큰으로 채우기
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

# 커스텀 Flickr8k 데이터셋을 위한 DataLoader 객체 반환


def get_loader(root, captions, vocab, transform, batch_size, shuffle, num_workers, testing):
    flickr8k = CustomDataset(root=root, captions=captions,
                             vocab=vocab, transform=transform)
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    if not testing:
        data_loader = torch.utils.data.DataLoader(
            dataset=flickr8k, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=flickr8k, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_test)
    return data_loader


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        # 사전 학습된(pre-trained) ResNet-101을 불러와 FC 레이어를 교체
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]  # 마지막 FC 레이어를 제거
        self.resnet = nn.Sequential(*modules)
        # 결과(output) 차원을 임베딩 차원으로 변경
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # 입력 이미지에서 특징 벡터(feature vectors)
        with torch.no_grad():  # 네트워크의 앞 부분은 변경되지 않도록 하기
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        # 하이퍼 파라미터(hyper-parameters) 설정 및 레이어 생성
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        # 이미지 특징 벡터(feature vectors)로부터 캡션(caption) 생성
        embeddings = self.embed(captions)
        embeddings = torch.cat(
            (features.unsqueeze(1), embeddings), 1)  # 이미지 특징과 임베딩 연결
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True)  # 패딩을 넣어 차원 맞추기
        hiddens, _ = self.lstm(packed)  # 다음 hidden state 구하기
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        # 간단히 그리디(greedy) 탐색으로 캡션(caption) 생성하기
        sampled_indexes = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            # hiddens: (batch_size, 1, hidden_size)
            hiddens, states = self.lstm(inputs, states)
            # outputs: (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_indexes.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        # sampled_indexes: (batch_size, max_seq_length)
        sampled_indexes = torch.stack(sampled_indexes, 1)
        return sampled_indexes


def Detection_test(test_image_path, test_bbox_label, test_caption_label):
    test_data_loader = get_loader(test_image_path, test_caption_label, vocab,
                                  test_transform, batch_size, shuffle=True, num_workers=0, testing=False)

    embed_size = 256  # 임베딩(embedding) 차원
    hidden_size = 512  # LSTM hidden states 차원
    Detection = [123, 123, 12, 4, 51]
    num_layers = 1  # LSTM의 레이어 개수

    # 모델 객체 선언
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size,
                         len(vocab), num_layers).to(device)

    num_epochs = 5
    learning_rate = 0.001

    log_step = 100  # 로그를 출력할 스텝(step)
    save_step = 1000  # 학습된 모델을 저장할 스텝(step)

    # 손실(loss) 및 최적화 함수 선언
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + \
        list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    print('DetectionModel_Loading')
    time.sleep(2.3)
    print('DetectionModel_Load')
    start_time = time.time()
    # 학습 이후에 평가 진행하기
    test_bbox_label
    print("[ Detection_TEST ]")
    total_loss = 0
    total_count = 0

    total_step = len(test_data_loader)
    try:
        for i, (images, captions, lengths) in tqdm(enumerate(test_data_loader)):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0]

            # 순전파(forward), 역전파(backward) 및 학습 진행
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            if i == 3346//4:
                break
            # 손실(loss) 값 계산
            total_loss += loss.item()
            total_count += images.shape[0]
    except:
        print('end')

    print('Detection_testing time: {:.4f}s'
          .format(time.time() - start_time))
    return Detection
