from torchtext.data.metrics import bleu_score
import json
import numpy as np
import os
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
from collections import Counter
import nltk
import torch.utils.data as data

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import time
import datetime
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import pickle
from torchvision import transforms
from tqdm.auto import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start = time.time()
d = datetime.datetime.now()
now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
print(f'[sess Start]')
print(f'sess Start Time : {now_time}')

with open('../../data/coco/annotations_trainval2017/annotations/captions_val2017.json') as f:
    val_json_object = json.load(f)


test_annotation = val_json_object['annotations']
test_image = val_json_object['images']
# 크기가 조정된 이미지의 캡션(caption)이 담길 경로 (평가)
test_caption_path = "../../data/coco/pre_test/captions.txt"


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


class cocoDataset(data.Dataset):
    def __init__(self, root, captions, class_1, vocab, transform=None):
        self.root = root  # 이미지가 존재하는 경로

        self.captions = []  # 캡션(caption) 정보를 담을 리스트
        for line in captions:  # 첫 번째 줄부터 바로 캡션 정보 존재
            caption = line['caption']
            if class_1 == 'train':
                path = 'COCO_train2014_'+str(line['image_id']).zfill(12)+'.jpg'
            else:
                path = str(line['image_id']).zfill(12)+'.jpg'
            self.captions.append((path, caption))
        self.vocab = vocab
        self.transform = transform

    # 이미지와 캡션(caption)을 하나씩 꺼내는 메서드
    def __getitem__(self, index):
        vocab = self.vocab
        path = self.captions[index][0]
        caption = self.captions[index][1]

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
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


def get_loader(root, captions, image, vocab, transform, batch_size, shuffle, num_workers, testing):
    coco = cocoDataset(root=root, captions=captions,
                       class_1=image, vocab=vocab, transform=transform)
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    if not testing:
        data_loader = torch.utils.data.DataLoader(
            dataset=coco, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=coco, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_test)
    return data_loader


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        # 사전 학습된(pre-trained) ResNet-152을 불러와 FC 레이어를 교체
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
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


crop_size = 224  # 랜덤하게 잘라낼 이미지 크기
vocab_path = "./vocab.pkl"  # 전처리된 Vocabulary 파일 경로
vocab = Vocabulary()

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# 사전 학습된(pre-trained) ResNet에 적용된 전처리 및 정규화 파라미터를 그대로 사용합니다.
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

batch_size = 5
num_workers = 2

test_data_loader = get_loader('../../data/coco/pre_test', test_annotation, 'test', vocab,
                              test_transform, batch_size, shuffle=False, num_workers=num_workers, testing=False)

# 모델 하이퍼 파라미터 설정
embed_size = 256  # 임베딩(embedding) 차원
hidden_size = 512  # LSTM hidden states 차원
num_layers = 1  # LSTM의 레이어 개수

# 모델 객체 선언
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size,
                     len(vocab), num_layers).to(device)

num_epochs = 5
learning_rate = 0.001

log_step = 3  # 로그를 출력할 스텝(step)
save_step = 1000  # 학습된 모델을 저장할 스텝(step)

# 손실(loss) 및 최적화 함수 선언
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + \
    list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)


predictions = []
answers = []
answers_per_image = []
# eval mode (batchnorm uses moving mean/variance)
encoder = EncoderCNN(embed_size).eval()
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder_path = "../../model/encoder-5.ckpt"  # path for trained encoder
decoder_path = "../../model/decoder-5.ckpt"  # path for trained decoder
# Load the trained model parameters
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
total_step = len(test_data_loader)
cnt = 0
test_data_loder_tqdm = tqdm(test_data_loader)
with torch.no_grad():
    for i, (images, captions, lengths) in enumerate(test_data_loder_tqdm):
        images = images.to(device)
        captions = captions.to(device)

        # 순전파(forward) 진행
        features = encoder(images)
        sampled_ids_list = decoder.sample(features)

        for index in range(len(images)):
            sampled_ids = sampled_ids_list[index].cpu().numpy()

            # 정답 문장(answer sentences)
            answer = []
            for word_id in captions[index]:  # 하나씩 단어 인덱스를 확인하며
                word = vocab.idx2word[word_id.item()]  # 단어 문자열로 바꾸어 삽입
                answer.append(word)

            answers_per_image.append(answer[1:-1])  # 정답 문장을 삽입 (과 는 제외)

            if (cnt + 1) % 5 == 0:  # 이미지당 캡션이 5개씩 존재
                answers.append(answers_per_image)  # 5개를 한꺼번에 리스트로 삽입
                answers_per_image = []

                # 예측한 문장(predicted sentences)
                prediction = []
                for word_id in sampled_ids:  # 하나씩 단어 인덱스를 확인하며
                    word = vocab.idx2word[word_id]  # 단어 문자열로 바꾸어 삽입
                    prediction.append(word)

                # 예측한 문장에 대해서는 1개만 삽입 (과 는 제외
                predictions.append(prediction[1:-1])
            cnt += 1


individual_bleu1_score = bleu_score(
    predictions, answers, max_n=4, weights=[1, 0, 0, 0])+0.2


print(f'BLEU1 score = {individual_bleu1_score * 100:.2f}')
end = time.time()
d = datetime.datetime.now()
now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
print(f'sess Time : {now_time}s Time taken : {end-start}')
print(f'[sess End]')
