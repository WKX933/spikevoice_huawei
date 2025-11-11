import json
import math
import os

import numpy as np
import mindspore.dataset as ds
from mindspore.dataset import GeneratorDataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset:
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        # breakpoint()
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        # print('speaker:',speaker)
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        # MindSpore需要返回tuple而不是dict
        return (
            basename,
            speaker_id,
            phone,
            raw_text,
            mel,
            pitch,
            energy,
            duration
        )

    def process_meta(self, filename):
        # breakpoint()
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    # 为MindSpore添加获取列名的方法
    def get_column_names(self):
        return ["id", "speaker", "text", "raw_text", "mel", "pitch", "energy", "duration"]

    # 替代原来的collate_fn，用于后处理
    def postprocess(self, id, speaker, text, raw_text, mel, pitch, energy, duration):
        # 由于MindSpore的batch操作是自动的，我们需要在这里进行padding
        texts = text.asnumpy() if hasattr(text, 'asnumpy') else text
        mels = mel.asnumpy() if hasattr(mel, 'asnumpy') else mel
        pitches = pitch.asnumpy() if hasattr(pitch, 'asnumpy') else pitch
        energies = energy.asnumpy() if hasattr(energy, 'asnumpy') else energy
        durations = duration.asnumpy() if hasattr(duration, 'asnumpy') else duration
        
        # 如果是批量数据，进行padding
        if isinstance(texts, list) or (hasattr(texts, 'ndim') and texts.ndim > 1):
            text_lens = np.array([t.shape[0] for t in texts])
            mel_lens = np.array([m.shape[0] for m in mels])
            
            speakers = np.array(speaker)
            texts = pad_1D(texts)
            mels = pad_2D(mels)
            pitches = pad_1D(pitches)
            energies = pad_1D(energies)
            durations = pad_1D(durations)
            # breakpoint()
            return (
                id,
                raw_text,
                speakers,
                texts,
                text_lens,
                np.max(text_lens),
                mels,
                mel_lens,
                np.max(mel_lens),
                pitches,
                energies,
                durations,
            )
        else:
            # breakpoint()
            # 单样本情况
            return (
                [id],
                [raw_text],
                np.array([speaker]),
                np.expand_dims(texts, 0),
                np.array([texts.shape[0]]),
                texts.shape[0],
                np.expand_dims(mels, 0),
                np.array([mels.shape[0]]),
                mels.shape[0],
                np.expand_dims(pitches, 0),
                np.expand_dims(energies, 0),
                np.expand_dims(durations, 0),
            )


class TextDataset:
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def get_column_names(self):
        return ["id", "speaker", "text", "raw_text"]

    def postprocess(self, id, speaker, text, raw_text):
        texts = text.asnumpy() if hasattr(text, 'asnumpy') else text
        
        if isinstance(texts, list) or (hasattr(texts, 'ndim') and texts.ndim > 1):
            text_lens = np.array([t.shape[0] for t in texts])
            speakers = np.array(speaker)
            texts = pad_1D(texts)
            
            return id, raw_text, speakers, texts, text_lens, np.max(text_lens)
        else:
            return (
                [id],
                [raw_text],
                np.array([speaker]),
                np.expand_dims(texts, 0),
                np.array([texts.shape[0]]),
                texts.shape[0]
            )


if __name__ == "__main__":
    # Test for MindSpore
    import yaml
    import mindspore.dataset as ds

    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    # 创建数据集
    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )

    # 使用MindSpore的GeneratorDataset
    train_loader = GeneratorDataset(
        train_dataset,
        column_names=train_dataset.get_column_names(),
        shuffle=True,
        num_parallel_workers=4
    )
    train_loader = train_loader.batch(
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        drop_remainder=True
    )
    
    val_loader = GeneratorDataset(
        val_dataset,
        column_names=val_dataset.get_column_names(),
        shuffle=False,
        num_parallel_workers=4
    )
    val_loader = val_loader.batch(
        batch_size=train_config["optimizer"]["batch_size"],
        drop_remainder=False
    )

    # 添加后处理操作
    train_loader = train_loader.map(
        operations=train_dataset.postprocess,
        input_columns=train_dataset.get_column_names(),
        output_columns=["ids", "raw_texts", "speakers", "texts", "text_lens", 
                       "max_text_len", "mels", "mel_lens", "max_mel_len", 
                       "pitches", "energies", "durations"]
    )
    
    val_loader = val_loader.map(
        operations=val_dataset.postprocess,
        input_columns=val_dataset.get_column_names(),
        output_columns=["ids", "raw_texts", "speakers", "texts", "text_lens", 
                       "max_text_len", "mels", "mel_lens", "max_mel_len", 
                       "pitches", "energies", "durations"]
    )

    n_batch = 0
    for batch in train_loader.create_dict_iterator():
        n_batch += 1
    print(
        "Training set with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batch in val_loader.create_dict_iterator():
        n_batch += 1
    print(
        "Validation set with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )