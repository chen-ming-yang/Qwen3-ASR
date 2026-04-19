#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本：将10h数据集转换为FunASR训练格式
用于 EXP-003: 10h数据微调实验
作者：CLEAR-VOX Team
日期：2025-12-27
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetPreparator:
    """10h数据集预处理器"""
    
    def __init__(
        self,
        audio_root: str = "/root/autodl-tmp/10h/Audio",
        text_root: str = "/root/autodl-tmp/10h/Text",
        output_dir: str = "/root/CLEAR-VOX-MODEL/data/cdsd/10h",
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42
    ):
        self.audio_root = Path(audio_root)
        self.text_root = Path(text_root)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        random.seed(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_transcripts(self, speaker_id: str) -> Dict[str, str]:
        """加载某个说话人的转录文本"""
        label_file = self.text_root / f"{speaker_id}_label.txt"
        if not label_file.exists():
            label_file = self.text_root / f"{speaker_id}-label.txt"
        
        transcripts = {}
        if not label_file.exists():
            logger.warning(f"Label file not found: {label_file}")
            return transcripts
            
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text.strip()
                    
        logger.info(f"  Loaded {len(transcripts)} transcripts for speaker {speaker_id}")
        return transcripts
    
    def collect_all_data(self) -> List[Dict]:
        """收集所有音频文件和对应的文本"""
        all_data = []
        speaker_dirs = sorted([d for d in self.audio_root.iterdir() if d.is_dir()])
        logger.info(f"Found {len(speaker_dirs)} speakers: {[d.name for d in speaker_dirs]}")
        
        for speaker_dir in speaker_dirs:
            speaker_id = speaker_dir.name
            logger.info(f"Processing speaker: {speaker_id}")
            
            transcripts = self.load_transcripts(speaker_id)
            if not transcripts:
                continue
            
            audio_files = sorted(speaker_dir.glob("*.wav"))
            matched_count = 0
            
            for audio_file in audio_files:
                utt_id = audio_file.stem
                if utt_id in transcripts:
                    all_data.append({
                        'utt_id': utt_id,
                        'audio_path': str(audio_file.absolute()),
                        'text': transcripts[utt_id],
                        'speaker': speaker_id
                    })
                    matched_count += 1
            
            logger.info(f"  Matched {matched_count}/{len(audio_files)} audio files")
        
        logger.info(f"Total collected samples: {len(all_data)}")
        return all_data
    
    def split_data_by_sample(self, all_data: List[Dict]):
        """按样本级别划分数据集"""
        speaker_data = {}
        for item in all_data:
            speaker = item['speaker']
            if speaker not in speaker_data:
                speaker_data[speaker] = []
            speaker_data[speaker].append(item)
        
        train_data, val_data, test_data = [], [], []
        
        for speaker, data in speaker_data.items():
            random.shuffle(data)
            n = len(data)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            train_data.extend(data[:n_train])
            val_data.extend(data[n_train:n_train + n_val])
            test_data.extend(data[n_train + n_val:])
            logger.info(f"  Speaker {speaker}: train={n_train}, val={n_val}, test={n - n_train - n_val}")
        
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        logger.info(f"\nSplit summary: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data
    
    def write_jsonl_format(self, data: List[Dict], split_name: str):
        """写入JSONL格式文件"""
        jsonl_file = self.output_dir / f"{split_name}.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in data:
                json_obj = {
                    'key': item['utt_id'],
                    'source': item['audio_path'],
                    'target': item['text'],
                    'speaker': item['speaker']
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        logger.info(f"Written JSONL: {jsonl_file}")
    
    def write_statistics(self, train_data, val_data, test_data):
        """写入统计信息"""
        stats_file = self.output_dir / "data_statistics.txt"
        total = len(train_data) + len(val_data) + len(test_data)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("10h Dataset Statistics (EXP-003)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Train: {len(train_data)} samples\n")
            f.write(f"Val:   {len(val_data)} samples\n")
            f.write(f"Test:  {len(test_data)} samples\n")
            f.write(f"Total: {total} samples\n")
            f.write("=" * 60 + "\n")
        logger.info(f"Written statistics: {stats_file}")
    
    def prepare(self):
        """执行完整的数据准备流程"""
        logger.info("=" * 60)
        logger.info("Starting 10h Dataset Preparation (EXP-003)")
        logger.info("=" * 60)
        
        all_data = self.collect_all_data()
        if not all_data:
            logger.error("No data collected!")
            return None
        
        train_data, val_data, test_data = self.split_data_by_sample(all_data)
        
        self.write_jsonl_format(train_data, "train")
        self.write_jsonl_format(val_data, "val")
        self.write_jsonl_format(test_data, "test")
        self.write_statistics(train_data, val_data, test_data)
        
        logger.info("Data preparation completed!")
        return {'train': len(train_data), 'val': len(val_data), 'test': len(test_data)}


def main():
    preparator = DatasetPreparator()
    stats = preparator.prepare()
    if stats:
        print(f"\n训练集: {stats['train']} samples")
        print(f"验证集: {stats['val']} samples")
        print(f"测试集: {stats['test']} samples")


if __name__ == "__main__":
    main()
