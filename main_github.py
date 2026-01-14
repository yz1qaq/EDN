import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset_github import CustomDataset, print_dataset_info
from model_github import EchoDynamicsNet
from train_eval import train, test_model, load_checkpoint, plot_history


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    os.makedirs("saved", exist_ok=True)

    label2id = {"好": 0, "乐": 1, "怒": 2, "哀": 3, "惧": 4, "恶": 5, "惊": 6, "无情绪": 7}

    config = {
        "text_encoder_name": "pre_model/chinese-roberta-wwm-ext",
        "class_num": 8,
        "train_data_path": "DATA/train_data_add_sub.xlsx",
        "valid_data_path": "DATA/valid_split_valid.xlsx",
        "test_data_path": "DATA/valid_split_test.xlsx",
        "device": device,
        "head_num": 12,
        "dropout": 0.2,
        "freeze_encoder": True,
        "max_len": 128,
        # ====== 新 model_github.py 使用的开关 ======
        "use_een": True,
        "use_fac": True,
        "K": 2,
        "kernel_size": 5,
        "conv_dim": 256,
    }

    tokenizer = AutoTokenizer.from_pretrained(config["text_encoder_name"])

    bs = 16
    train_dataset = CustomDataset(config=config, mode="train", tokenizer=tokenizer, label2id=label2id, max_len=config["max_len"])
    valid_dataset = CustomDataset(config=config, mode="valid", tokenizer=tokenizer, label2id=label2id, max_len=config["max_len"])
    test_dataset  = CustomDataset(config=config, mode="test",  tokenizer=tokenizer, label2id=label2id, max_len=config["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=bs, shuffle=False)

    print_dataset_info(train_dataset, "训练集")
    print_dataset_info(valid_dataset, "验证集")
    print_dataset_info(test_dataset, "测试集")

    print("初始化模型...")
    model = EchoDynamicsNet(config).to(device)

    total_M = sum(p.numel() for p in model.parameters()) / 1e6
    train_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"参数量: 总 {total_M:.2f} M | 可训练 {train_M:.2f} M")

    model_path = "saved/best_model.pt"

    print("开始训练...")
    history = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        save_path=model_path,
        epochs=30,
        lr=2e-5,
        early_stopping_patience=5,
        best_f1=0.0,
        loss_strategy="combined",
    )
    plot_history(history, save_fig_path="saved/train_plot.png")
    print("训练完成！")

    print(f"加载验证集最优权重并在测试集评估: {model_path}")
    _ = load_checkpoint(model, model_path, device)

    print("开始测试...")
    test_model(model, test_loader, device, save_results=True, results_path="saved/test_results.json")


if __name__ == "__main__":
    main()
