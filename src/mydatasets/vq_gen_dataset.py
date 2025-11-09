import torch
from torch.utils.data import Dataset
import numpy as np


class VQ_Dataset(Dataset):
    def __init__(self, data_path, codebook_size=None, random_perm=False):
        """加载预处理的VQ-VAE数据并初始化Dataset

        Args:
            data_path: 预处理数据文件路径(.pt)
            codebook_size: codebook大小
        """
        # 加载完整数据
        processed_data = torch.load(data_path, map_location="cpu")

        self.x = processed_data["x"]  # 节点特征
        self.embed_ind = processed_data["embed_ind"]
        self.slices = processed_data["slices"]
        self.codebook = processed_data["codebook"]
        self.generate_config = processed_data["generation_config"]
        self.is_reduced = processed_data["is_reduced"]

        # 校验codebook尺寸
        if (
            not self.is_reduced
            and codebook_size
            and len(self.codebook) != codebook_size
        ):
            raise ValueError(
                f"Mismatch codebook size, expect {codebook_size} got {len(self.codebook)}"
            )

        self.start_token = len(self.codebook)  # 用于序列开始的特殊token
        self.end_token = self.start_token + 1

        self.random_perm = random_perm

        # self.sequences = []
        # for sample in self.data_samples:
        #     # 转成CPU上的整数张量，并确保形状为(seq_len,)
        #     indices = sample["embed_ind"].cpu().squeeze().long()
        #     assert indices.ndim == 1, "Embed indices应该是一维序列"
        #     extended_seq = torch.cat([
        #         torch.tensor([self.start_token]),
        #         indices,
        #         torch.tensor([self.end_token])
        #     ])
        #     self.sequences.append(extended_seq)

        self.cum_slices = torch.cat(
            [torch.tensor([0]), self.slices.cumsum(dim=0)], dim=0
        )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """返回一个样本用于训练生成模型

        Return:
            dict: 包含输入序列和标签的字典（适用于类似GPT的自回归训练）
        """
        # indices = self.sequences[idx]
        # if not self.random_perm:
        #     return {
        #         "input_ids": indices[:-1],    # 输入给模型的序列部分
        #         "labels": indices[1:],        # 需要预测的下一个token
        #         "origin_length": len(indices)  # 原始序列长度（可用于padding恢复）
        #     }
        # else:
        #     perm = torch.randperm(len(indices) - 2)
        #     indices[1:-1] = indices[1:-1][perm]
        #     return {
        #         "input_ids": indices[:-1],    # 输入给模型的序列部分
        #         "labels": indices[1:],        # 需要预测的下一个token
        #         "origin_length": len(indices)  # 原始序列长度（可用于padding恢复）
        #     }
        # 计算节点范围
        start = self.cum_slices[idx]
        end = start + self.slices[idx]

        # 获取原始数据
        indices = self.embed_ind[start:end]

        # 添加特殊token
        seq = torch.cat(
            [torch.tensor([self.start_token]), indices, torch.tensor([self.end_token])]
        )

        # 随机置换（如果启用）
        if self.random_perm and len(seq) > 2:
            perm = torch.randperm(len(seq) - 2) + 1
            seq[1:-1] = seq[perm]

        # TODO BUGS HERE,
        return {
            "input_ids": seq[:-1],
            "labels": seq[1:],
            "origin_length": len(seq),
            "origin_seq": seq,
        }

    def get_codebook(self):
        """获取关联的codebook张量"""
        return self.codebook


class QM9_VQ_Dataset_Diffusion(Dataset):
    def __init__(self, data_path, codebook_size=None, random_perm=False):
        """加载预处理的VQ-VAE数据并初始化Dataset

        Args:
            data_path: 预处理数据文件路径(.pt)
            codebook_size: codebook大小
        """
        # 加载完整数据
        processed_data = torch.load(data_path, map_location="cpu")

        self.x = processed_data["x"]  # 节点特征
        self.embed_ind = processed_data["embed_ind"]
        self.slices = processed_data["slices"]
        self.codebook = processed_data["codebook"]
        self.generate_config = processed_data["generation_config"]
        self.is_reduced = processed_data["is_reduced"]

        # 校验codebook尺寸
        if (
            not self.is_reduced
            and codebook_size
            and len(self.codebook) != codebook_size
        ):
            raise ValueError(
                f"Mismatch codebook size, expect {codebook_size} got {len(self.codebook)}"
            )

        self.start_token = len(self.codebook)  # 用于序列开始的特殊token
        self.end_token = self.start_token + 1

        self.random_perm = random_perm

        # self.sequences = []
        # for sample in self.data_samples:
        #     # 转成CPU上的整数张量，并确保形状为(seq_len,)
        #     indices = sample["embed_ind"].cpu().squeeze().long()
        #     assert indices.ndim == 1, "Embed indices应该是一维序列"
        #     extended_seq = torch.cat([
        #         torch.tensor([self.start_token]),
        #         indices,
        #         torch.tensor([self.end_token])
        #     ])
        #     self.sequences.append(extended_seq)

        self.cum_slices = torch.cat(
            [torch.tensor([0]), self.slices.cumsum(dim=0)], dim=0
        )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """返回一个样本用于训练生成模型

        Return:
            dict: 包含输入序列和标签的字典（适用于类似GPT的自回归训练）
        """
        # indices = self.sequences[idx]
        # if not self.random_perm:
        #     return {
        #         "input_ids": indices[:-1],    # 输入给模型的序列部分
        #         "labels": indices[1:],        # 需要预测的下一个token
        #         "origin_length": len(indices)  # 原始序列长度（可用于padding恢复）
        #     }
        # else:
        #     perm = torch.randperm(len(indices) - 2)
        #     indices[1:-1] = indices[1:-1][perm]
        #     return {
        #         "input_ids": indices[:-1],    # 输入给模型的序列部分
        #         "labels": indices[1:],        # 需要预测的下一个token
        #         "origin_length": len(indices)  # 原始序列长度（可用于padding恢复）
        #     }
        # 计算节点范围
        start = self.cum_slices[idx]
        end = start + self.slices[idx]

        # 获取原始数据
        seq = self.embed_ind[start:end]

        if self.random_perm and len(seq) > 1:
            perm = torch.randperm(len(seq))
            seq = seq[perm]

        return seq

    def get_codebook(self):
        """获取关联的codebook张量"""
        return self.codebook


class QM9_VQ_Dataset_Regression(Dataset):
    def __init__(self, data_path, codebook_size=None, random_perm=False):
        """加载预处理的VQ-VAE数据并初始化Dataset

        Args:
            data_path: 预处理数据文件路径(.pt)
            codebook_size: codebook大小
        """
        # 加载完整数据
        processed_data = torch.load(data_path, map_location="cpu")

        self.x = processed_data["x"]  # 节点特征
        self.embed_ind = processed_data["embed_ind"]
        self.slices = processed_data["slices"]
        self.codebook = processed_data["codebook"]
        self.generate_config = processed_data["generation_config"]
        self.is_reduced = processed_data["is_reduced"]
        self.y = processed_data["y"]

        # 校验codebook尺寸
        if (
            not self.is_reduced
            and codebook_size
            and len(self.codebook) != codebook_size
        ):
            raise ValueError(
                f"Mismatch codebook size, expect {codebook_size} got {len(self.codebook)}"
            )

        self.start_token = len(self.codebook)  # 用于序列开始的特殊token
        self.end_token = self.start_token + 1

        self.random_perm = random_perm

        self.cum_slices = torch.cat(
            [torch.tensor([0]), self.slices.cumsum(dim=0)], dim=0
        )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """返回一个样本用于训练生成模型

        Return:
            dict: 包含输入序列和标签的字典（适用于类似GPT的自回归训练）
        """
        start = self.cum_slices[idx]
        end = start + self.slices[idx]

        # 获取原始数据
        seq = self.embed_ind[start:end]

        if self.random_perm and len(seq) > 1:
            perm = torch.randperm(len(seq))
            seq = seq[perm]

        y = self.y[idx : idx + 1]

        return seq, y

    def get_codebook(self):
        """获取关联的codebook张量"""
        return self.codebook

    # collate_fn
    @staticmethod
    def collate_fn(batch):
        seqs, ys = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
        batch_vector = torch.arange(len(seqs)).repeat_interleave(lengths)
        seqs_concat = torch.cat(seqs, dim=0)
        y = torch.cat(ys, dim=0)

        edge_indices_list = []
        current_node_offset = 0  # 用于在批次中全局偏移节点索引

        for num_nodes_in_graph in lengths:
            if num_nodes_in_graph <= 1:
                # 节点数小于等于1的图没有边
                edge_indices_list.append(torch.empty((2, 0), dtype=torch.long))
            else:
                # 为当前图生成局部边索引 (节点从 0 到 num_nodes_in_graph-1)
                # 方法：直接创建所有可能的 (源,目标) 对，然后移除自环

                # 创建表示图中所有节点的张量: tensor([0, 1, ..., num_nodes_in_graph-1])
                local_node_indices = torch.arange(num_nodes_in_graph, dtype=torch.long)

                # src_local: tensor([0,0,0, 1,1,1, 2,2,2]) (假设num_nodes_in_graph=3)
                # dst_local: tensor([0,1,2, 0,1,2, 0,1,2]) (假设num_nodes_in_graph=3)
                src_local = (
                    local_node_indices.view(-1, 1)
                    .repeat(1, num_nodes_in_graph)
                    .reshape(-1)
                )
                dst_local = local_node_indices.repeat(num_nodes_in_graph)

                # 移除自环 (源节点不等于目标节点)
                mask = src_local != dst_local
                local_src_edges = src_local[mask]
                local_dst_edges = dst_local[mask]

                local_edge_index = torch.stack(
                    [local_src_edges, local_dst_edges], dim=0
                )

                # 将局部边索引加上当前节点偏移量，得到全局边索引
                global_edge_index = local_edge_index + current_node_offset
                edge_indices_list.append(global_edge_index)

            current_node_offset += num_nodes_in_graph
        batched_edge_index = torch.cat(edge_indices_list, dim=1)

        return {
            "seq": seqs_concat,
            "batch": batch_vector,
            "y": y,
            "edge_index": batched_edge_index,
            "lengths": lengths,
        }


if __name__ == "__main__":
    # 测试代码
    dataset = QM9_VQ_Dataset_Regression(
        "runs/zinc-vqvae-egt-3/upbeat-meadow-31/zinc_vqdataset_reduced.pt",
        codebook_size=512,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=dataset.collate_fn
    )

    for batch in dataloader:
        print(batch)
        break
