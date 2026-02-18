import torch
import torch.utils.data as data_utils

class PreTokenizedFoodDataset(data_utils.Dataset):
    def __init__(self, input_ids_list, targets):
        self.input_ids = input_ids_list
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        
        mask = (ids != 0).long()
        
        label = torch.tensor(self.targets[idx], dtype=torch.float)

        return {
            'input_ids': ids,
            'attention_mask': mask,
            'labels': label
        }