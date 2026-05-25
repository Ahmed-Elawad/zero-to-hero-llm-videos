from train import Trainor
from model import BigramModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

print(f"pytorch version {torch.__version__}")
print(f"GPU Available {torch.cuda.is_available()}")

data = open('./input.txt').read()

block_size = 8
num_embeddings = 32

trainor = Trainor(data, device, block_size).encode().split_data()
vocab_size = len(trainor.chars)
model = BigramModel(
    vocab_size, 
    num_embeddings, 
    block_size, 
    device
)
model = model.to(device)

num_of_steps = 1000
trainor.train(model, num_of_steps)

idx = torch.zeros((1, 1), dtype=torch.long, device=device)

print(
    trainor.decode(
        model.generate(
            idx=idx,
            max_new_tokens=400
        )[0].tolist()
    )
)