import torch


class Trainor:
    def __init__(self, data, device, block_size):
        self.data  = data 
        self.chars = sorted(list(set(data)))
        self.stoi = {s:i for i,s in enumerate(self.chars)}
        self.itos = {s:i for i,s in self.stoi.items()}
        self.block_size = block_size
        self.seed = 1337
        self.batch_size = 32
        self.eval_interval = 100
        self.device = device
        self.eval_iters = 200
        
        torch.manual_seed(self.seed)
    
    def encode(self):
        encode = lambda s: [self.stoi[c] for c in s]
        self.encoded = torch.tensor(encode(self.data), dtype=torch.long)
        return self
        
    def decode(self, data):
        decode = lambda l: ''.join([self.itos[i] for i in l])
        return decode(data)

    def split_data(self):
        self.n = int(0.9*len(self.data))
        self.train_split = self.encoded[:self.n]
        self.val_split = self.encoded[self.n:]
        return self

    def get_batch(self, split):
        data = self.train_split if split == 'train' else self.val_split
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model):
        out = {}
        model.eval()

        for split in ['train_split', 'val_split']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()

            out[split] = losses.mean()
        model.train()
        return out

    def train(self, model, num_of_steps):
        for step in range(num_of_steps):

            if step % self.eval_interval == 0:
                losses = self.estimate_loss(model)
                print(f"step={step} -- train loss: {losses['train_split']:.4f}, val loss {losses['val_split']:.4f}")

            xb, yb = self.get_batch('train')

            logits, loss = model(xb, yb)
            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.optimizer.step()


        print(f'{loss=}')
            
        return self

