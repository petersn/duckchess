import numpy as np
import torch
import random
from tqdm import tqdm

import model_pytorch

class EWMA:
    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.value = None

    def apply(self, x):
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--games", metavar="PATH", nargs="+", help="Path to .json self-play games files.")
    parser.add_argument("--old-path", metavar="PATH", help="Path for input network.")
    parser.add_argument("--new-path", metavar="PATH", required=True, help="Path for output network.")
    parser.add_argument("--steps", metavar="COUNT", type=int, default=1000, help="Training steps.")
    parser.add_argument("--minibatch-size", metavar="COUNT", type=int, default=1024, help="Minibatch size.")
    parser.add_argument("--learning-rate", metavar="LR", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--data-file", metavar="PATH", type=str, help="Saved .npz file of features/targets.")
    parser.add_argument("--save-every", metavar="STEPS", default=0, type=int, help="Save a model every n steps.")
    args = parser.parse_args()
    print("Arguments:", args)

    import make_dataset

    if args.data_file:
        print("Loading data from:", args.data_file)
        data = np.load(args.data_file)
        train_features = data["features"]
        train_policy = data["policy"]
        train_value = data["value"]
    else:
        train_features, train_policy, train_value = \
            make_dataset.process_game_paths(args.games)

    print("Converting tensors")
    train_features = torch.tensor(train_features)
    train_policy = torch.tensor(train_policy.reshape((-1, 64 * 64)))
    train_value = torch.tensor(train_value)

    print("Got data:", train_features.shape, train_policy.shape, train_value.shape)

    def make_batch(batch_size):
        indices = np.random.randint(0, len(train_features), size=batch_size)
        return (
            # We convert the features array after indexing, to reduce memory consumption.
            train_features[indices].cuda().to(torch.float32),
            train_policy[indices].cuda().to(torch.float32),
            train_value[indices].cuda(),
        )

    model = model_pytorch.Network()
    model.cuda()
    print("Parameter count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Load up the model
    if args.old_path is not None:
        print("Loading from:", args.old_path)
        model.load_state_dict(torch.load(args.old_path))

    # Perform generator pretraining.
    cross_en = torch.nn.CrossEntropyLoss()
    mse_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    policy_loss_ewma = EWMA()
    value_loss_ewma = EWMA()

    for i in range(args.steps):
        optimizer.zero_grad()
        features, target_policy, target_value = make_batch(args.minibatch_size)
        policy_output, value_output = model(features)
        policy_loss = cross_en(policy_output, target_policy)
        value_loss = mse_func(value_output, target_value)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        policy_loss_ewma.apply(policy_loss.item())
        value_loss_ewma.apply(value_loss.item())

        if i % 100 == 0:
            print("[%7i] loss = %.4f (policy = %.4f  value = %0.4f)" % (
                i, policy_loss_ewma.value + value_loss_ewma.value, policy_loss_ewma.value, value_loss_ewma.value,
            ))
        if args.save_every and i % args.save_every == 0:
            print("Saving to:", args.new_path)
            torch.save(model.state_dict(), args.new_path)

    print("Saving to:", args.new_path)
    torch.save(model.state_dict(), args.new_path)
