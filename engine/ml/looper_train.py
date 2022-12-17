import numpy as np
import torch
import random
from tqdm import tqdm

import model_pytorch_BIG as model_pytorch

import wandb

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
    parser.add_argument("--old-optim-state-path", metavar="PATH", help="Path for input AdamW state.")
    parser.add_argument("--new-path", metavar="PATH", required=True, help="Path for output network.")
    parser.add_argument("--new-optim-state-path", metavar="PATH", help="Path for output AdamW state.")
    parser.add_argument("--steps", metavar="COUNT", type=int, default=1000, help="Training steps.")
    parser.add_argument("--minibatch-size", metavar="COUNT", type=int, default=1024, help="Minibatch size.")
    parser.add_argument("--learning-rate", metavar="LR", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--data-file", metavar="PATH", type=str, help="Saved .npz file of features/targets.")
    parser.add_argument("--save-every", metavar="STEPS", default=0, type=int, help="Save a model every n steps.")
    parser.add_argument("--project-name", metavar="NAME", type=str, help="The project string to pass to wandb")
    args = parser.parse_args()
    wandb.init(project=args.project_name, name=args.new_path)
    print("Arguments:", args)

    import make_dataset

    if args.data_file:
        print("Loading data from:", args.data_file)
        dataset = make_dataset.Dataset.load(args.data_file)
    else:
        dataset = make_dataset.collect_data(args.games)

    print("Converting tensors")
    train_features = torch.tensor(dataset.features)
    train_policy_indices = torch.tensor(dataset.policy_indices)
    train_policy_probs = torch.tensor(dataset.policy_probs)
    #train_policy = torch.tensor(train_policy.reshape((-1, 64 * 64)))
    train_wdl_index = torch.tensor(dataset.wdl_index.astype(np.int64))
    train_mcts_root_value = torch.tensor(dataset.mcts_root_value)

    print("Got data:", train_features.shape, train_policy_indices.shape, train_policy_probs.shape, train_wdl_index.shape, train_mcts_root_value.shape)

    wandb.config.update(args)
    wandb.config.update({
        #"old_path": args.old_path,
        #"new_path": args.new_path,
        #"steps": args.steps,
        #"minibatch_size": args.minibatch_size,
        #"learning_rate": args.learning_rate,
        "datapoint_count": len(train_features),
    })

    def make_batch(batch_size):
        indices = np.random.randint(0, len(train_features), size=batch_size)
        return (
            # We convert the features array after indexing, to reduce memory consumption.
            train_features[indices].cuda().to(torch.float32),
            train_policy[indices].cuda().to(torch.float32),
            train_value[indices].cuda(),
        )

    def make_batch(batch_size):
        MOVE_COUNT = 64 * 64
        indices = np.random.randint(0, len(train_features), size=batch_size)
        # We now unpack our sparse representation of the policy.
        policy_block = torch.zeros(batch_size * MOVE_COUNT, dtype=torch.float32)
        # Both idx and probs have shape (batch, policy_truncation).
        orig_idx = train_policy_indices[indices]
        probs = train_policy_probs[indices]
        assert orig_idx.shape == probs.shape == (batch_size, 32)
        # Remap -1 in orig_idx to 0. I think this is safe, as from=0 to=0 should never have prob?
        idx = torch.where(orig_idx == -1, torch.tensor(0, dtype=torch.int16), orig_idx)
        policy_block.index_add_(
            dim=0,
            index=(idx + MOVE_COUNT * torch.arange(batch_size).unsqueeze(-1)).flatten(),
            source=probs.flatten(),
        )
        policy_block = policy_block.reshape((batch_size, MOVE_COUNT))
        row_normalization = torch.sum(policy_block, dim=1)
        policy_block /= row_normalization.unsqueeze(-1)
        #print("Unpacked into:", policy_block.shape)
        bs, trunc = orig_idx.shape
        return (
            # We convert the features array after indexing, to reduce memory consumption.
            train_features[indices].cuda().to(torch.float32),
            policy_block.cuda(),
            train_wdl_index[indices].cuda(),
            train_mcts_root_value[indices].cuda(),
            #train_value[indices].cuda(),
        )

    print("Attempting to create model.")
    model = model_pytorch.Network()
    model.cuda()
    print("Parameter count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    cross_en = torch.nn.CrossEntropyLoss()
    mse_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    policy_loss_ewma = EWMA()
    wdl_loss_ewma = EWMA()
    mcts_root_value_prediction_loss_ewma = EWMA()


    # Load up the model
    if args.old_path is not None:
        print("Loading from:", args.old_path)
        model.load_state_dict(torch.load(args.old_path))
    if args.old_optim_state_path is not None:
        print("Loading optimizer state from:", args.old_optim_state_path)
        optimizer.load_state_dict(torch.load(args.old_optim_state_path))

    def save_model():
        print("Saving to:", args.new_path)
        torch.save(model.state_dict(), args.new_path)
        if args.new_optim_state_path is not None:
            print("Saving optimizer state to:", args.new_optim_state_path)
            torch.save(optimizer.state_dict(), args.new_optim_state_path)

    for i in range(args.steps):
        optimizer.zero_grad()
        features, target_policy, target_wdl_index, target_mcts_root_value = make_batch(args.minibatch_size)
        policy_output, wdl_output, mcts_root_value_prediction = model(features)
        policy_loss = cross_en(policy_output, target_policy)
        wdl_loss = cross_en(wdl_output, target_wdl_index[:, 0])
        mcts_root_value_prediction_loss = mse_func(mcts_root_value_prediction, target_mcts_root_value)
        loss = policy_loss + wdl_loss + mcts_root_value_prediction_loss
        loss.backward()
        wandb.log({
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "wdl_loss": wdl_loss.item(),
            "mcts_root_value_prediction_loss": mcts_root_value_prediction_loss.item(),
            "grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0),
        })
        optimizer.step()
        policy_loss_ewma.apply(policy_loss.item())
        wdl_loss_ewma.apply(wdl_loss.item())
        mcts_root_value_prediction_loss_ewma.apply(mcts_root_value_prediction_loss.item())

        if i % 100 == 0:
            print("[%7i] loss = %.4f (policy = %.4f  wdl = %0.4f  root-val = %0.4f)" % (
                i,
                policy_loss_ewma.value + wdl_loss_ewma.value + mcts_root_value_prediction_loss_ewma.value,
                policy_loss_ewma.value,
                wdl_loss_ewma.value,
                mcts_root_value_prediction_loss_ewma.value,
            ))
        if args.save_every and i % args.save_every == 0:
            save_model()

    save_model()
