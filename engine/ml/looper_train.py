import numpy as np
import torch
from tqdm import tqdm

import model_pytorch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--games", metavar="PATH", required=True, nargs="+", help="Path to .json self-play games files.")
    parser.add_argument("--old-path", metavar="PATH", help="Path for input network.")
    parser.add_argument("--new-path", metavar="PATH", required=True, help="Path for output network.")
    parser.add_argument("--steps", metavar="COUNT", type=int, default=1000, help="Training steps.")
    parser.add_argument("--minibatch-size", metavar="COUNT", type=int, default=1024, help="Minibatch size.")
    parser.add_argument("--learning-rate", metavar="LR", type=float, default=2e-4, help="Learning rate.")
    args = parser.parse_args()
    print("Arguments:", args)

    import make_dataset
    features, policy, value = make_dataset.process_game_paths(args.games)
    features = torch.tensor(features)
    policy = torch.tensor(policy)
    value = torch.tensor(value)

    def make_batch(batch_size):
        indices = np.random.randint(0, len(features), size=batch_size)
        return (
            # We convert the features array after indexing, to reduce memory consumption.
            features[indices].cuda().to(torch.float32),
            policy[indices].cuda(),
            value[indices].cuda(),
        )

    # Perform generator pretraining.
    cross_en = torch.nn.CrossEntropyLoss()
    mse_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sm_loss_ewma = EWMA()
    tanh_loss_ewma = EWMA()

    # Load up the model
    if args.old_path is not None:
        print("Loading from:", args.old_path)
        model.load_state_dict(torch.load(args.old_path))

    for i in range(args.steps):
        optimizer.zero_grad()
        inp, pol, val = make_batch(args.minibatch_size)
        sm_output, tanh_output = model(inp)
        # Print all shapes.
        # print("inp", inp.shape)
        # print("pol", pol.shape)
        # print("val", val.shape)
        # print("sm_output", sm_output.shape)
        # print("tanh_output", tanh_output.shape)
        sm_loss = cross_en(sm_output, pol)
        tanh_loss = mse_func(tanh_output, val)
        loss = sm_loss + tanh_loss
        loss.backward()
        optimizer.step()
        sm_loss_ewma.apply(sm_loss.item())
        tanh_loss_ewma.apply(tanh_loss.item())

        if i % 100 == 0:
            print("[%7i] loss = %.4f (policy = %.4f  value = %0.4f)" % (
                i, sm_loss_ewma.value + tanh_loss_ewma.value, sm_loss_ewma.value, tanh_loss_ewma.value,
            ))
    
    print("Saving to:", args.new_path)
    torch.save(model.state_dict(), args.new_path)

