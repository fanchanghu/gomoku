import logging
from gomoku.vpg_train_flow import VPGTrainFlow


def parse_args():
    import sys

    if len(sys.argv) == 1:
        print("No argument provided, defaulting to continue training.")
        return "continue"

    if sys.argv[1] == "init":
        print("train from scratch, no model loaded.")
        return "init"
    elif sys.argv[1] == "continue":
        print("continue training from the latest model.")
        return "continue"
    else:
        # exit
        print(f"Unknown argument: {sys.argv[1]}. Use 'init' or 'continue'.")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mode = parse_args()

    train_flow = VPGTrainFlow(mode)
    train_flow.run(max_k=10000, eval_interval=10, save_interval=100)
