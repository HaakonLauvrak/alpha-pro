from play.play import PLAY

# models = ["hex_4_0ep_1000searches", "hex_4_9ep_1000searches", "hex_4_19ep_1000searches"]
models = ["hex_7_1000ep_100000searches"]

if __name__ == "__main__":
    # PLAY().search_and_train_hex()
    # PLAY().generate_training_data_hex()
    # PLAY().train_hex_actor()
    # PLAY().search_and_train_nim()
    # PLAY().play_hex_mcts()
    # PLAY().play_nim_mcts()
    PLAY().topp(models)