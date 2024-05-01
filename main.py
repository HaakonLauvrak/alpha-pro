from play.play import PLAY

models = ["hex_4_1ep_100searches", "hex_4_26ep_100searches", "hex_4_51ep_100searches"]
          

if __name__ == "__main__":
    # PLAY().search_and_train_hex()
    # PLAY().generate_training_data_hex()
    # PLAY().train_hex_actor()
    PLAY().search_and_train_nim()
    # PLAY().play_hex_mcts()
    # PLAY().play_nim_mcts()
    # PLAY().topp(models, 50)


    