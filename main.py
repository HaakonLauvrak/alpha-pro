from play.play import PLAY

models = ["hex_4_1ep_1000searches", "hex_4_6ep_1000searches", "hex_4_11ep_1000searches"]
          

if __name__ == "__main__":
    PLAY().search_and_train_hex()
    # PLAY().generate_training_data_hex()
    # PLAY().train_hex_actor()
    # PLAY().search_and_train_nim()
    # PLAY().play_hex_mcts()
    # PLAY().play_nim_mcts()
    # PLAY().topp(models, 50)


    