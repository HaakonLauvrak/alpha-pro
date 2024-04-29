from play.play import PLAY

models = ["hex_7_1ep_5000searches", "hex_7_5ep_5000searches"]
          

if __name__ == "__main__":

    # PLAY().search_and_train_hex()
    # PLAY().generate_training_data_hex()
    PLAY().train_hex_actor()
    # PLAY().search_and_train_nim()
    # PLAY().play_hex_mcts()
    # PLAY().play_nim_mcts()
    # PLAY().topp(models, 10)


    