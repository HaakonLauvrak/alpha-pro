from play.play import PLAY

models = ["hex_4_1ep_100searches", "hex_4_3ep_100searches", "hex_4_5ep_100searches", "hex_4_7ep_100searches", "hex_4_9ep_100searches"]
          

if __name__ == "__main__":

    PLAY().search_and_train_hex()
    # PLAY().generate_training_data_hex()
    # PLAY().train_hex_actor()
    # PLAY().search_and_train_nim()
    # PLAY().play_hex_mcts()
    # PLAY().play_nim_mcts()
    # PLAY().topp(models, 20)


    