from play.play import PLAY

models = ["hex_4_0ep_1000searches", "hex_4_200ep_1000searches", "hex_4_1001ep_1000searches", "hex_4_2501ep_1000searches"]

if __name__ == "__main__":
    # PLAY().search_and_train_hex()
    # PLAY().generate_training_data_hex()
    # PLAY().train_hex_actor()
    # PLAY().search_and_train_nim()
    # PLAY().play_hex_mcts()
    # PLAY().play_nim_mcts()
    
    """ Play a TOPP between cached models"""
    # PLAY().topp(models)

    """Create models and run a TOPP between them"""
    models = PLAY().search_and_train_hex()
    # PLAY().topp(models, names=False)