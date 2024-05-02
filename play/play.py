from actor.replay_buffer import REPLAY_BUFFER
from game_logic.nim_state_manager import NIM_STATE_MANAGER
from gui.hex_board import HEX_BOARD
from gui.hex_game_visualizer import HEX_BOARD_VISUALIZER
from game_logic.hex_state_manager import HEX_STATE_MANAGER
from gui.nim_board import NIM_BOARD
from mcts.mcts import MonteCarloTreeSearch
import config.config as config
from actor.anet import ANET
from tournaments.topp import Tournament
import matplotlib.pyplot as plt

class PLAY():

    def __init__(self) -> None:
        pass

    def play_nim_mcts(self): 
        """
        Play a game of NIM using MCTS without ANET
        """
        board = NIM_BOARD()
        board.set_state(config.nim_N)
        state = [board, 1]
        sm = NIM_STATE_MANAGER()
        ann = ANET("nim")
        print("Starting nim game")
        print(f"N = {config.nim_N}, K = {config.nim_K}")
        mcts = MonteCarloTreeSearch(state, ann, sm)
        while not sm.isGameOver(state):
            bestAction = mcts.search(random_move=True)
            sm.makeMove(bestAction, state)
            mcts.update_root(bestAction)
            print(f"Player {-state[1]} took: ", bestAction)
        print(f"Player {-state[1]} wins!")


    def play_hex_mcts(self):
        """
        Play a game of HEX using MCTS without ANET
        """
        sm = HEX_STATE_MANAGER()
        anet = ANET("training_net")
        state = [HEX_BOARD(config.board_size), 1]
        mcts = MonteCarloTreeSearch(state, anet, sm)
        #FOR TESTING
        replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
        visualizer = HEX_BOARD_VISUALIZER(config.board_size)
        visualizer.update_board(state[0].get_cells())
        while not sm.isGameOver(state):
            bestAction = mcts.search(random_move=True)
            sm.makeMove(bestAction, state)
            x_train, y_train = mcts.extract_training_data()
            replay_buffer.add(x_train, y_train)
            mcts.update_root(bestAction)
            visualizer.update_board(state[0].get_cells())
            print(f"Player {state[1]} took: ", bestAction)
        print(f"Player {state[1] * -1} wins!")
        visualizer.close()

    def search_and_train_hex(self):
        """
        Play a game of HEX using MCTS with ANET
        """
        sm = HEX_STATE_MANAGER(0)
        anet = ANET("0 games")
        anet.save_model(0)
        state = [HEX_BOARD(config.board_size), 1]
        mcts = MonteCarloTreeSearch(state, anet, sm)
        
        visualizer = HEX_BOARD_VISUALIZER(config.board_size)
        visualizer.update_board(state[0].get_cells())
        replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
        
        for i in range(config.num_episodes + 1):
            print(i)
            while not sm.isGameOver(state):
                bestAction = mcts.search(random_move=True)
                print(bestAction)
                sm.makeMove(bestAction, state)
                x_train, y_train = mcts.extract_training_data()
                replay_buffer.add(x_train, y_train)
                mcts.update_root(bestAction)
                visualizer.update_board(state[0].get_cells())
            
            sm.increment_episode() #must be done here to avoid incrementing episode when simulations reach end state.
            state = [HEX_BOARD(config.board_size), 1 if i % 2 == 0 else -1]
            sm.setState(state)
            mcts = MonteCarloTreeSearch(state, anet, sm)
           
            if (i + 1) % (config.num_episodes // config.M) == 0:
                anet = ANET(f"{i + 1} games")
                anet.train_model(replay_buffer.get_all())
                anet.save_model(i+1)

        print("Done")
    

    def search_and_train_nim(self):
        """
        Play a game of Nim using MCTS with ANET
        """
        sm = NIM_STATE_MANAGER()
        anet = ANET("nim")
        board = NIM_BOARD()
        board.set_state(config.nim_N)
        state = [board, 1]
        mcts = MonteCarloTreeSearch(state, anet, sm)
        rbuf = REPLAY_BUFFER(config.replay_buffer_size)
        acc = []
        for i in range(config.num_episodes):
            print("Game ", i)
            while not sm.isGameOver(state):
                mcts.search()
                bestAction = mcts.best_action()
                print("Stones left: ", state[0].get_state())
                sm.makeMove(bestAction, state)
                print(f"Player {-state[1]} took: ", bestAction)
                mcts.update_root(bestAction)
            print(f"Player {-state[1]} wins!")
            sm.increment_episode()
            x_train, y_train = mcts.extract_training_data()
            rbuf.add(x_train, y_train)
            acc.append(anet.train_model(rbuf.get_all()))
            board = NIM_BOARD()
            board.set_state(config.nim_N)
            state = [board, 1]
            sm.setState(state)
            mcts = MonteCarloTreeSearch(state, anet, sm)
        # print(acc)
        print("Done")

    def topp(self, models: list):
        """
        Takes a list of four models and plays a round robin tournament between them
        """
        rounds = config.G
        players = []

        for i in range(len(models)):
            anet = ANET(f"Player{i + 1}")
            model = anet.load_model_by_name(models[i])
            anet.set_model(model)
            players.append(anet)

        sm = HEX_STATE_MANAGER(epsilon=0)
        tournament = Tournament(players, sm, rounds, "hex")
        results = tournament.play_tournament()
        main_dict = {}
        for dict in results:
            for key in dict:
                if key in main_dict:
                    main_dict[key] += dict[key]
                else:
                    main_dict[key] = dict[key]
            print(dict)
        print(main_dict)

        #plot main dict as bar chart
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(main_dict.keys(), main_dict.values())
        plt.show()

    def generate_training_data_hex(self):
        """
        Generates training data for HEX using MCTS without anet. Saves data to file.
        """
        sm = HEX_STATE_MANAGER()
        anet = ANET("training_net")
        state = [HEX_BOARD(config.board_size), 1]
        replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
    
        game_counter = 0
        for i in range(config.num_episodes):
            if replay_buffer.get_size() == config.replay_buffer_size:
                replay_buffer.save(f"training_data/hex{config.board_size}_training_data_{game_counter}games_AAA")
                replay_buffer = REPLAY_BUFFER(config.replay_buffer_size)
            if i % 2 == 0:
                state = [HEX_BOARD(config.board_size), -1]
            else:
                state = [HEX_BOARD(config.board_size), 1]
            
            mcts = MonteCarloTreeSearch(state, anet, sm)
            while not sm.isGameOver(state):
                mcts.search(random_move=True)
                bestAction = mcts.best_action()
                sm.makeMove(bestAction, state)
                x_train, y_train = mcts.extract_training_data()
                replay_buffer.add(x_train, y_train)
                mcts.update_root(bestAction)
            game_counter += 1
            print(f"Game {game_counter} finished")
            print(f"Replay buffer size: {replay_buffer.get_size()}")
            if i == config.num_episodes // 5 or i == config.num_episodes // 3 or i == config.num_episodes // 2 or i == 2 * config.num_episodes // 3:
                replay_buffer.save(f"training_data/hex{config.board_size}_training_data_{game_counter}games_AAA")

        replay_buffer.save(f"training_data/hex{config.board_size}_training_data_{game_counter}games_AAA")

    def train_hex_actor(self):
        """
        Trains the actor network for HEX using previously generated training data.
        """

        anet = ANET("training_net_0")
        anet.save_model(0)

        replay_buffer = REPLAY_BUFFER(10000000)
 
        replay_buffer.load(f"training_data/hex4_training_data_100games_AA.npz")
        replay_buffer.load(f"training_data/hex4_training_data_100games_A.npz")
        anet = ANET("training_net_200")
        anet.train_model(replay_buffer.get_all())
        anet.save_model(200)

        replay_buffer.load(f"training_data/hex4_training_data_1001games_AAA.npz")
        anet = ANET("training_net_1001")
        anet.train_model(replay_buffer.get_all())
        anet.save_model(1001)

        replay_buffer.load(f"training_data/hex4_training_data_2501games_AAA.npz")
        anet = ANET("training_net_2501")
        anet.train_model(replay_buffer.get_all())
        anet.save_model(2501)
        
        # acc = []
        # loss = []
        # mae = []

        # training_score = anet.train_model(replay_buffer.get_all())
        # loss.append(training_score[0])
        # acc.append(training_score[1])
        # mae.append(training_score[2])

        # anet.save_model(config.num_episodes)

        # # Print avg accuracy, loss and mae
        # print("Avg accuracy: ", sum(acc) / len(acc))
        # print("Avg loss: ", sum(loss) / len(loss))
        # print("Avg mae: ", sum(mae) / len(mae))
        # print("Learning rate: ", config.learning_rate)
        # print("Epochs: ", config.epochs)
        # print("Batch size: ", config.batch_size)
        # print("Dimensions conv: ", config.dimensions_conv)
        # print("Dimensions dense: ", config.dimensions_dense)
        # print("Activation: ", config.activation)
        # print("Optimizer: ", config.optimizer)

        # #Save results to file
        # with open("training_results_hex.txt", "a") as f:
        #     f.write("Avg accuracy: " + str(sum(acc) / len(acc)) + "\n")
        #     f.write("Avg loss: " + str(sum(loss) / len(loss)) + "\n")
        #     f.write("Avg mae: " + str(sum(mae) / len(mae)) + "\n")
        #     f.write("Learning rate: " + str(config.learning_rate) + "\n")
        #     f.write("Epochs: " + str(config.epochs) + "\n")
        #     f.write("Batch size: " + str(config.batch_size) + "\n")
        #     f.write("Dimensions conv: " + str(config.dimensions_conv) + "\n")
        #     f.write("Dimensions dense: " + str(config.dimensions_dense) + "\n")
        #     f.write("Activation: " + str(config.activation) + "\n")
        #     f.write("Optimizer: " + str(config.optimizer) + "\n" + "\n")

            