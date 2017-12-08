#! /usr/bin/env python

"""
@authors: Brian Hutchinson (Brian.Hutchinson@wwu.edu)

This script trains and evaluates a convolutional neural network for
digit recognition on MNIST.

For usage, run with the -h flag.

Example command:

python lab4_demo.py data/train.csv.gz data/dev.csv.gz

"""

import tensorflow as tf
import argparse
import sys
import numpy as np
import mines
import mines_viewer
from time import sleep
import pygame
from pygame import *

def get_game_state(game):
    gw, gh = game.getSize()
    vec = []
    for gx in range(gw):
        col = []
        for gy in range(gh):
            val = 0 # unrevealed cell value
            cell = game.cell[gx][gy]
            if cell.revealed:
                val = 1.0+game.surrounding((gx,gy))
            col.append(val/9.0) # map values to 0.0-1.0
        vec.append(col)
    return np.asanyarray(vec)

cached_inputs = None

def get_possible_score_array(game):
    gw,gh = game.getSize()
    scores = []
    size = gw*gh
    global cached_inputs
    if cached_inputs is None or len(cached_inputs) != size:
        cached_inputs = []
        for i in range(size):
            cached_inputs.append([0]*i + [1] + [0]*(size-i-1))
    for test_in in cached_inputs:
        score = score_output(test_in, game, True)
        scores.append(score)
    return np.array(scores)

def get_labels(game, dontAlter = True):
    gw, gh = game.getSize()
    vec = []
    for gx in range(gw):
        col = []
        for gy in range(gh):
            cell = game.cell[gx][gy]
            adj_arr = game.adjacentPositions((gx,gy))
            #revealed_neighbor = False
            #neigh_count = 0
            if cell.mine:
                # danger or potential danger
                val = 0
            else:
                # its a safe place to click
                val = 1
            
            class_count = 2
            # make it a one hot
            col.append([0]*val+[1]+[0]*(class_count-val-1))
        vec.append(col) # map values to 0.0-1.0
    return np.asanyarray(vec)


def conv2d(x,k,L,Lprime,S,P,f):
    """
    Creates the weights for a 2d convolution layer and adds the 
    corresponding convolutional layer to the graph.

    :param x: (4d tensor) tensor of inputs with dim (MB x W x W x L)
    :param k: (integer) the receptive fields will be k x k, spatially
    :param L: (integer) the input channels
    :param Lprime: (integer) the number of kernels to apply
    :param S: (integer) the stride (will be used in both spatial dims)
    :param P: (string) either "SAME" or "VALID" (specifies padding strategy)
    :param f: (string) the hidden activation (relu, tanh or linear)

    :return: (4d tensor) the result of the convolutional layer
                         (will be MB x Wprime x Wprime x Lprime)
    """

    # convolution weights (a k x k x L x Lprime 4d tensor)
    W = tf.get_variable(name="W",\
                        shape=(k,k,L,Lprime),\
                        dtype=tf.float32,\
                        initializer=tf.glorot_uniform_initializer()) 

    # pick activation and modify bias constant, if needed
    b_const = 0.0
    if f == "relu":
        b_const = 0.1
        act = tf.nn.relu
    elif f == "tanh":
        act = tf.nn.tanh
    elif f == "identity":
        act = tf.identity
    else:
        sys.exit("Error: Invalid f (%s)" % f)

    # bias weights (a Lprime dim vector)
    b = tf.get_variable(name="b",
                        shape=(Lprime),
                        initializer=tf.constant_initializer(b_const))

    # tf.nn.conv2d does the heavy lifting for us
    z = tf.nn.conv2d(x,W,strides=[1,S,S,1],padding=P)
    z = z + b # don't forget the bias!

    a = act(z)

    return a

def max_pool(x,k,S):
    """
    Adds a max pooling layer to the graph.

    :param x: (4d tensor) tensor of inputs with dim (MB x W x W x L)
    :param k: (integer) will pool over k x k spatial regions
    :param S: (integer) the stride (will be used in both spatial dims)

    :return: (4d tensor) the result of the max pooling layer
                         (will be MB x Wprime x Wprime x L)
    """

    # tf.nn.max_pool does the heavy lifting for us
    # note: using SAME padding makes the dimensionality reduction
    #       easier to compute (e.g. if k=2 and S=2, Wprime = W/2)
    return tf.nn.max_pool(x,[1,k,k,1],[1,S,S,1],padding="SAME")


def build_q_learn_model(game):
    """
    modified from q learning tensorflow example at:
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
    """
    # feed-forward part of the network used to choose actions
    grid_size = game.getSize()
    gw,gh = grid_size
    cell_count = gw*gh
    action_count = cell_count * 2
    inputs = tf.placeholder(shape=[gw,gh],dtype=tf.float32,name="x_in")
    flat_in = tf.reshape(inputs, (-1,cell_count))
    W = tf.Variable(tf.random_uniform([cell_count,action_count],0,0.01), name="W1")
    Qout = tf.matmul(flat_in,W,name="q_out")
    predict = tf.argmax(Qout,1,name="action")

    # obtain the loss by taking the sum of squares difference
    # between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[1,action_count],dtype=tf.float32, name="next_q")
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss, name="update")
    init = tf.initialize_all_variables()
    return init

def perform_action(game, action_idx):
    gw, gh = game.getSize()
    # actions are just flagging and revealing any given cell
    action_ct = (gw*gh*2)
    flag = action_idx >= gw*gh
    idx = action_idx % (gw*gh)
    gx = idx // gh
    gy = int(idx % gh)
    cell = game.cell[gx][gy]
    was_revealed = cell.revealed
    was_flagged = cell.flag
    # perform the action. method will return whether it was valid move
    action = game.flag if flag else game.reveal
    valid = action((gx,gy))
    if not valid:
        return 0
    if flag:
        if cell.mine:
            return -1 if was_flagged else 1
        else:
            return 1 if was_flagged else -1
    else:
        return -5 if cell.mine else 1
    
def train_q_learn_model(init, game):
    y = .99
    e = 0.1
    num_episodes = 2000
    step_list = []
    reward_list = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # reset environment and get first new observation
            game.reset()
            state = get_game_state(game)
            total_reward = 0
            done = False
            steps = 0
            while steps < 99 and not done:
                steps += 1
                act, Q = sess.run(fetches=["action:0", "q_out:0"],
                                  feed_dict={"x_in:0":state})
                act = act[0]
                if np.random.rand(1) < e:
                    # get random action from the space of actions we can take
                    pass
                print("action: "+str(act))
                reward = perform_action(game,act)
                # do the action                    
                next_state = get_game_state(game)
                
                Q_prime = sess.run(fetches=["q_out:0"], feed_dict={"x_in:0":next_state})
                max_prime = np.max(Q_prime)
                targ_q = Q
                targ_q[0,act] = reward + y * max_prime
                # train using predicted and target q
                _,W1 = sess.run(fetches=["update","W1"], feed_dict={"x_in:0":state,"next_q:0":targ_q})
                total_reward += reward
                state = next_state
                done = game.win or game.lose
                if done:
                    # reduce chance of random action as we train model
                    e = 1./((i/50) + 10)
                    break
            step_list.append(steps)
            reward_list.append(total_reward)
                         

def parse_all_args():
    """
    Parses arguments

    :return: the parsed arguments object
    """
    # parse arguments
    parser = argparse.ArgumentParser()


    parser.add_argument("-model", type=int, help="The model specifier (an int) [default = 1]", default=1)
    parser.add_argument("-lr",type=float,\
            help="The learning rate (a float) [default = 0.002]",default=0.003)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (an int) [default = 1]",default=256)
    parser.add_argument("-epochs",type=int,\
            help="The number of epochs to train for [default = 60]",default=100)
    parser.add_argument("-f", type = str,
            help="Hidden activation (in the set {sigmoid, tanh, relu}) [default: \"tanh\"]", default="relu")
    
    return parser.parse_args()

def gen_test_set(game, how_many):
    in_states = []
    labels = []
    for i in range(how_many):
        if i % 10 == 0:
            print("%s of %s..."%(i, how_many))
        game.setRandomState()
        state = np.array(get_game_state(game))
        #scores = np.array(get_possible_score_array(game))
        in_states.append(state)
        #score_maps.append(scores)
        labels.append(np.array(get_labels(game)))
        

    in_states = np.asanyarray(in_states)
    labels = np.asanyarray(labels)
    print("all done... (%s) (%s)"%(len(in_states), in_states.shape))
    print("with (%s) (%s)"%(len(labels), labels.shape))
    print("tyranny")
    return (in_states, labels)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_simulation(sess, game, random_start, ms_per_move):
    # time for showtime
    if random_start:
        game.setRandomState()
    else:
        game.reset()
    stuck = False
    gw, gh = game.getSize()
    cell_count = gw*gh
    # create a viewer for the simulation
    screen = pygame.display.set_mode((600,600))
    viewer = mines_viewer.MinefieldViewer(game, screen)
    
    viewer.invalidate()
    viewer.update()
    offX, offY = viewer.modelSurfaceOffset
    cw,ch = viewer.cellSize
    moves_made = 0
    while not (game.win or game.lose):
        # get the game state
        state = np.array(get_game_state(game))
        mb_x = np.reshape(state,(1,gw,gh,1)) 
        # feed the state
        [pred] = sess.run(fetches=["logits:0"],feed_dict={"x:0":mb_x})
        # find a cell that the model thinks is safe
        pred = pred[0] # MB size = 1
        safe_cell_idx = -1
        safe_cell_prob = 0.0
        cell_choice = None
        SEEN = 0
        SAFE = 1
        MINE = 0
        MYSTERY = 3
        targX, targY = (0,0)
        print("REDRAW!")
        viewer.invalidate()
        viewer.update()
        for cell_idx in range(cell_count):
            targX = cell_idx // gh
            targY = cell_idx % gh
            cell_prob = softmax(pred[cell_idx])
            
            #print("cell_prob: %s"%cell_prob)
            # cell prob has C entries adding up to 1.0
            cell_class = np.argmax(cell_prob)
            safe_prob = cell_prob[SAFE]
            #print("Cell at (%s,%s) classified as %s with probability %s"%
            #      (targX, targY, cell_class, safe_prob))
            cell = game.cell[targX][targY]
            # draw a rectangle to represent the classification
            col_safe = (0,255,0)
            col_mine = (255,0,0)
            col_visited = (0,0,255)
            col_mystery = (126, 0, 255)
            if cell_class == SAFE:
                guess_col = col_safe
            else:
                guess_col = col_mine
            """elif cell_class == SEEN:
                guess_col = col_visited
            elif cell_class == MINE:
                guess_col = col_mine
            else:
                guess_col = col_mystery"""

            if cell.revealed:
                act_col = col_visited
            elif cell.mine:
                act_col = col_mine
            else:
                act_col = col_safe
            
            rect = (offX + targX * cw, offY + targY * ch, cw//4, ch//4)
            inset = 2
            inner = (rect[0]+inset,rect[1]+inset,rect[2]-inset*2,rect[3]-inset*2)
            pygame.draw.rect(screen, guess_col, rect)
            pygame.draw.rect(screen, act_col, inner)
            
            if safe_prob > safe_cell_prob and not cell.revealed:
                safe_cell_idx = cell_idx
                safe_cell_prob = safe_prob
        moves_made += 1
        # try revealing the cell
        targX = safe_cell_idx // gh
        targY = safe_cell_idx % gh
        rect = (int(offX + (targX+0.5) * cw), offY + targY * ch, cw//4, ch//4)
        pick_col = (255, 255, 0)
        pygame.draw.rect(screen, pick_col, rect)
        game.reveal((targX, targY))
        print("chosen cell: (%s, %s)"%(targX, targY))
        pygame.event.pump()
        pygame.display.update()
        pygame.time.delay(ms_per_move)
        #input()
        if game.win or game.lose:
            # show win / lose screen
            viewer.invalidate()
            viewer.update()
            pygame.event.pump()
            pygame.time.delay(ms_per_move)
    
    print("total move made: %s"%moves_made)

def q_learn_main():

    game = mines.Minefield(mines.DifficultyFactory.EASY)
    gw,gh = game.getSize()

    test_size = 8800
    games_per_epoch = 100

    N,D = test_size, gw*gh
    """
    dev   = np.genfromtxt(args.dev,delimiter=",")
    dev_x = dev[:,1:]
    dev_y = dev[:,0]
    dev_x /= 255.0 # map from [0,255] to [0,1]"""

    # reshape dev once (train will be reshaped each MB)
    # (our graph assumes tensor-shaped input: N x W x W x L)
    C = 2

    pygame.init()
    init = build_q_learn_model(game)
    train_q_learn_model(init, game)
    
def convolution_main():
    """
    Parse arguments, build CNN, run training loop, report dev each epoch.

    """
    # parse arguments
    args = parse_all_args()
    if args.model > 5 or args.model < 1:
        print("Model must be between 1 and 5 (inclusive)")
        return

    # load and normalize data
    """train = np.genfromtxt(args.train,delimiter=",")
    train_x = train[:,1:]
    train_y = train[:,0]
    train_x /= 255.0 # map from [0,255] to [0,1]
    """
    print("generating test data...")

    game = mines.Minefield(mines.DifficultyFactory.EASY)
    gw,gh = game.getSize()

    test_size = 8800
    games_per_epoch = 100
    train_x, train_y = gen_test_set(game, test_size)
    dev_size = test_size//10
    dev_x, dev_y = gen_test_set(game, dev_size)

    N,D = test_size, gw*gh
    """
    dev   = np.genfromtxt(args.dev,delimiter=",")
    dev_x = dev[:,1:]
    dev_y = dev[:,0]
    dev_x /= 255.0 # map from [0,255] to [0,1]"""

    # reshape dev once (train will be reshaped each MB)
    # (our graph assumes tensor-shaped input: N x W x W x L)
    C = 2
    dev_x = np.reshape(dev_x,(-1,gw,gh,1))
    dev_y = np.reshape(dev_y,(-1,gw,gh,C))

    init = build_model_1(args, (gw,gh), C)
    pygame.init()
    sim_count = 7
    patience = 12
    best_dev = 0.0
    since_best = 0
    # train
    with tf.Session() as sess:
        sess.run(fetches=[init]) # passing as tensor variable rather than name
                                 # just to mix things up

        for epoch in range(args.epochs):
            # shuffle data once per epoch
            idx = np.random.permutation(N)
            train_x = train_x[idx,:]
            train_y = train_y[idx,:]

            # train on each minibatch
            for update in range(int(np.floor(N/args.mb))):
                mb_x = train_x[(update*args.mb):((update+1)*args.mb),:]
                mb_x = np.reshape(mb_x,(args.mb,gw,gh,1)) # reshape vector into tensor

                mb_y = train_y[(update*args.mb):((update+1)*args.mb),:]
                mb_y = np.reshape(mb_y,(args.mb,gw,gh,C)) # reshape vector into tensor
                #print("mb_x shape: %s"%mb_x.shape)
                #print("mb_y shape: %s"%mb_y.shape)
                # note: using strings for fetches and feed_dict
                _,my_obj = sess.run(fetches=["train_step","obj:0"],\
                        feed_dict={"x:0":mb_x,"y_true:0":mb_y}) 

            # evaluate once per epoch on dev set
            [obj,acc] = sess.run(fetches=["obj:0","acc:0"],\
                    feed_dict={"x:0":dev_x,"y_true:0":dev_y})
            
            print("Epoch %d: dev=%.5f (obj=%.5f)" % (epoch,acc,obj))
            if acc > best_dev:
                since_best = 0
                best_dev = acc
            else:
                since_best += 1
            if since_best == patience:
                print("Early stopping...")
                break
        # run a few simulations
        for i in range(sim_count):
            run_simulation(sess, game, False, 1000)
    print("all done...")
    pygame.quit()
    pygame.time.delay(1)

if __name__ == "__main__":
    q_learn_main()
    pass
