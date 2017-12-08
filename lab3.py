#! /usr/bin/env python

"""
@authors: Brian Hutchinson (Brian.Hutchinson@wwu.edu)
          Kyle Marshall
          Nick Mounier

This script trains and evaluates a multinomial logistic regression model.
It supports an arbitrarily deep neural network and two modes: classification / autoencoder mode
Supported activation functions are sigmoid, relu, and tanh

For usage, run with the -h flag.

Example command:

python lab3_demo.py data/train.csv.gz data/dev.csv.gz

"""

import tensorflow as tf
import argparse
import sys
import numpy as np
import math
import mines

#model = "/tmp/model.ckpt"
global train_sess
global train_game
train_game = None
train_sess = None

def score_output(output_vec, game, dontAlter = True):
    score_safe_reveal = 1
    
    # tries the move prodced by the model
    # interpret the output vectorf
    o_len = tf.size(output_vec)
    val, idx = max((v,i) for i,v in enumerate(output_vec))
    # the index will determine the move
    gw, gh = game.getSize()
    cell_ct = gw*gh
    cell_idx = idx%(cell_ct)
    # output_vec is 2*cell_ct in length
    # if choice in first half, reveal mine
    # otherwise place flag
    pos = (cell_idx // gw, cell_idx % gh)
    x, y = pos
    cell = game.cell[x][y]
    scre = 0
    if not cell.revealed and not cell.mine:
        # 1 point if no adjacent cells have been revealed
        neighbor_revealed = False
        for adj_x, adj_y in game.adjacentPositions(pos):
            if game.cell[adj_x][adj_y].revealed:
                neighbor_revealed = True

        if not dontAlter:
            game.reveal(pos)
        return 2 if neighbor_revealed else 1
    elif cell.revealed:
        return -2
    return -1

def get_game_state(game):
    gw, gh = game.getSize()
    vec = []
    for gx in range(gw):
        for gy in range(gh):
            val = 0 # unrevealed cell value
            cell = game.cell[gx][gy]
            if cell.revealed:
                s = 1.0+game.surrounding((gx,gy))
            vec.append(val/9.0) # map values to 0.0-1.0
    return np.array(vec)

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

def build_graph(D, C, args):
    
    """
    Constructs the multinomial logistic regression graph.
    :param D: (integer) the input feature dimension
    :param C: (integer) the number of classes
    :param args: arguments like learning rate...
    :param game: the minesweeper game model so we can run simulations
    """
    # set seed
    #tf.set_random_seed(args.seed)
    #np.random.seed(args.seed)

    # parse hidden dimension string as list of ints
    lr = args.lr
    dims = []
    if args.hidden_dims != "":
        raw_dims = args.hidden_dims.split(",")
        dims = list(map(int, raw_dims))
    
    # placeholders
    # note: None leaves the dimension unspecified, allowing us to
    #       feed in either minibatches (during training) or the dev set
    x1      = tf.placeholder(dtype=tf.float32,shape=(None,D),name="x1")
    #y_true = tf.placeholder(dtype=tf.int64,shape=(None),name="y_true")

    p_scores      = tf.placeholder(dtype=tf.float32,shape=(None,D),name="possible_scores")
    #rev_score = tf.placeholder(dtype=tf.float32, shape=(),name="prev_score")
    
    # variables
    hidden_layer_count = len(dims)
    
    # weight matrices and bias vectors
    W = []
    B = []
    from_dim = D
    for i in range(hidden_layer_count):
        hidden_dim = dims[i]
        # matrix transforms from from_dims to hidden_dim
        temp_w = tf.get_variable(name="w%d"%i, shape = (from_dim, hidden_dim), dtype = tf.float32,
                                     initializer=tf.glorot_uniform_initializer())
        b = None
        if args.f == 'relu':
            b = tf.get_variable(name="b%i"%i, dtype = tf.float32,
                            initializer=tf.constant([0.1]*hidden_dim))
        else:
            b = tf.get_variable(name="b%i"%i, shape = (hidden_dim), dtype = tf.float32,
                                initializer=tf.zeros_initializer())
        B.append(b)
        W.append(temp_w)
        from_dim = hidden_dim
        
    # add final matrix from last hidden layer to output
    W.append(tf.get_variable(name="wf", shape=(from_dim, C), dtype = tf.float32,
                             initializer=tf.glorot_uniform_initializer()))
    # and the final bias vector
    B.append(tf.get_variable(name="bf", shape=(C), dtype = tf.float32,
                             initializer=tf.zeros_initializer()))


    # f will be used as the activation for all hidden layers
    f = None
    if args.f == 'tanh':
        f = tf.tanh
    elif args.f == 'sigmoid':
        f = tf.sigmoid
    elif args.f == 'relu':
        f = tf.nn.relu
    else:
        print("bad activation function")
        return

    # forward propogation through all hidden layers
    in_layer = x1
    for i in range(hidden_layer_count):
        mat = W[i]
        b = B[i]
        z = tf.matmul(in_layer, mat) + b
        a = f(z)
        in_layer = a

    # for the output, do not use f as activation
    mat = W[hidden_layer_count]
    z = tf.matmul(in_layer, mat) + B[hidden_layer_count]
    # producing logits
    sig = tf.nn.sigmoid(z)

    
    #diff = p_scores * sig
    #res = tf.select(tf.less(d, -0.5), -2, tf.select(tf.greater(d, 0.5), 0, 1))
    #score_func = lambda d: -2 if d < 0 else 0 if d > 0.5 else 1
    """
    thresh = tf.constant(-0.5)
    penalty_mask = tf.map_fn(lambda x: tf.less(x, 0.5), diff, dtype="bool")
    bonus_mask = tf.map_fn(lambda x: tf.logical_and(tf.greater_equal(x, 0.0), tf.less(x, 0.5)), diff, dtype="bool")
    penalties = -2*tf.ones_like(diff)
    bonuses = tf.ones_like(diff)
    zeros = tf.zeros_like(diff)"""
    #penalty = tf.boolean_mask(penalties, penalty_mask)

    """
    func = lambda x: tf.case({tf.less(x, -0.5): bad_score,
                    tf.less(x, 0): neutral_score,
                    tf.less(x, 0.5): good_score},
                    default=neutral_score)
    """
    
    tf.losses.sigmoid_cross_entropy(
        multi_class_labels,
        logits,
        weights=1.0,
        label_smoothing=0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
    )
    softmax = tf.nn.softmax(z)
    
    total_score = tf.reduce_sum(tf.multiply(p_scores, softmax), name="score")
    
    # loss (performs softmax implicitly for us - supports minibatches)
    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=sig, labels=p_scores)
    #obj = tf.reduce_mean(cross_entropy,name="obj")    
    #out_softmax = tf.nn.softmax(z)
    #choice = tf.argmax(out_softmax)
    #score = p_scores[i]
    #,"score")
    
    
    #obj = tf.reduce_mean(tf.squared_difference(out_sig, ideal_sig), name="error")
    
    # side effect operations
    train_step = tf.train.AdamOptimizer(lr).minimize(-total_score,name="train_step")

    init = tf.global_variables_initializer()
    return init

def parse_args():
    """
    Parses args, loads and normalizes data, builds graph, trains and evaluates.

    """
    # parse arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument("-h", help="show this help message and exit")
    parser.add_argument("-f", type = str, help="Hidden activation (in the set {sigmoid, tanh, relu}) [default: \"tanh\"]", default="relu")
    parser.add_argument("-lr",type=float,\
            help="The learning rate (a float)",default=0.01)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (an int)",default=1)

    parser.add_argument("-epochs",type=int,\
            help="The number of epochs to train for",default=200)
    parser.add_argument("-patience", type=int,
                        help = "How many epochs to continue training without improving\
                        dev accuracy (int) [default: 10]", default=10)
    parser.add_argument("-seed", type=int, help = "random seed (int) [default: 497]",
                        default = 497)
    parser.add_argument("-model", type = str, help = "Save the best model with this prefix\
                        (string) [default:\"/tmp/model.ckpt\"]", default = "/tmp/model.ckpt")

    parser.add_argument("-hidden-dims", type= str,\
               help = "A string with the dimension of each hidden layer, comma -delimited (string) [default: \"50,50\"]",\
               default = "50,50,50")
    args = parser.parse_args()
    # validate args
    if args.f not in ['relu', 'sigmoid', 'tanh']:
        print("Error: invalid activation function")
        return None
    return args


def gen_test_set(game, how_many):
    in_states = []
    score_maps = []
    for i in range(how_many):
        if i % 10 == 0:
            print("%s of %s..."%(i, how_many))
        game.setRandomState()
        state = np.array(get_game_state(game))
        scores = np.array(get_possible_score_array(game))
        in_states.append(state)
        score_maps.append(scores)
        
    print("all done...")
    in_states = np.asanyarray(in_states)
    print("with (%s)"%len(score_maps))
    score_maps = np.asanyarray(score_maps)
    print("tyranny")
    return (in_states, score_maps)

def main():
    args = parse_args()

    # get default minesweeper model

    game = mines.Minefield(mines.DifficultyFactory.EASY)
    gw,gh = game.getSize()

    # compute relevant dimensions
    D = gw*gh # input is represented with 1 float per cell
    C = D # output represented with 1 floats per cell, (1 possible moves per cell)

    # for early stopping
    best_score = 0
    
    # epochs without better accuracy
    badcount = 0


    
    # converged = false
    # run graph
    games_per_epoch = 50
    global train_sess
    train_sess = tf.Session()
    # build graph
    init = build_graph(D,C,args)
    # allow best model to be saved
    saver = tf.train.Saver()
    ckpt_fn = args.model

    test_size = 100
    print("generating test data...")
    test_x, test_scores = gen_test_set(game, test_size)
    print("data generated.")
    with train_sess.as_default():
        train_sess.run(init) # note: can specify fetches by tensor/op name
        
        for epoch in range(args.epochs):

            # run games_per_epoch test games
            err_sum = 0
            for t in range(test_size):
                game.setRandomState()
                state = test_x[t]
                scores = test_scores[t]
                lbl = [i 

                _,error = train_sess.run(fetches=["train_step","score:0"],\
                    feed_dict={"x1:0":[state],"lbl:0",[lbl],"lbl_weights:0":[scores]})
                err_sum += error
            avg_err = err_sum / games_per_epoch
            print("Epoch #%s avg score: %s"%(epoch, avg_err))
            
            """
            # evaluate once per epoch on dev set
            [score] = sess.run(fetches=["score:0"],\
                    feed_dict={"x1:0":dev_x})
            if score > best_score:
                best_score = score
                badcount = 0
                # save best model
                saver.save(sess, ckpt_fn)
            else:
                badcount += 1

            print ("Epoch %d: dev=%.5f badcount=%d" % (epoch,my_dev_err,badcount))

            # early stopping
            if badcount == args.patience:
                print("Converged due to early stopping...")
                break
            """
        #print("Best score=%.5f"%low_dev_err)
            

if __name__ == "__main__":
    main()
