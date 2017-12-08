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
        adj_p = game.adjacentPositions(pos)
        for adj_x, adj_y in adj_p:
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
            
            neigh_count = 0
            neigh_min = 2
            # 0 = revealed (safe+dont click), 1 = safe, 2 = mine
            if len(adj_arr) == 3:
                # in corner
                neigh_min = 1
            elif len(adj_arr) == 5:
                # on edge
                neigh_min = 2
            else:
                # inner field
                neigh_min = 3
            for p in adj_arr:
                nx,ny = p
                n = game.cell[nx][ny]
                if n.revealed:
                    neigh_count += 1
                    if neigh_count == neigh_min:
                        break

                
            mystery = neigh_count < neigh_min
            
            #if cell.revealed:
            #    # I've seen it
            #    val = 0
            if cell.revealed:
                # don't punish them for not knowing the danger...
               val = 0
            elif mystery:
                # danger or potential danger
                val = 2
            elif cell.mine:
                val = 1
            else:
                # its a safe place to click
                val = 0
            
            class_count = 3
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



def build_model_1(args, grid_size, C):
    """
    Adds a CNN to the graph.

    :param args: (string) the parsed argument object

    """
    gw,gh = grid_size
    cell_count = gw*gh
    # placeholders
    
    x = tf.placeholder(dtype=tf.float32,shape=(None,gw,gh,1),name="x")
    y_true = tf.placeholder(dtype=tf.int64,shape=(None,gw,gh,C),name="y_true")
    a0 = x # for notational simplicity/consistency

    filters = 42
    
    # build CNN, putting each layer in its own scope
    with tf.variable_scope("layer1_conv"):
        a1 = conv2d(a0,k=5,L=1,Lprime=filters,S=1,P="SAME",f="relu")
        # a1 is MB x 9 x 9 x 32
        # need to transform MB * in_size
    
    with tf.variable_scope("layer2_conv"):
        a2 = conv2d(a1,k=3,L=filters,Lprime=C,S=1,P="SAME",f="relu")
        # a1 is MB x 9 x 9 x 32
    """with tf.variable_scope("layer3_conv"):
        a3 = conv2d(a2,k=3,L=C,Lprime=1,S=1,P="SAME",f="relu")
        # a1 is MB x 9 x 9 x 32"""

    conv_out = a2
        
    # flatten the tensor
    in_size = cell_count*C
    #in_size = cell_count
    residual = tf.add(a0, conv_out)
    
    hidden_in = tf.reshape(conv_out,(-1, in_size))
    hidden_dims = [cell_count*C]
    hidden_layer_count = len(hidden_dims)
    # weight matrices and bias vectors for DNN component

    # add skip connection from a0 to the DNN by adding a0 elemwise to hidden_in
    
    
    W = []
    B = []
    from_dim = in_size
    for i in range(hidden_layer_count):
        units = hidden_dims[i]
        # matrix transforms from from_dims to hidden_dim
        temp_w = tf.get_variable(name="w%d"%i, shape = (from_dim,units), dtype = tf.float32,
                                     initializer=tf.glorot_uniform_initializer())
        b = None
        if args.f == 'relu':
            b = tf.get_variable(name="b%i"%i, dtype = tf.float32,
                            initializer=tf.constant([0.1]*units))
        else:
            b = tf.get_variable(name="b%i"%i, shape = (units), dtype = tf.float32,
                                initializer=tf.zeros_initializer())
        B.append(b)
        W.append(temp_w)
        from_dim = units
        # add final matrix from last hidden layer to output
    W.append(tf.get_variable(name="wf", shape=(from_dim, cell_count*C), dtype = tf.float32,
                             initializer=tf.glorot_uniform_initializer()))
    # and the final bias vector
    B.append(tf.get_variable(name="bf", shape=(cell_count*C), dtype = tf.float32,
                             initializer=tf.zeros_initializer()))


    # f will be used as the activation for all hidden layers
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
    in_layer = hidden_in
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


    # define loss
    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=a3,labels=y_true)
    #obj = tf.reduce_mean(cross_entropy,name="obj")
    lbl = tf.reshape(y_true, (-1, cell_count, C))
    pred = tf.reshape(z, (-1, cell_count, C), name="logits")
    entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=lbl,
        logits=pred)
    obj = tf.reduce_mean(entropy)
    obj_id = tf.identity(obj, name="obj")
    # optimizer
    train_step = tf.train.AdamOptimizer(args.lr).minimize(obj,name="train_step")

    # define accuracy
    
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(lbl,axis=2),tf.argmax(pred,axis=2)),tf.float32),name="acc")
    
    #acc = tf.reduce_mean(tf.cast(tf.square_difference(cast_y_true,a3),tf.float32),name="acc")

    # define init op last
    init = tf.global_variables_initializer()

    return init


def parse_all_args():
    """
    Parses arguments

    :return: the parsed arguments object
    """
    # parse arguments
    parser = argparse.ArgumentParser()


    parser.add_argument("-model", type=int, help="The model specifier (an int) [default = 1]", default=1)
    parser.add_argument("-lr",type=float,\
            help="The learning rate (a float) [default = 0.002]",default=0.0018)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (an int) [default = 1]",default=512)
    parser.add_argument("-epochs",type=int,\
            help="The number of epochs to train for [default = 100]",default=100)
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
    screen = pygame.display.set_mode((800,800))
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
        SAFE = 0
        MINE = 1
        MYSTERY = 2
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
            elif cell_class == MYSTERY:
                guess_col = col_mystery
            else:
                guess_col = col_mine
            """elif cell_class == SEEN:
                guess_col = col_visited
            elif cell_class == MINE:
                guess_col = col_mine
            else:
                guess_col = col_mystery"""

            if cell.mine:
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
    
def main():
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

    test_size = 100000
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
    C = 3
    dev_x = np.reshape(dev_x,(-1,gw,gh,1))
    dev_y = np.reshape(dev_y,(-1,gw,gh,C))

    init = build_model_1(args, (gw,gh), C)
    pygame.init()
    sim_count = 10
    patience = 25
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
            run_simulation(sess, game, False, 930)
    print("all done...")
    pygame.quit()
    pygame.time.delay(1)

if __name__ == "__main__":
    main()
    pass
