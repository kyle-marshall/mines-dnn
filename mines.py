#!/usr/bin/env python3

import os
import random
import pygame
from pygame import *
import math

class Difficulty:
    def __init__(self,name,width,height,mineCount):
        self.name = name
        self.width = width
        self.height = height
        self.mines = mineCount

class DifficultyFactory:
    EASY = Difficulty("Easy"  ,  9,  9, 10)
    MEDIUM = Difficulty("Medium", 16, 16, 40)
    EXPERT = Difficulty("Expert", 30, 16, 99)
    MORE_EXPERT = Difficulty("More Expert", 40, 25, 200)
    RIDICULOUS = Difficulty("Ridiculous", 50, 30, 320)
                             
class Cell:
    def __init__(self):
        self.mine = False
        self.revealed = False
        self.flag = False
        self.question = False

class Minefield:
    # Minefield is a model for the game of minesweeper
    def __init__(self, diff = None):
        if diff is None:
            diff = DifficultyFactory.EASY
        self.difficulty = diff
        self.mines = diff.mines
        self.width = diff.width
        self.height = diff.height
        self.cell = []
        self.startWithIsland = False
        self.lose = False
        self.win = False
        self.started = False
        self.useQuestion = False
        self.flagged = 0
        self.initGrid();
        print("Size: (%d, %d), Mines: %d"%(self.width,self.height,self.mines))
        
    def initGrid(self):
        self.cell = []
        for x in range(self.width):
            col = []
            for y in range(self.height):
                col.append(Cell())
            self.cell.append(col)

    def reset(self):
        self.lose = False
        self.win = False
        self.started = False
        self.flagged = 0
        self.initGrid()

    def setDifficulty(self, diff):
        self.width = diff.width
        self.height = diff.height
        self.mines = diff.mines
        self.reset()

    def setRandomState(self):
        # place new mines
        self.reset()
        # randomly reveal part of the board
        targRatio = random.random()*0.8
        #if targRatio == 0:
        #    return
        #targRatio = 1.0
        visited = []
        
        rat = 0.0
        while rat < targRatio and not (self.win or self.lose):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            cell = self.cell[x][y]
            if cell.revealed:
                continue
            elif cell.mine:
                if not cell.flag:
                    cell.flag = True
            else:
                self.reveal((x,y))
            rat = self.ratioRevealed()
        
    def getSize(self):
        return (self.width, self.height)

    def percentRevealed(self):
        i = 0
        for x in range(self.width):
            for y in range(self.height):
                if self.cell[x][y].revealed or (self.cell[x][y].mine and self.cell[x][y].flag):
                    i += 1
        s = round(float(float(i)/float(self.width*self.height)*100),2) 
        return str(s)+"%"

    def ratioRevealed(self):
        i = 0
        for x in range(self.width):
            for y in range(self.height):
                if self.cell[x][y].revealed or (self.cell[x][y].mine and self.cell[x][y].flag):
                    i += 1
        s = float(float(i)/float(self.width*self.height))
        return s
    
    def firstReveal(self,pos):
        self.setMines(pos)
        self.started = True
        self.reveal(pos)

    def setMines(self,safePos):
        safeX, safeY = safePos
        verbose = False
        if verbose: print("placing %d mines."%self.mines)
        complete = False
        t = 0
        # safeRange is how many cells past safePos
        # must also not contain a mine
        safeRange = 2 if self.startWithIsland else 0
        while not complete:
            x = random.randint(0,self.width-1)
            y = random.randint(0,self.height-1)
            if abs(safeX-x)+abs(safeY-y) > safeRange:
                if not self.cell[x][y].mine:
                    self.cell[x][y].mine = True
                    t += 1
            if t == self.mines:
                complete = True

    def checkWin(self):
        verbose = False
        win = True
        for x in range(self.width):
            for y in range(self.height):
                if not self.cell[x][y].revealed:
                    if not (self.cell[x][y].mine and self.cell[x][y].flag):
                        win = False
        if win and verbose:
            print("We have a winner!")
        self.win = win

    def solve(self):
        #reveal all empty cells, flag all mines
        for x in range(self.width):
            for y in range(self.height):
                if self.cell[x][y].mine:
                    if not self.cell[x][y].flag:
                        self.cell[x][y].flag = True
                        self.flagged += 1
                else:
                    if self.cell[x][y].flag:
                        self.cell[x][y].flag = False
                        self.flagged -= 1
                    if not self.cell[x][y].revealed:
                        self.cell[x][y].revealed = True
        self.lose = False
        self.win = True

    def reveal(self,pos):
        debug = False
        if not self.started:
            # first reveal places mines after knowing the position
            # to ensure that the spot is safe
            if debug: print("first reveal...")
            self.firstReveal(pos)
        else:
            if self.win or self.lose:
                return
            x, y = pos
            cell = self.cell[x][y]
            if debug: print("cell info: (revealed: %s, flag: %s, mine: %s)"%(cell.revealed, cell.flag, cell.question))
            if not self.cell[x][y].revealed and not self.cell[x][y].flag and not self.cell[x][y].question:
                self.cell[x][y].revealed = True
                if self.cell[x][y].mine:
                    self.lose = True
                else:
                    s = self.surrounding(pos)
                    if debug: print("%d mines surround."%s)
                    if s == 0:
                        self.revealAround(pos)
                self.checkWin()
                    
    def revealAround(self,pos):
        debug = False
        if debug: print("Autoreveal around (%d, %d)"%pos)
        for adjPos in self.adjacentPositions(pos):
            x, y = adjPos
            if not self.cell[x][y].revealed:
                if debug: print("Auto revealing (%d, %d)"%adjPos)
                self.reveal(adjPos)

    def flag(self,pos):
        if self.win or self.lose:
            return
        x, y = pos
        cell = self.cell[x][y]
        if not cell.revealed:
            if cell.flag:
                cell.flag = False
                if self.useQuestion:
                    cell.question = True
                self.flagged -= 1
            elif self.useQuestion and cell.question:
                cell.question = False
            else:
                cell.flag = True
                self.flagged += 1

    def adjacentPositions(self, pos):
        positions = []
        px, py = pos
        for x_i in range(3):
            for y_i in range(3):
                tx = px - 1 + x_i
                ty = py - 1 + y_i
                if tx >= 0 and tx < self.width and ty >= 0 and ty < self.height:
                    if not (tx == px and ty == py):
                        positions.append((tx, ty))
        return positions
                        
    def surrounding(self, pos):
        mineCount = 0
        for adjPos in self.adjacentPositions(pos):
            x, y = adjPos
            if self.cell[x][y].mine:
                mineCount += 1
        return mineCount
