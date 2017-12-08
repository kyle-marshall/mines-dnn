#!/usr/bin/env python3

import os
import random
import pygame
from pygame import *
import math
import mines

ScreenSize = (800,600)

def ratioColor(c1,c2,rat):
    # returns an rgb tuple that is between c1 and c2 depending on ratio
    # i.e. if rat is 0.0, c1 is returned, if 1.0 then c2
    
    rat = min(max(rat, 0.0), 1.0) # limit rat to range [0.0-1.0]
    colChange = tuple(int((c2[i]-c1[i])*rat) for i in range(3))
    r,g,b = (c1[i]+colChange[i] for i in range(3))
    #print("%s + %s = %s"%(c1, colChange, (r,g,b)))
    return (r,g,b)

def randomColor():
    return tuple([random.randint(0,255) for i in range(3)])

class MinefieldViewer:
    # MinefieldViewer is a viewer for the minesweeper model (MineField)
    def __init__(self, minefield, surf, image = None):
        self.model = minefield;
        self.cellSize = (0, 0) # gets set when surface is set
        self.needsRedraw = False

        # surface represents the whole game window
        # modelSurface is just used to draw the minefield
        self.surface = None
        self.modelSurface = None
        self.modelSurfaceOffset = (0,0)
        self.setSurface(surf)
        if image is not None:
            self.setImage(image)

        self.initConstants()
        self.initColors()
        self.initFonts()
        self.genSprites()
        self.invalidate()
        self.titleHeight = 30

    def initConstants(self):
        # keys for colors and pre-rendered sprites
        self.HIDDEN = "hidden"
        self.HIGHLIGHT = "highlight"
        self.REVEALED = "revealed"
        self.EMPTY = "empty"
        self.FLAG = "flag"
        self.MINE = "mine"
        self.QUESTION = "question"
        self.NUMBER = "number"
        self.MISTAKE = "mistake"
        self.WIN = "win"
        self.LOSE = "lose"
        self.BACKGROUND_1 = "background_1"
        self.BACKGROUND_2 = "background_2"
        self.GUI_TEXT = "gui_text"
        self.MAGIC_PINK = "magic_pink"

    def initFonts(self):
        self.fontPath = "FjallaOne-Regular.ttf"
        self.cellFont = pygame.font.Font(self.fontPath, 12)
        self.guiFont = pygame.font.Font(self.fontPath, 16)
        self.titleFont = pygame.font.Font(self.fontPath, 18)

    def initColors(self, fn = None):
        self.colors = {}
        self.colors[self.MAGIC_PINK] = (255, 0, 255)
        if fn is None or fn == "":
            # default colors
            self.colors[self.HIDDEN] =   (200, 200, 220)
            self.colors[self.HIGHLIGHT] =(230, 230, 240)
            self.colors[self.REVEALED] = (240, 240, 250)
            self.colors[self.EMPTY] =    (170, 170, 245)
            self.colors[self.NUMBER] =   (20,  20,  20)
            self.colors[self.FLAG] =     (250, 35,  35)
            self.colors[self.QUESTION] = (145, 30,  245)
            self.colors[self.MINE] =     (100, 110, 105)
            self.colors[self.MISTAKE] =  (180, 50,  5)
            self.colors[self.WIN] =      (40,  220, 40)
            self.colors[self.LOSE] =     (220, 40,  32)
            self.colors[self.BACKGROUND_1] = (70, 70, 90)
            self.colors[self.BACKGROUND_2] = (60, 60, 80)
            self.colors[self.GUI_TEXT] = (210, 200, 230)
            for n in range(1, 10):
                # assign random colors to number digits
                key = self.getNumberKey(n)
                self.colors[key] = tuple([random.randint(0,255) for i in range(3)])
        # TODO get colors from file

    def getNumberKey(self, n):
        # keys for colors and pre-rendered sprites
        return "number_%d"%n

    def invalidate(self):
        self.needsRedraw = True
    def setSurface(self, surface):
        self.surface = surface
        sw, sh = surface.get_size()
        # make model (the minefield) surface fill some ratio of the total window width/height
        wRat = 0.8
        hRat = 0.8
        # proposed model surface size:
        modelSurfSize = (int(wRat*sw), int(hRat*sh))
        mw,mh = modelSurfSize
        gw, gh = self.model.getSize() # grid size
        # calculate cellSize so cells fit in model surface
        self.cellSize = (int(mw / gw), int(mh / gh))
        cw, ch = self.cellSize
        # use the rounded cellSize to tweak the modelSurfSize slightly
        # so that cells fit better
        if mw % cw != 0:
            mw = int(mw / cw) * cw
        if mh % ch != 0:
            mh = int(mh / ch) * ch
        modelSurfSize = (mw,mh)
        # finally make the model surface
        # and set it's offset to center it
        self.modelSurface = pygame.surface.Surface(modelSurfSize)
        self.modelSurfaceOffset = ((sw-mw)//2, ((sh-mh)//2))
        
        print("screen size: (%dpx, %dpx)"%(sw, sh))
        print("modelSurfaceSize: (%d,%d)"%modelSurfSize)
        print("modelSurfaceOffset: (%d,%d)"%self.modelSurfaceOffset)
        print("minefield size: (%d, %d)"%(gw, gh))
        
    def getSurface(self):
        return self.surface
    def getWindowSurface(self):
        return self.surface
    def getModelSurface(self):
        return self.modelSurface
        
    def setImage(self, img):
        targetSize = (self.cellSize*self.model.width,self.cellSize*self.model .height)
        self.image = pygame.transform.scale(self.image,targetSize)

    def genSprite(self, size, txt, fg, bg):
        # very simple sprite generation using
        # rectangle size, some text, foreground/background colors
        cw, ch = size
        sprite = pygame.Surface((cw,ch))
        sprite.fill(bg)
        if not (txt is None or txt == ""):
            tw, th = self.cellFont.size(txt)
            xOff = cw//2 - tw//2
            yOff = ch//2 - th//2
            sprite.blit(self.cellFont.render(txt, True, fg), (xOff, yOff))
        return sprite
            
    def genSprites(self):
        cs = self.cellSize
        mid = tuple([int(dim/2) for dim in cs])
        midX, midY = mid
        self.sprites = {}
        # hidden cell
        self.sprites[self.HIDDEN] = self.genSprite(cs, "", None, self.colors[self.HIDDEN])
        # digit sprites 0-9   (0 is a blank cell)
        for x in range(0,10):
            key = self.getNumberKey(x)
            txt = "" if x == 0 else str(x)
            fgCol = None if x == 0 else self.colors[key]
            self.sprites[key] = self.genSprite(cs, txt, fgCol, self.colors[self.REVEALED])
        # marker sprites
        self.sprites[self.FLAG] = self.genSprite(cs, "M", self.colors[self.HIDDEN], self.colors[self.FLAG])
        self.sprites[self.QUESTION] = self.genSprite(cs, "?", self.colors[self.HIDDEN], self.colors[self.QUESTION])
        # highlighted cell (drawn over highlighted hidden cells)
        self.sprites[self.HIGHLIGHT] = self.genSprite(cs, "", None, self.colors[self.HIGHLIGHT])
        # the mine sprite
        mine = pygame.surface.Surface(cs)
        self.sprites[self.MINE] = mine
        mine.fill(self.colors[self.MAGIC_PINK])
        mine.set_colorkey(self.colors[self.MAGIC_PINK])
        pygame.draw.circle(mine, self.colors[self.MINE], mid, min(midX, midY)-2)
        # highlight mistake
        self.sprites[self.MISTAKE] = self.genSprite(cs, "X", self.colors[self.MINE], self.colors[self.MISTAKE])
                
        """
        self.spriteX = pygame.Surface(s)
        self.spriteX.fill((255,0,255))
        pygame.draw.line(self.spriteX,(20,20,20),(self.flagPadding,self.flagPadding),(s[0]-self.flagPadding-1,s[1]-self.flagPadding-1))
        pygame.draw.line(self.spriteX,(20,20,20),(self.flagPadding,s[1]-self.flagPadding-1),(s[0]-self.flagPadding-1,self.flagPadding))
        self.spriteX.set_colorkey((255,0,255))
        """
        
    def drawModel(self):
        sw, sh = self.modelSurface.get_size()
        gw, gh = self.model.getSize()
        
        cw, ch = self.cellSize
        # draw cell with 1 px trimmed off height and width
        # so that borders show
        cellRect = (0,0,cw-1,ch-1)
        surf = self.modelSurface;
        m = self.model
        
        if m.win:
            pygame.draw.rect(surf, self.colors[self.WIN], (0,0,sw,sh))
        elif m.lose:
            pygame.draw.rect(surf, self.colors[self.LOSE], (0,0,sw,sh))
        else:
            pygame.draw.rect(surf, (0, 0, 0), (0,0,sw,sh))

        for gx in range(gw):
            for gy in range(gh):
                pos = (gx * cw, gy * ch)
                if not m.cell[gx][gy].revealed:
                    # draw hidden cell
                    surf.blit(self.sprites[self.HIDDEN], pos, cellRect)
                    if m.cell[gx][gy].flag:
                        surf.blit(self.sprites[self.FLAG], pos, cellRect)
                    elif m.cell[gx][gy].question:
                        surf.blit(self.sprites[self.QUESTION], pos, cellRect) 
                    """if self.lose and not self.cell[x][y].mine:
                            Display.blit(self.sprites[X_MARKER],p)
                       """
                    # show unrevealed mines if player lost
                    if m.lose and m.cell[gx][gy].mine:
                        surf.blit(self.sprites[self.MINE], pos, cellRect)
                    
                elif m.cell[gx][gy].revealed and not m.cell[gx][gy].mine:
                    s = m.surrounding((gx,gy))
                    key = self.getNumberKey(s)
                    surf.blit(self.sprites[key], pos, cellRect)
                else:
                    surf.blit(self.sprites[self.MISTAKE], pos, cellRect)
                    surf.blit(self.sprites[self.MINE], pos, cellRect)
                    
    def drawGui(self):
        sw, sh = self.surface.get_size()
        centerX = int(sw/2)
        guiTextCol = self.colors[self.GUI_TEXT]
        renderText(self.surface,self.titleFont, "MineSweepy", guiTextCol, (centerX, 15), True)
        minesLeft = self.model.mines - self.model.flagged
        renderText(self.surface,self.guiFont, "Mines Left: %d"%minesLeft, guiTextCol,(20,30))
        
        diffText = "Difficulty: %s"%self.model.difficulty.name
        diffTextSize = self.guiFont.size(diffText)

        diffChangeText = "(change with # keys or Z/X)"
        diffChangeTextSize = self.cellFont.size("(change with # keys or Z/X)")
        
        renderText(self.surface, self.guiFont, diffText, guiTextCol,(sw-diffTextSize[0]-20,30))
        renderText(self.surface, self.cellFont, diffChangeText, guiTextCol,(sw-diffChangeTextSize[0]-20,60))
        self.surface.blit(self.modelSurface, self.modelSurfaceOffset)

        if self.model.lose or self.model.win:
            renderText(self.surface, self.guiFont, "Press 'R' for new game", guiTextCol,(centerX, sh-30), True)
            rat = self.model.ratioRevealed()
            pr = self.model.percentRevealed()
            c = ratioColor((255,50,0), (20,255,0), rat)
            renderText(self.surface, self.titleFont, "Revealed: %s"%pr, c, (centerX, 40), True)
            
            msg = "You Lose" if self.model.lose else "You Win!"
            msgCol = self.colors[self.LOSE] if self.model.lose else self.colors[self.WIN]
            renderText(self.surface, self.titleFont, msg, msgCol, (centerX, 60), True)

    def drawAll(self):
        # clear screen
        self.surface.fill((0,0,0))

        # draw minefield
        self.drawModel()
        # draw gui elements
        self.drawGui()
        
        pygame.display.update()
        self.needsRedraw = False

    def onClick(self, surfPos, e):
        x,y = surfPos
        cw,ch = self.cellSize
        mx,my = self.modelSurfaceOffset
        gx = int((x-mx)//cw)
        gy = int((y-my)//ch)
        gw, gh = self.model.getSize()
        print("Click (%dpx, %dpx) --> (%d, %d)"%(x,y,gx,gy))
        if gx >= 0 and gx < gw and gy >= 0 and gy < gh:
            if e.button == 1:
                # left click
                self.model.reveal((gx, gy))
            elif e.button == 3:
                # right click
                self.model.flag((gx, gy))
            self.invalidate()

    def update(self):
        if self.needsRedraw:
            self.drawAll()
        
def renderText(surf,font,text,color,pos,centerx=False, centery=False):
    x = pos[0]
    y = pos[1]
    s = font.size(text)
    if centerx:
        x -= int(s[0]/2)
    if centery:
        y -= int(s[1]/2)
    surf.blit(font.render(text,True,color),(x,y))

def genBackground():
    for x in range(int(ScreenSize[0]/40)):
        for y in range(int(ScreenSize[1]/40)):
            if x%2 != y%2:
                pygame.draw.rect(Background, ColorBG1,(x*40,y*40,40,40))
            else:
                pygame.draw.rect(Background, ColorBG2,(x*40,y*40,40,40))

def main():
    pygame.init()
    screen = pygame.display.set_mode(ScreenSize)
    pygame.display.set_caption("MineSweepy")

    running = True
    difficulty = []
    difficulty.append(mines.DifficultyFactory.EASY)
    difficulty.append(mines.DifficultyFactory.MEDIUM)
    difficulty.append(mines.DifficultyFactory.EXPERT)
    difficulty.append(mines.DifficultyFactory.MORE_EXPERT)
    difficulty.append(mines.DifficultyFactory.RIDICULOUS)
    diffIdx = 2
    defaultDifficulty = difficulty[diffIdx]
    
    """    
    colorFiles = []
    for fn in os.listdir("color"):
        if fn.endswith(".txt"):
            colorFiles.append("color/"+fn)
    currentColor = 0
    for x in range(len(colorFiles)):
        if colorFiles[x] == "color/default.txt":
            currentColor = x
    loadColors(colorFiles[currentColor])
    """
    
    game = mines.Minefield(defaultDifficulty)
    viewer = MinefieldViewer(game, screen)

    keyToDifficultyIdx = {}
    for i in range(len(difficulty)):
        # pygame keyboard constants are integers (K_0 -> K_9 is an ascending sequence)
        keyToDifficultyIdx[K_1+i] = i
        if i == 9:
            print("keys 1-9 already mapped, too many difficulties")
            break
        
    while running:
        for e in pygame.event.get():
            
            if e.type == QUIT:
                running = False
                
            elif e.type == KEYDOWN:
                
                if e.key == K_ESCAPE:
                    running = False
                    
                elif e.key == K_r:
                    game.reset()
                    viewer.invalidate()
                    
                elif e.key in keyToDifficultyIdx:
                    diffIdx = keyToDifficultyIdx[e.key]
                    game.setDifficulty(difficulty[diffIdx])
                    viewer.invalidate()
                    
                elif e.key == K_z:
                    if diffIdx > 0:
                        diffIdx -= 1
                        game.setDifficulty(difficulty[diffIdx])
                        viewer.invalidate()
                        
                elif e.key == K_x:
                    if diffIdx < len(difficulty)-1:
                        diffIdx += 1
                        game.setDifficulty(difficulty[diffIdx])
                        viewer.invalidate()
                elif e.key == K_q:
                    game.setRandomGameState()
                    viewer.invalidate()
                """
                elif e.key == K_b:
                    randomizeColors()
                elif e.key == K_p:
                    printColors()
                elif e.key == K_m:
                    loadColors(colorFiles[currentColor])
                """
                
            elif e.type == MOUSEBUTTONDOWN:
                # pass position relative to the viewer and the event itself
                viewer.onClick(e.pos, e)
                
        viewer.update()
    pygame.quit()

if __name__ == "__main__":
    main()
