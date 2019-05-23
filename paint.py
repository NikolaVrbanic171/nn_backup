""" paint.py 
    dataset creator"""

import math

import numpy as np
import pygame
from NeuralNetwork import NeuralNetwork

curr_gesta = np.array([])
dataset = np.array([])
M = 20
nn = NeuralNetwork()
zadnja_klasa = None


def euklidskaUdaljenost(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def obradiIPohraniGestu(class_num):
    global curr_gesta, dataset,M
    '''centriranje'''
    sum_x = sum_y = 0
    i = 0
    while i < curr_gesta.size:
        sum_x += curr_gesta[i]
        sum_y += curr_gesta[i+1]
        i += 2

    centar_x = 2*sum_x/curr_gesta.size
    centar_y = 2*sum_y/curr_gesta.size
    i = 0
    while i < curr_gesta.size:
        curr_gesta[i] -= centar_x
        curr_gesta[i+1] -= centar_y
        i += 2


    '''skaliranje'''
    mx = 0
    my = 0
    i = 0
    while i < curr_gesta.size:
        if abs(curr_gesta[i]) > mx: mx = abs(curr_gesta[i])
        if abs(curr_gesta[i+1]) > my: my = abs(curr_gesta[i+1])
        i += 2
    curr_gesta /= max([mx, my])
    '''uzorkovanje'''
    duljina = 0
    obradena_gesta = np.zeros(2*M+5)
    i= 0
    while i < curr_gesta.size-2:
        duljina += euklidskaUdaljenost(curr_gesta[i], curr_gesta[i+1], curr_gesta[i+2], curr_gesta[i+3])
        i += 2
    i = 0
    k = 0
    trenutna_udaljenost = 0
    ciljna_udaljenost = k/2 * duljina / (M - 1)
    while i < curr_gesta.size - 2:
        if trenutna_udaljenost >= ciljna_udaljenost:
            obradena_gesta[k] = curr_gesta[i]
            obradena_gesta[k+1] = curr_gesta[i+1]
            k += 2
            ciljna_udaljenost = k / 2 * duljina / (M - 1)
        trenutna_udaljenost += euklidskaUdaljenost(curr_gesta[i], curr_gesta[i+1], curr_gesta[i+2], curr_gesta[i+3])
        i += 2

    if not k/2 == M:
        obradena_gesta[k] = curr_gesta[-2]
        obradena_gesta[k+1] = curr_gesta[-1]
    obradena_gesta[2*M+class_num] = 1
    curr_gesta = np.array([])
    if dataset.size == 0:
        dataset = obradena_gesta.reshape((1, 2*M+5))
    else:
        dataset = np.concatenate((dataset, obradena_gesta.reshape((1, 2*M+5))), axis=0)
    return obradena_gesta


def checkKeys(myData):
    """test for various keyboard inputs"""
    global curr_gesta, dataset, zadnja_klasa
    # extract the data
    (event, background, drawColor, class_num, keepGoing) = myData
    # print myData

    if event.key == pygame.K_q:
        # quit
        np.savetxt(fname="datasetStjepan.txt", X=dataset, delimiter="\t")
        #keepGoing = False
    elif event.key == pygame.K_c:
        # clear screen
        background.fill((255, 255, 255))
        curr_gesta = np.array([])
    elif event.key == pygame.K_e:
        gesta = obradiIPohraniGestu(1)
        nn.forward_pass(gesta[:-5])
        izlaz = nn.forward_pass(gesta[:-5])[0]
        zadnja_klasa = ispisiRezultat(np.argmax(izlaz))

    elif event.key == pygame.K_s:
        # spremi uzorak
        obradiIPohraniGestu(class_num-1)
        background.fill((255, 255, 255))
    elif event.key == pygame.K_1:
        class_num = 1
    elif event.key == pygame.K_2:
        class_num = 2
    elif event.key == pygame.K_3:
        class_num = 3
    elif event.key == pygame.K_4:
        class_num = 4
    elif event.key == pygame.K_5:
        class_num = 5
    # return all values
    myData = (event, background, drawColor, class_num, keepGoing)
    return myData


def ispisiRezultat(class_num):
    ispis = "Napisali ste znak "
    if class_num == 0:
        ispis += "alfa"
    elif class_num == 1:
        ispis += "beta"
    elif class_num == 2:
        ispis += "gamma"
    elif class_num == 3:
        ispis += "delta"
    elif class_num == 4:
        ispis += "epsilon"
    myFont = pygame.font.SysFont("None", 30)
    return myFont.render(ispis, 1, (1, 0, 0))


def showStats(drawColor, class_num):
    myFont = pygame.font.SysFont("None", 20)
    stats = "Trenutno odabrana klasa: %d" % (class_num)
    statSurf = myFont.render(stats, 1, drawColor)
    return statSurf


def main():
    global curr_gesta, M, nn, zadnja_klasa
    '''Inicijalizacija i treniranje mreže'''
    dataset = np.loadtxt("dataset.txt")
    np.set_printoptions(3)
    input_data = dataset[:, :-5]
    output_data = dataset[:, -5:]
    nn.add_layer(40, 10)
    nn.add_layer(10, 8)
    nn.add_layer(8, 5)
    nn.train(100000, 10, input_data, output_data, 0.025, 1e-4)

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption(
        "(1)-(5) chose class num,(c)lear, (s)ave current, (q)uit and save dataset")

    background = pygame.Surface(screen.get_size())
    background.fill((255, 255, 255))

    #clock = pygame.time.Clock()
    keepGoing = True
    lineStart = (0, 0)
    drawColor = (0, 0, 0)
    lineWidth = 1
    class_num = 1
    while keepGoing:
       # clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keepGoing = False
            elif event.type == pygame.MOUSEMOTION:
                lineEnd = pygame.mouse.get_pos()
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    pygame.draw.line(background, drawColor, lineStart, lineEnd, lineWidth)
                    curr_gesta = np.append(curr_gesta, [lineEnd[0], lineEnd[1]])
                lineStart = lineEnd
            elif event.type == pygame.KEYDOWN:
                myData = (event, background, drawColor, class_num, keepGoing)
                myData = checkKeys(myData)
                (event, background, drawColor, class_num, keepGoing) = myData
        screen.blit(background, (0, 0))
        myLabel = showStats(drawColor, class_num)
        screen.blit(myLabel, (450, 450))
        if zadnja_klasa != None:
            screen.blit(zadnja_klasa, (200, 400))
        pygame.display.flip()


if __name__ == "__main__":
    main()
