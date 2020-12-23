

# state can be the two scores, can change maxCount to see how many moves ahead
# you want to look.

maxCount = 3
def maxi(state, count):
    if count>maxCount:
        return (None, None)
    bestScore = None
    bestMove = None
    for move in moves:
        newState = makeMove(move)
        newState, newScore = mini(state, count+1)
        if (newState, newScore)==(None, None):
            continue
        if bestScore==None or score>bestScore:
                bestScore = newScore
                bestMove = move
    return bestMove, bestScore
def mini(state, count):
    if count>maxCount:
        return (None, None)
    bestScore = None
    bestMove = None
    for move in moves:
        state = makeMove(move)
        newState, newScore = maxi(newState, count+1)
        if (newState, newScore)==(None, None):
            continue
        if(bestScore==None or score<bestScore):
            bestScore = newScore
            bestMove = move
    return bestMove
def minimax(state):
    if gameover:
        return
    #last move is the player's last move
    return maxi(state, 0)
    
