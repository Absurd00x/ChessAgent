import chess

board = chess.Board()
print(board)

move = chess.Move.from_uci("e2e5")
board.push(move)
print()
print(board)

# TODO:
# 1. Поляк авераджинг для таргет и онлайн сетей
# 2. Отображение в браузере с возможностью отмены хода
# 3. Сохранение моделей на диск
# 4. Testing pycharm internal commit system