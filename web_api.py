from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chess
import torch

from nw import CNNActorCritic
from agent import MCTS, position_deque
from env import board_to_planes
from constants import CHECKPOINT_PATH


app = FastAPI(
    title="Chess AlphaZero-like API",
    description="Простой API для игры против шахматного агента",
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Глобальные объекты

# Одна доска на всё приложение
board: chess.Board | None = None

device = "cpu"
model = CNNActorCritic().to(device)

try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
except FileNotFoundError:
    raise RuntimeError(
        f"Не найден файл с весами по пути {CHECKPOINT_PATH}. "
        f"Убедись, что модель обучена и файл существует."
    )

mcts = MCTS(number_of_simulations=1024, model=model, device=device)


class MoveRequest(BaseModel):
    from_square: str
    to_square: str
    promotion: str | None = None

class MoveResponse(BaseModel):
    board_fen: str
    engine_move: str | None
    game_over: bool
    result: str | None


def _ensure_game_started():
    if board is None:
        raise HTTPException(status_code=400, detail="Игра ещё не начата. Вызови /new_game.")

@app.post("/new_game", response_model=MoveResponse)
def new_game():
    global board
    board = chess.Board()

    position_deque.clear()

    return MoveResponse(
        board_fen=board.fen(),
        engine_move=None,
        game_over=False,
        result=None,
    )


@app.post("/make_move", response_model=MoveResponse)
def make_move(req: MoveRequest):
    global board
    _ensure_game_started()
    assert board is not None

    # Ход человека

    try:
        from_sq = chess.parse_square(req.from_square)
        to_square = chess.parse_square(req.to_square)
    except ValueError:
        raise HTTPException(status_code=400, detail="Некорректный формат клетки")

    # Превращение
    promotion_piece = None
    if req.promotion:
        promo_map = {
            'q': chess.QUEEN,
            'r': chess.ROOK,
            'b': chess.BISHOP,
            'n': chess.KNIGHT,
        }
        promotion_piece = promo_map.get(req.promotion.lower())
        if promotion_piece is None:
            raise HTTPException(status_code=400, detail="Некорректная фигура для превращения")

    user_move = chess.Move(from_sq, to_square, promotion=promotion_piece)

    if user_move not in board.legal_moves:
        raise HTTPException(status_code=400, detail="Невозможный ход.")

    board.push(user_move)

    position_deque.append(board_to_planes(board))

    if board.is_game_over():
        return MoveResponse(
            board_fen=board.fen(),
            engine_move=None,
            game_over=True,
            result=board.result(),
        )

    engine_move, policy = mcts.run(board)
    if engine_move is None:
        raise HTTPException(
            status_code=500,
            detail="Движок не смог выбрать ход, MCTS вернул None. Проверь логи сервера"
        )

    board.push(engine_move)
    position_deque.append(board_to_planes(board))

    game_over = board.is_game_over()
    result = board.result() if game_over else None

    return MoveResponse(
        board_fen=board.fen(),
        engine_move=engine_move.uci(),
        game_over=game_over,
        result=result,
    )

@app.post("/engine_move", response_model=MoveResponse)
def engine_move():
    global board
    _ensure_game_started()
    assert board is not None

    if board.is_game_over():
        return MoveResponse(
            board_fen=board.fen(),
            engine_move=None,
            game_over=True,
            result=board.result(),
        )

    engine_move, policy = mcts.run(board)
    if engine_move is None:
        raise HTTPException(
            status_code=500,
            detail="Движок не смог выбрать ход, MCTS вернул None. Проверь логи сервера"
        )

    board.push(engine_move)
    position_deque.append(board_to_planes(board))

    game_over = board.is_game_over()
    result = board.result() if game_over else None

    return MoveResponse(
        board_fen=board.fen(),
        engine_move=engine_move.uci(),
        game_over=game_over,
        result=result,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Шахматы с агентом</title>

    <!-- chessboard.js CSS -->
    <link rel="stylesheet" href="/static/css/chessboard-1.0.0.min.css">

    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 20px;
            display: flex;
            flex-direction: column;
            align-items: center; /* центрируем всё по горизонтали */
        }

        h1 {
            margin-bottom: 10px;
        }
        .top-bar {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .layout {
            display: flex;
            gap: 20px;
            align-items: flex-start;
            margin-top: 20px;
        }
        #board {
            width: 480px;
        }
        #status {
            margin-top: 10px;
            min-height: 20px;
        }
        #fen {
            margin-top: 10px;
            background: #f5f5f5;
            padding: 5px;
            font-family: monospace;
        }
        .moves-panel {
            min-width: 200px;
            font-size: 14px;
        }
        .moves-panel h3 {
            margin: 0 0 8px 0;
        }
        .moves-panel table {
            border-collapse: collapse;
            width: 100%;
        }
        .moves-panel th,
        .moves-panel td {
            padding: 2px 4px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        .moves-panel th {
            font-weight: 600;
        }
    </style>
</head>
<body>
    <h1>Шахматы с агентом</h1>

    <div class="top-bar">
        <button onclick="newGame()">Новая партия</button>
        <label style="margin-left: 10px;">
            Цвет:
            <select id="colorSelect">
                <option value="w">Белыми</option>
                <option value="b">Чёрными</option>
            </select>
        </label>
        <span id="status"></span>
    </div>

    <div class="layout">
        <div>
            <div id="board"></div>
            <pre id="fen"></pre>
        </div>
        <div id="moves" class="moves-panel">
            <h3>Ходы</h3>
            <p>Пока ходов нет.</p>
        </div>
    </div>

    <!-- Локальные библиотеки -->
    <script src="/static/js/jquery-3.7.1.min.js"></script>
    <script src="/static/js/chess.min.js"></script>
    <script src="/static/js/chessboard-1.0.0.min.js"></script>

    <script>
        let board = null;      // объект доски (chessboard.js)
        let game = null;       // логика игры (chess.js)
        let isMyTurn = true;   // ход человека?
        let moves = [];        // список ходов в SAN
        let gameResult = null; // строка результата: "1-0", "0-1" или "1/2-1/2"
        let humanColor = "w";  // "w" — играем белыми, "b" — чёрными

        function setStatus(msg, isError) {
            const el = document.getElementById('status');
            el.textContent = msg || "";
            el.style.color = isError ? "red" : "black";
        }

        function updateFen() {
            const fenEl = document.getElementById('fen');
            if (game) {
                fenEl.textContent = game.fen();
            } else {
                fenEl.textContent = "";
            }
        }

        function renderMoveList() {
            const container = document.getElementById('moves');
            if (!container) return;

            if (moves.length === 0) {
                container.innerHTML = "<h3>Ходы</h3><p>Пока ходов нет.</p>";
                return;
            }

            let html = "<h3>Ходы</h3><table>";
            html += "<tr><th>#</th><th>Белые</th><th>Чёрные</th></tr>";

            for (let i = 0; i < moves.length; i += 2) {
                const moveNum = (i / 2) + 1;
                const white = moves[i] || "";
                const black = moves[i + 1] || "";
                html += "<tr><td>" + moveNum + "</td><td>" + white + "</td><td>" + black + "</td></tr>";
            }

            html += "</table>";

            if (gameResult) {
                html += "<div style='margin-top:4px;font-weight:600;'>Результат: " + gameResult + "</div>";
            }

            container.innerHTML = html;
        }


        function resetMoveList() {
            moves = [];
            gameResult = null;
            renderMoveList();
        }

        function addMoveToList(san) {
            moves.push(san);
            renderMoveList();
        }

        function onDragStart(source, piece, position, orientation) {
            // Нельзя двигать фигуры, если игра не начата
            if (!game) return false;

            // Игра уже закончена
            if (game.game_over()) return false;

            // Сейчас не наш ход
            if (!isMyTurn) return false;

            const isWhitePiece = piece[0] === "w";
            const isBlackPiece = piece[0] === "b";

            // Человек играет этим цветом, другим — нельзя
            if (humanColor === "w" && isBlackPiece) return false;
            if (humanColor === "b" && isWhitePiece) return false;
        }

        function onDrop(source, target) {
            if (!game) return "snapback";
            if (source === target) return "snapback";

            let moveConfig = {
                from: source,
                to: target,
                promotion: "q"  // по умолчанию превращаем в ферзя
            };

            // Проверка на promotion (превращение пешки)
            const piece = game.get(source);
            if (piece && piece.type === "p") {
                const fromRank = source[1];
                const toRank = target[1];

                const isWhitePromotion = (piece.color === "w" && fromRank === "7" && toRank === "8");
                const isBlackPromotion = (piece.color === "b" && fromRank === "2" && toRank === "1");

                if (isWhitePromotion || isBlackPromotion) {
                    let p = window.prompt("Превращение (q, r, b, n):", "q");
                    if (!p) p = "q";
                    p = p.toLowerCase();
                    if (!["q", "r", "b", "n"].includes(p)) {
                        p = "q";
                    }
                    moveConfig.promotion = p;
                }
            }

            // Пробуем сделать ход локально (chess.js)
            const move = game.move(moveConfig);

            // Нелегальный ход — откат
            if (move === null) {
                return "snapback";
            }

            // Локально ход ок — обновляем доску
            board.position(game.fen());
            updateFen();

            // Добавляем ход белых в список (SAN)
            if (move.san) {
                addMoveToList(move.san);
            }

            const fenBeforeEngine = game.fen();  // позиция перед ходом движка
            isMyTurn = false;

            // Отправляем ход на сервер (человек всегда белыми)
            sendMoveToServer(move.from, move.to, move.promotion, fenBeforeEngine);
        }

        async function sendMoveToServer(from, to, promotion, fenBeforeEngine) {
            try {
                const res = await fetch("/make_move", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        from_square: from,
                        to_square: to,
                        promotion: promotion || null
                    })
                });

                const data = await res.json();

                if (!res.ok) {
                    // Сервер сказал, что ход невозможен/ошибка
                    const msg = data.detail || JSON.stringify(data);
                    setStatus("Ошибка сервера: " + msg, true);

                    // Откатываем последний ход игрока
                    game.undo();
                    board.position(game.fen());
                    updateFen();
                    isMyTurn = true;
                    return;
                }

                // Считаем SAN хода движка по позиции до его хода
                let engineSan = null;
                if (data.engine_move) {
                    const uci = data.engine_move; // например "e7e5" или "e7e8q"
                    const fromSq = uci.slice(0, 2);
                    const toSq = uci.slice(2, 4);
                    const promo = uci.length > 4 ? uci[4] : undefined;

                    try {
                        const tmp = new Chess(fenBeforeEngine);
                        const engineMoveObj = tmp.move({
                            from: fromSq,
                            to: toSq,
                            promotion: promo
                        });
                        if (engineMoveObj && engineMoveObj.san) {
                            engineSan = engineMoveObj.san;
                        } else {
                            engineSan = uci;
                        }
                    } catch (e) {
                        engineSan = uci;
                    }
                }

                // Сервер — источник истины, позицию берём из его FEN
                game.load(data.board_fen);
                board.position(game.fen());
                updateFen();

                // Добавляем ход чёрных в список
                if (engineSan) {
                    addMoveToList(engineSan);
                }

                if (data.game_over) {
                    gameResult = data.result;  // "1-0", "0-1" или "1/2-1/2"
                    renderMoveList();          // перерисуем список с результатом
                    setStatus("Игра окончена. Результат: " + data.result, false);
                    isMyTurn = false;
                } else {
                    setStatus("Ход принят. Движок ответил: " + (data.engine_move || "?"), false);
                    isMyTurn = true;
                }
            } catch (err) {
                setStatus("Ошибка при запросе к серверу: " + err.message, true);
                // Откат хода на всякий случай
                game.undo();
                board.position(game.fen());
                updateFen();
                isMyTurn = true;
            }
        }
                async function requestEngineMove(fenBeforeEngine) {
            try {
                const res = await fetch("/engine_move", { method: "POST" });
                const data = await res.json();

                if (!res.ok) {
                    const msg = data.detail || JSON.stringify(data);
                    setStatus("Ошибка сервера при ходе движка: " + msg, true);
                    isMyTurn = true;
                    return;
                }

                let engineSan = null;
                if (data.engine_move) {
                    const uci = data.engine_move;
                    const fromSq = uci.slice(0, 2);
                    const toSq = uci.slice(2, 4);
                    const promo = uci.length > 4 ? uci[4] : undefined;

                    try {
                        const tmp = new Chess(fenBeforeEngine);
                        const engineMoveObj = tmp.move({
                            from: fromSq,
                            to: toSq,
                            promotion: promo
                        });
                        if (engineMoveObj && engineMoveObj.san) {
                            engineSan = engineMoveObj.san;
                        } else {
                            engineSan = uci;
                        }
                    } catch (e) {
                        engineSan = uci;
                    }
                }

                game.load(data.board_fen);
                board.position(game.fen());
                updateFen();

                if (engineSan) {
                    addMoveToList(engineSan);
                }

                if (data.game_over) {
                    gameResult = data.result;
                    renderMoveList();
                    setStatus("Игра окончена. Результат: " + data.result, false);
                    isMyTurn = false;
                } else {
                    isMyTurn = true;
                    setStatus(
                        "Ход движка выполнен: " + (data.engine_move || "?") + ". Теперь твой ход.",
                        false
                    );
                }
            } catch (err) {
                setStatus("Ошибка при ходе движка: " + err.message, true);
                isMyTurn = true;
            }
        }

        async function newGame() {
            setStatus("", false);
            try {
                const res = await fetch("/new_game", { method: "POST" });
                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.detail || ("HTTP " + res.status));
                }

                // Если доска ещё не создана — создаём её здесь
                if (!board) {
                    const config = {
                        draggable: true,
                        position: "start",
                        orientation: "white",
                        onDragStart: onDragStart,
                        onDrop: onDrop,
                        pieceTheme: "/static/img/chesspieces/wikipedia/{piece}.png"
                    };
                    board = Chessboard("board", config);
                }

                // читаем выбранный цвет
                const select = document.getElementById("colorSelect");
                humanColor = select ? select.value : "w";

                // разворачиваем доску в нужную сторону
                board.orientation(humanColor === "w" ? "white" : "black");

                // Создаём новую игру из стартового FEN, который вернул сервер
                game = new Chess(data.board_fen);
                board.position(game.fen());
                updateFen();
                resetMoveList();

                if (humanColor === "w") {
                    isMyTurn = true;
                    setStatus("Новая партия начата. Ты играешь белыми.", false);
                } else {
                    // мы играем чёрными — движок делает первый ход
                    isMyTurn = false;
                    setStatus("Новая партия начата. Ты играешь чёрными. Движок делает первый ход...", false);
                    const fenBeforeEngine = game.fen();
                    await requestEngineMove(fenBeforeEngine);
                }
            } catch (err) {
                setStatus("Ошибка при создании новой партии: " + err.message, true);
            }
        }

    </script>
</body>
</html>"""
