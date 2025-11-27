from __future__ import annotations

import os
from collections import deque

import chess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from constants import CHECKPOINT_PATH, LAST_POSITIONS
from db import SessionLocal, Game, Move, create_tables
from env import board_to_planes
from nw import CNNActorCritic
from agent import MCTS


app = FastAPI(
    title="Chess Agent Web API",
    description="Игра в шахматы против агента. Есть web UI, API, хранение в БД.",
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def _startup():
    create_tables()


# -------- model / mcts --------
def _pick_device() -> str:
    dev = os.getenv("DEVICE", "cpu")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return dev


device = _pick_device()
model = CNNActorCritic().to(device)

if os.path.exists(CHECKPOINT_PATH):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
else:
    # чтобы сервис запускался даже без весов
    print(f"[WARN] checkpoint not found: {CHECKPOINT_PATH}. Using random weights.")
    model.eval()

WEB_MCTS_SIMULATIONS = int(os.getenv("WEB_MCTS_SIMULATIONS", "256"))
mcts = MCTS(number_of_simulations=WEB_MCTS_SIMULATIONS, model=model, device=device)


# -------- schemas --------
class NewGameRequest(BaseModel):
    human_color: str = "w"  # "w" or "b"


class MoveRequest(BaseModel):
    game_id: str
    from_square: str
    to_square: str
    promotion: str | None = None  # "q/r/b/n" or None


class EngineMoveRequest(BaseModel):
    game_id: str


class MoveResponse(BaseModel):
    game_id: str
    board_fen: str
    engine_move: str | None
    moves_san: list[str]
    game_over: bool
    result: str | None
    termination: str | None


# -------- helpers --------
def _promotion_piece(p: str | None):
    if not p:
        return None
    promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
    out = promo_map.get(p.lower())
    if out is None:
        raise HTTPException(status_code=400, detail="Некорректная фигура для превращения (q/r/b/n)")
    return out


def _outcome_info(board: chess.Board):
    if not board.is_game_over():
        return False, None, None
    outcome = board.outcome()
    result = board.result()
    termination = outcome.termination.name if outcome is not None else None
    return True, result, termination


def _load_board_and_hist(db, game: Game):
    """
    Восстанавливаем board и историю последних позиций из БД ходов.
    Это делает сервис stateless (не зависит от глобальной board/position_deque).
    """
    rows = (
        db.query(Move)
        .filter(Move.game_id == game.id)
        .order_by(Move.ply)
        .all()
    )

    board = chess.Board()
    hist = deque(maxlen=LAST_POSITIONS)
    hist.append(board_to_planes(board))

    for r in rows:
        mv = chess.Move.from_uci(r.uci)
        if mv not in board.legal_moves:
            raise HTTPException(status_code=500, detail=f"БД содержит нелегальный ход: {r.uci}")
        board.push(mv)
        hist.append(board_to_planes(board))

    return board, hist, rows


def _moves_san(rows: list[Move]) -> list[str]:
    return [r.san for r in rows]


def _get_game(db, game_id: str) -> Game:
    g = db.get(Game, game_id)
    if g is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return g


def _human_color_bool(human_color: str) -> bool:
    if human_color not in ("w", "b"):
        raise HTTPException(status_code=400, detail="human_color must be 'w' or 'b'")
    return chess.WHITE if human_color == "w" else chess.BLACK


def _save_move(db, game: Game, board: chess.Board, mv: chess.Move):
    # SAN надо брать ДО push
    san = board.san(mv)
    ply = len(board.move_stack) + 1
    board.push(mv)
    db.add(Move(game_id=game.id, ply=ply, uci=mv.uci(), san=san))


# -------- endpoints --------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/new_game", response_model=MoveResponse)
def new_game(req: NewGameRequest):
    human_color = req.human_color if req.human_color in ("w", "b") else "w"
    human_bool = _human_color_bool(human_color)

    with SessionLocal() as db:
        board = chess.Board()
        g = Game(human_color=human_color, fen=board.fen(), status="ongoing", result=None)
        db.add(g)
        db.commit()
        db.refresh(g)

        engine_move_uci = None

        # Если человек играет чёрными — движок делает первый ход сразу
        if human_bool == chess.BLACK:
            board_hist = deque(maxlen=LAST_POSITIONS)
            board_hist.append(board_to_planes(board))

            mv, _policy = mcts.run(board, position_history=board_hist)
            if mv is None:
                raise HTTPException(status_code=500, detail="Движок не смог выбрать ход (MCTS вернул None)")
            _save_move(db, g, board, mv)
            engine_move_uci = mv.uci()

        game_over, result, termination = _outcome_info(board)
        g.fen = board.fen()
        if game_over:
            g.status = "finished"
            g.result = result
        db.commit()

        rows = (
            db.query(Move)
            .filter(Move.game_id == g.id)
            .order_by(Move.ply)
            .all()
        )

        return MoveResponse(
            game_id=g.id,
            board_fen=board.fen(),
            engine_move=engine_move_uci,
            moves_san=_moves_san(rows),
            game_over=game_over,
            result=result,
            termination=termination,
        )


@app.post("/make_move", response_model=MoveResponse)
def make_move(req: MoveRequest):
    with SessionLocal() as db:
        g = _get_game(db, req.game_id)

        board, hist, rows = _load_board_and_hist(db, g)
        if board.is_game_over():
            game_over, result, termination = _outcome_info(board)
            return MoveResponse(
                game_id=g.id,
                board_fen=board.fen(),
                engine_move=None,
                moves_san=_moves_san(rows),
                game_over=game_over,
                result=result,
                termination=termination,
            )

        human_bool = _human_color_bool(g.human_color)
        if board.turn != human_bool:
            raise HTTPException(status_code=400, detail="Сейчас не ход человека")

        # parse move
        try:
            from_sq = chess.parse_square(req.from_square)
            to_sq = chess.parse_square(req.to_square)
        except ValueError:
            raise HTTPException(status_code=400, detail="Некорректный формат клетки")

        promo = _promotion_piece(req.promotion)

        move_plain = chess.Move(from_sq, to_sq)
        move_promo = chess.Move(from_sq, to_sq, promotion=promo) if promo is not None else None

        if move_plain in board.legal_moves:
            user_move = move_plain
        elif move_promo is not None and move_promo in board.legal_moves:
            user_move = move_promo
        else:
            raise HTTPException(status_code=400, detail="Невозможный ход")

        # save user move
        _save_move(db, g, board, user_move)
        hist.append(board_to_planes(board))

        # если после хода человека игра закончилась
        if board.is_game_over():
            game_over, result, termination = _outcome_info(board)
            g.fen = board.fen()
            g.status = "finished"
            g.result = result
            db.commit()

            rows = (
                db.query(Move)
                .filter(Move.game_id == g.id)
                .order_by(Move.ply)
                .all()
            )
            return MoveResponse(
                game_id=g.id,
                board_fen=board.fen(),
                engine_move=None,
                moves_san=_moves_san(rows),
                game_over=game_over,
                result=result,
                termination=termination,
            )

        # ход движка
        engine_move, _policy = mcts.run(board, position_history=hist)
        if engine_move is None:
            raise HTTPException(status_code=500, detail="Движок не смог выбрать ход (MCTS вернул None)")

        _save_move(db, g, board, engine_move)
        hist.append(board_to_planes(board))

        game_over, result, termination = _outcome_info(board)

        g.fen = board.fen()
        if game_over:
            g.status = "finished"
            g.result = result
        else:
            g.status = "ongoing"
            g.result = None

        db.commit()

        rows = (
            db.query(Move)
            .filter(Move.game_id == g.id)
            .order_by(Move.ply)
            .all()
        )

        return MoveResponse(
            game_id=g.id,
            board_fen=board.fen(),
            engine_move=engine_move.uci(),
            moves_san=_moves_san(rows),
            game_over=game_over,
            result=result,
            termination=termination,
        )


@app.post("/engine_move", response_model=MoveResponse)
def engine_move(req: EngineMoveRequest):
    with SessionLocal() as db:
        g = _get_game(db, req.game_id)
        board, hist, rows = _load_board_and_hist(db, g)

        if board.is_game_over():
            game_over, result, termination = _outcome_info(board)
            return MoveResponse(
                game_id=g.id,
                board_fen=board.fen(),
                engine_move=None,
                moves_san=_moves_san(rows),
                game_over=game_over,
                result=result,
                termination=termination,
            )

        human_bool = _human_color_bool(g.human_color)
        if board.turn == human_bool:
            raise HTTPException(status_code=400, detail="Сейчас ход человека, движок ходить не должен")

        mv, _policy = mcts.run(board, position_history=hist)
        if mv is None:
            raise HTTPException(status_code=500, detail="Движок не смог выбрать ход (MCTS вернул None)")

        _save_move(db, g, board, mv)
        hist.append(board_to_planes(board))

        game_over, result, termination = _outcome_info(board)
        g.fen = board.fen()
        if game_over:
            g.status = "finished"
            g.result = result
        db.commit()

        rows = (
            db.query(Move)
            .filter(Move.game_id == g.id)
            .order_by(Move.ply)
            .all()
        )

        return MoveResponse(
            game_id=g.id,
            board_fen=board.fen(),
            engine_move=mv.uci(),
            moves_san=_moves_san(rows),
            game_over=game_over,
            result=result,
            termination=termination,
        )


@app.get("/games/{game_id}")
def get_game(game_id: str):
    with SessionLocal() as db:
        g = _get_game(db, game_id)
        return {
            "game_id": g.id,
            "status": g.status,
            "human_color": g.human_color,
            "result": g.result,
            "fen": g.fen,
            "created_at": g.created_at.isoformat(),
        }


@app.get("/games/{game_id}/moves")
def get_moves(game_id: str):
    with SessionLocal() as db:
        _ = _get_game(db, game_id)
        rows = (
            db.query(Move)
            .filter(Move.game_id == game_id)
            .order_by(Move.ply)
            .all()
        )
        return [{"ply": m.ply, "uci": m.uci, "san": m.san, "ts": m.created_at.isoformat()} for m in rows]


# -------- Web UI --------
@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Шахматы с агентом</title>
    <link rel="stylesheet" href="/static/css/chessboard-1.0.0.min.css">
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .top-bar { display:flex; gap:10px; align-items:center; }
        .layout { display:flex; gap:20px; align-items:flex-start; margin-top:20px; }
        #board { width: 480px; }
        #status { margin-top: 10px; min-height: 20px; }
        #fen { margin-top: 10px; background:#f5f5f5; padding:5px; font-family: monospace; }
        .moves-panel { min-width: 240px; font-size: 14px; }
        .moves-panel table { border-collapse: collapse; width: 100%; }
        .moves-panel th, .moves-panel td { padding: 2px 4px; border-bottom: 1px solid #ddd; text-align:left; }
        .moves-panel th { font-weight: 600; }
    </style>
</head>
<body>
    <h1>Шахматы с агентом</h1>

    <div class="top-bar">
        <button onclick="newGame()">Новая партия</button>
        <label>
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

    <script src="/static/js/jquery-3.7.1.min.js"></script>
    <script src="/static/js/chess.min.js"></script>
    <script src="/static/js/chessboard-1.0.0.min.js"></script>

    <script>
        let boardUI = null;
        let game = null;
        let gameId = null;

        let isMyTurn = true;
        let moves = [];
        let gameResult = null;
        let termination = null;
        let humanColor = "w";

        function setStatus(msg, isError) {
            const el = document.getElementById('status');
            el.textContent = msg || "";
            el.style.color = isError ? "red" : "black";
        }

        function updateFen() {
            const fenEl = document.getElementById('fen');
            fenEl.textContent = game ? game.fen() : "";
        }

        function renderMoveList() {
            const container = document.getElementById('moves');
            if (!container) return;

            if (!moves || moves.length === 0) {
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
                const extra = termination ? (" (" + termination + ")") : "";
                html += "<div style='margin-top:6px;font-weight:700;'>Результат: " + gameResult + extra + "</div>";
            }

            container.innerHTML = html;
        }

        function onDragStart(source, piece) {
            if (!game) return false;
            if (game.game_over()) return false;
            if (!isMyTurn) return false;

            const isWhitePiece = piece[0] === "w";
            const isBlackPiece = piece[0] === "b";
            if (humanColor === "w" && isBlackPiece) return false;
            if (humanColor === "b" && isWhitePiece) return false;
        }

        function onDrop(source, target) {
          if (!game || !gameId) return "snapback";
          if (source === target) return "snapback";
        
          // базовый ход БЕЗ promotion
          let moveConfig = { from: source, to: target };
        
          // promotion prompt
          const piece = game.get(source);
          let promo = null;
        
          if (piece && piece.type === "p") {
            const fromRank = source[1], toRank = target[1];
            const isWhitePromotion = (piece.color === "w" && fromRank === "7" && toRank === "8");
            const isBlackPromotion = (piece.color === "b" && fromRank === "2" && toRank === "1");
            if (isWhitePromotion || isBlackPromotion) {
              promo = window.prompt("Превращение (q, r, b, n):", "q");
              if (!promo) promo = "q";
              promo = promo.toLowerCase();
              if (!["q","r","b","n"].includes(promo)) promo = "q";
              moveConfig.promotion = promo; // добавляем ТОЛЬКО здесь
            }
          }
        
          // локальная проверка chess.js
          const mv = game.move(moveConfig);
          if (mv === null) return "snapback";
        
          boardUI.position(game.fen());
          updateFen();
        
          isMyTurn = false;
          sendMoveToServer(source, target, promo); // promo может быть null
        }

        async function sendMoveToServer(from, to, promotion) {
            try {
                const res = await fetch("/make_move", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        game_id: gameId,
                        from_square: from,
                        to_square: to,
                        promotion: promotion || null
                    })
                });

                const data = await res.json();

                if (!res.ok) {
                    const msg = data.detail || JSON.stringify(data);
                    setStatus("Ошибка сервера: " + msg, true);
                    // откат локального хода
                    game.undo();
                    boardUI.position(game.fen());
                    updateFen();
                    isMyTurn = true;
                    return;
                }

                gameId = data.game_id;
                game.load(data.board_fen);
                boardUI.position(game.fen());
                updateFen();

                moves = data.moves_san || [];
                gameResult = data.result || null;
                termination = data.termination || null;
                renderMoveList();

                if (data.game_over) {
                    setStatus("Игра окончена. Результат: " + data.result + (data.termination ? (" ("+data.termination+")") : ""), false);
                    isMyTurn = false;
                } else {
                    isMyTurn = true;
                    setStatus("Ход принят. Движок ответил: " + (data.engine_move || "?"), false);
                }
            } catch (err) {
                setStatus("Ошибка при запросе к серверу: " + err.message, true);
                game.undo();
                boardUI.position(game.fen());
                updateFen();
                isMyTurn = true;
            }
        }

        async function newGame() {
            setStatus("", false);
            try {
                const select = document.getElementById("colorSelect");
                humanColor = select ? select.value : "w";

                const res = await fetch("/new_game", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ human_color: humanColor })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));

                if (!boardUI) {
                    boardUI = Chessboard("board", {
                        draggable: true,
                        position: "start",
                        orientation: "white",
                        onDragStart: onDragStart,
                        onDrop: onDrop,
                        pieceTheme: "/static/img/chesspieces/wikipedia/{piece}.png"
                    });
                }

                gameId = data.game_id;
                boardUI.orientation(humanColor === "w" ? "white" : "black");

                game = new Chess(data.board_fen);
                boardUI.position(game.fen());
                updateFen();

                moves = data.moves_san || [];
                gameResult = data.result || null;
                termination = data.termination || null;
                renderMoveList();

                // после /new_game, если человек чёрными — движок уже сделал первый ход
                isMyTurn = true;

                if (humanColor === "w") {
                    setStatus("Новая партия начата. Ты играешь белыми.", false);
                } else {
                    setStatus("Новая партия начата. Ты играешь чёрными. Движок уже сделал первый ход.", false);
                }
            } catch (err) {
                setStatus("Ошибка при создании новой партии: " + err.message, true);
            }
        }
    </script>
</body>
</html>"""
