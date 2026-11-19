import os
import json
import logging
import datetime as dt

import asyncpg
from anthropic import AsyncAnthropic
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agente-tareas")

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
client = AsyncAnthropic()  # usa ANTHROPIC_API_KEY
pool: asyncpg.Pool | None = None
history: dict[int, list[dict]] = {}  # chat_id -> turnos de texto recientes
MAX_TURNS = 20

# Opcional: IDs de Telegram permitidos, separados por coma (evita que otros gasten tu API)
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}

SYSTEM = """Eres un asistente que gestiona las tareas del usuario por Telegram.
Responde en español, breve y claro (se lee en un móvil). Usa las herramientas para
crear, listar, actualizar, completar y borrar tareas; nunca inventes tareas ni IDs.
Si falta un dato imprescindible, pregunta una sola cosa. Prioridad: 1 alta, 2 media, 3 baja.
Fechas en formato AAAA-MM-DD. Hoy es {today}."""

TOOLS = [
    {
        "name": "add_task",
        "description": "Crea una tarea nueva.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "due_date": {"type": "string", "description": "AAAA-MM-DD"},
                "priority": {"type": "integer", "enum": [1, 2, 3]},
                "notes": {"type": "string"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "list_tasks",
        "description": "Lista las tareas, ordenadas por prioridad y fecha.",
        "input_schema": {
            "type": "object",
            "properties": {"include_done": {"type": "boolean"}},
        },
    },
    {
        "name": "update_task",
        "description": "Modifica título, fecha, prioridad o notas de una tarea.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "title": {"type": "string"},
                "due_date": {"type": "string", "description": "AAAA-MM-DD"},
                "priority": {"type": "integer", "enum": [1, 2, 3]},
                "notes": {"type": "string"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "complete_task",
        "description": "Marca una tarea como completada.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "integer"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "delete_task",
        "description": "Elimina una tarea de forma definitiva.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "integer"}},
            "required": ["task_id"],
        },
    },
]


async def init_db(app):
    global pool
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"])
    async with pool.acquire() as c:
        await c.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                title TEXT NOT NULL,
                notes TEXT,
                due_date DATE,
                priority INT NOT NULL DEFAULT 2,
                done BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


def parse_date(s):
    return dt.date.fromisoformat(s) if s else None


async def run_tool(chat_id: int, name: str, args: dict) -> str:
    async with pool.acquire() as c:
        if name == "add_task":
            row = await c.fetchrow(
                "INSERT INTO tasks (chat_id, title, notes, due_date, priority) "
                "VALUES ($1,$2,$3,$4,$5) RETURNING id",
                chat_id,
                args["title"],
                args.get("notes"),
                parse_date(args.get("due_date")),
                args.get("priority", 2),
            )
            return json.dumps({"ok": True, "id": row["id"]})

        if name == "list_tasks":
            q = "SELECT id, title, notes, due_date, priority, done FROM tasks WHERE chat_id=$1"
            if not args.get("include_done"):
                q += " AND done = FALSE"
            q += " ORDER BY priority, due_date NULLS LAST, id"
            rows = await c.fetch(q, chat_id)
            return json.dumps([dict(r) for r in rows], default=str, ensure_ascii=False)

        if name == "update_task":
            fields, vals = [], [chat_id, args["task_id"]]
            for key in ("title", "notes", "priority"):
                if key in args:
                    vals.append(args[key])
                    fields.append(f"{key}=${len(vals)}")
            if "due_date" in args:
                vals.append(parse_date(args["due_date"]))
                fields.append(f"due_date=${len(vals)}")
            if not fields:
                return json.dumps({"ok": False, "error": "nada que actualizar"})
            res = await c.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE chat_id=$1 AND id=$2", *vals
            )
            return json.dumps({"ok": res.endswith("1")})

        if name == "complete_task":
            res = await c.execute(
                "UPDATE tasks SET done=TRUE WHERE chat_id=$1 AND id=$2",
                chat_id, args["task_id"],
            )
            return json.dumps({"ok": res.endswith("1")})

        if name == "delete_task":
            res = await c.execute(
                "DELETE FROM tasks WHERE chat_id=$1 AND id=$2", chat_id, args["task_id"]
            )
            return json.dumps({"ok": res.endswith("1")})

    return json.dumps({"ok": False, "error": "herramienta desconocida"})


async def ask_claude(chat_id: int, text: str) -> str:
    turns = history.setdefault(chat_id, [])
    messages = turns + [{"role": "user", "content": text}]
    system = SYSTEM.format(today=dt.date.today().isoformat())

    for _ in range(8):  # límite de rondas de herramientas
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text").strip()
            break
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                try:
                    out = await run_tool(chat_id, b.name, b.input)
                except Exception as e:  # el modelo ve el error y puede corregir
                    log.exception("tool error")
                    out = json.dumps({"ok": False, "error": str(e)})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})
    else:
        answer = "No pude terminar esa acción, inténtalo de nuevo."

    answer = answer or "Listo."
    turns += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
    del turns[:-MAX_TURNS]
    return answer


def allowed(update: Update) -> bool:
    return not ALLOWED or update.effective_user.id in ALLOWED


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(
        "Hola, soy tu asistente de tareas. Escríbeme cosas como:\n"
        "• Añade comprar leche mañana\n"
        "• ¿Qué tengo pendiente?\n"
        "• Marca la 3 como hecha"
    )


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update) or not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    await ctx.bot.send_chat_action(chat_id, ChatAction.TYPING)
    try:
        answer = await ask_claude(chat_id, update.message.text)
    except Exception:
        log.exception("error")
        answer = "Algo falló, inténtalo otra vez en un momento."
    await update.message.reply_text(answer)


def main():
    app = (
        ApplicationBuilder()
        .token(os.environ["TELEGRAM_BOT_TOKEN"])
        .post_init(init_db)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling()


if __name__ == "__main__":
    main()
