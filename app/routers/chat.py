"""Endpoints to ask questions about a repo and read past conversations."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import CurrentUser, current_user, get_repo_or_404, owned_by, owner_id
from app.db import Conversation, Message, Repo, get_db
from app.demo import limit
from app.llm import embedding_identity
from app.rag import MAX_HISTORY_MESSAGES, stream_answer, to_chat_history
from app.schemas import AskRequest, ConversationOut, MessageOut

router = APIRouter(tags=["chat"])

MAX_TITLE_LENGTH = 80


def require_searchable(repo: Repo) -> None:
    """Raise a 409 unless the repo is indexed, with the embedding model in use now."""
    if repo.status != "ready":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Repo isn't ready to answer questions (status: {repo.status}).")
    provider, model = embedding_identity()
    if (repo.embedding_provider, repo.embedding_model) != (provider, model):
        built_with = f"{repo.embedding_provider} / {repo.embedding_model}" if repo.embedding_provider else "an unknown model"
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Re-index needed: this repo was indexed with {built_with}, but the app now uses {provider} / {model}.",
        )


@router.post("/api/repos/{repo_id}/ask", dependencies=[Depends(limit("ask"))])
def ask(
    repo_id: int, body: AskRequest, user: CurrentUser | None = Depends(current_user), db: Session = Depends(get_db)
) -> StreamingResponse:
    """Stream an answer as NDJSON. Problems found before streaming starts return normal HTTP errors."""
    repo = get_repo_or_404(db, repo_id, user)
    require_searchable(repo)

    if body.conversation_id is None:
        conversation = Conversation(repo_id=repo_id, title=body.question[:MAX_TITLE_LENGTH], owner_user_id=owner_id(user))
        db.add(conversation)
        db.flush()
        previous: list[Message] = []
    else:
        conversation = db.get(Conversation, body.conversation_id)
        if conversation is None or conversation.repo_id != repo_id or conversation.owner_user_id != owner_id(user):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found for this repo.")
        newest_first = db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.id.desc())
            .limit(MAX_HISTORY_MESSAGES)
        ).all()
        previous = list(reversed(newest_first))

    history = to_chat_history(previous)
    db.add(Message(conversation_id=conversation.id, role="user", content=body.question))
    db.commit()

    return StreamingResponse(
        stream_answer(repo_id, conversation.id, body.question, history),
        media_type="application/x-ndjson",
    )


@router.get("/api/repos/{repo_id}/conversations", response_model=list[ConversationOut])
def list_conversations(
    repo_id: int, user: CurrentUser | None = Depends(current_user), db: Session = Depends(get_db)
) -> list[Conversation]:
    """List this user's conversations about a repo, newest first. Guests share the public ones."""
    get_repo_or_404(db, repo_id, user)
    query = (
        select(Conversation)
        .where(Conversation.repo_id == repo_id, owned_by(Conversation.owner_user_id, user))
        .order_by(Conversation.id.desc())
    )
    return list(db.scalars(query))


@router.get("/api/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(
    conversation_id: int, user: CurrentUser | None = Depends(current_user), db: Session = Depends(get_db)
) -> list[Message]:
    """List a conversation's messages, oldest first."""
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.owner_user_id != owner_id(user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    return list(db.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)))
