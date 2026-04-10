"""Retire legacy chatbot-era tables and columns no longer used by assistant runtime."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260410_0002"
down_revision = "20260408_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "chatbot_conversations" in existing_tables:
        existing_columns = {column["name"] for column in inspector.get_columns("chatbot_conversations")}
        if "folder_id" in existing_columns:
            op.drop_column("chatbot_conversations", "folder_id")

    for table_name in (
        "chatbot_agent_executions",
        "prompt_presets",
        "document_chunks",
        "document_cache",
        "chatbot_folders",
    ):
        if table_name in existing_tables:
            op.drop_table(table_name)


def downgrade() -> None:
    op.create_table(
        "chatbot_folders",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("is_collapsed", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
    )

    op.add_column("chatbot_conversations", sa.Column("folder_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_chatbot_conversations_folder_id_chatbot_folders",
        "chatbot_conversations",
        "chatbot_folders",
        ["folder_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "prompt_presets",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
    )
    op.create_index("ix_prompt_presets_name", "prompt_presets", ["name"], unique=True)

    op.create_table(
        "document_cache",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("document_type", sa.String(length=20), nullable=False),
        sa.Column("symbol", sa.String(length=10), nullable=True),
        sa.Column("source_url", sa.String(length=1000), nullable=False),
        sa.Column("cik", sa.String(length=20), nullable=True),
        sa.Column("accession_number", sa.String(length=30), nullable=True),
        sa.Column("filing_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fiscal_year", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("document_hash", sa.String(length=64), nullable=True),
        sa.Column("full_text", sa.Text(), nullable=True),
        sa.Column("text_length", sa.Integer(), nullable=True),
        sa.Column("token_count_estimate", sa.Integer(), nullable=True),
        sa.Column("is_chunked", sa.Boolean(), nullable=True),
        sa.Column("extraction_method", sa.String(length=30), nullable=True),
        sa.Column("extraction_error", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.UniqueConstraint("source_url"),
    )
    op.create_index("idx_document_cache_cik", "document_cache", ["cik"])
    op.create_index("idx_document_cache_symbol", "document_cache", ["symbol"])
    op.create_index("idx_document_cache_type", "document_cache", ["document_type"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section_name", sa.String(length=200), nullable=True),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("chunk_tokens", sa.Integer(), nullable=True),
        sa.Column("embedding", sa.Text(), nullable=True),
        sa.Column("embedding_model", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["document_id"], ["document_cache.id"]),
    )
    op.create_index("idx_document_chunks_document", "document_chunks", ["document_id"])
    op.create_index("idx_document_chunks_section", "document_chunks", ["section_name"])

    op.create_table(
        "chatbot_agent_executions",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("agent_type", sa.String(length=50), nullable=False),
        sa.Column("step_number", sa.Integer(), nullable=True),
        sa.Column("input_prompt", sa.Text(), nullable=True),
        sa.Column("raw_output", sa.Text(), nullable=True),
        sa.Column("parsed_output", sa.JSON(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("model_used", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["message_id"], ["chatbot_messages.id"]),
    )
    op.create_index("ix_chatbot_agent_executions_message_id", "chatbot_agent_executions", ["message_id"])
