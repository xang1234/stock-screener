"""Baseline schema for PostgreSQL deployments."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260408_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('app_settings',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('key', sa.String(length=100), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_app_settings_category', 'app_settings', ['category'])
    op.create_index('ix_app_settings_id', 'app_settings', ['id'])
    op.create_index('ix_app_settings_key', 'app_settings', ['key'], unique=True)

    op.create_table('chatbot_folders',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('is_collapsed', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_chatbot_folders_id', 'chatbot_folders', ['id'])

    op.create_table('content_items',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=True),
        sa.Column('source_type', sa.String(length=20), nullable=False),
        sa.Column('source_name', sa.String(length=100), nullable=True),
        sa.Column('external_id', sa.String(length=200), nullable=True),
        sa.Column('title', sa.String(length=500), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('url', sa.String(length=500), nullable=True),
        sa.Column('author', sa.String(length=100), nullable=True),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('is_processed', sa.Boolean(), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('extraction_error', sa.Text(), nullable=True),
        sa.UniqueConstraint('source_type', 'external_id', name='uix_source_external_id'),
    )
    op.create_index('idx_content_unprocessed', 'content_items', ['is_processed', 'published_at'])
    op.create_index('ix_content_items_id', 'content_items', ['id'])
    op.create_index('ix_content_items_is_processed', 'content_items', ['is_processed'])
    op.create_index('ix_content_items_published_at', 'content_items', ['published_at'])
    op.create_index('ix_content_items_source_id', 'content_items', ['source_id'])
    op.create_index('ix_content_items_source_type', 'content_items', ['source_type'])

    op.create_table('content_sources',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('source_type', sa.String(length=20), nullable=False),
        sa.Column('url', sa.String(length=500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=True),
        sa.Column('fetch_interval_minutes', sa.Integer(), nullable=True),
        sa.Column('pipelines', sa.JSON(), nullable=True),
        sa.Column('last_fetched_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_items_fetched', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('source_type', 'url', name='uix_source_type_url'),
    )
    op.create_index('ix_content_sources_id', 'content_sources', ['id'])
    op.create_index('ix_content_sources_source_type', 'content_sources', ['source_type'])

    op.create_table('document_cache',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('document_type', sa.String(length=20), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=True),
        sa.Column('source_url', sa.String(length=1000), nullable=False),
        sa.Column('cik', sa.String(length=20), nullable=True),
        sa.Column('accession_number', sa.String(length=30), nullable=True),
        sa.Column('filing_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('fiscal_year', sa.Integer(), nullable=True),
        sa.Column('title', sa.String(length=500), nullable=True),
        sa.Column('document_hash', sa.String(length=64), nullable=True),
        sa.Column('full_text', sa.Text(), nullable=True),
        sa.Column('text_length', sa.Integer(), nullable=True),
        sa.Column('token_count_estimate', sa.Integer(), nullable=True),
        sa.Column('is_chunked', sa.Boolean(), nullable=True),
        sa.Column('extraction_method', sa.String(length=30), nullable=True),
        sa.Column('extraction_error', sa.Text(), nullable=True),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('source_url'),
    )
    op.create_index('idx_document_cache_cik', 'document_cache', ['cik'])
    op.create_index('idx_document_cache_symbol', 'document_cache', ['symbol'])
    op.create_index('idx_document_cache_type', 'document_cache', ['document_type'])
    op.create_index('ix_document_cache_id', 'document_cache', ['id'])
    op.create_index('ix_document_cache_symbol', 'document_cache', ['symbol'])

    op.create_table('feature_runs',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False, autoincrement=True),
        sa.Column('as_of_date', sa.Date(), nullable=False),
        sa.Column('run_type', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('code_version', sa.Text(), nullable=True),
        sa.Column('universe_hash', sa.Text(), nullable=True),
        sa.Column('input_hash', sa.Text(), nullable=True),
        sa.Column('config_json', sa.JSON(), nullable=True),
        sa.Column('correlation_id', sa.Text(), nullable=True),
        sa.Column('stats_json', sa.JSON(), nullable=True),
        sa.Column('warnings_json', sa.JSON(), nullable=True),
    )
    op.create_index('ix_feature_runs_as_of_date', 'feature_runs', ['as_of_date'])
    op.create_index('ix_feature_runs_correlation_id', 'feature_runs', ['correlation_id'])
    op.create_index('ix_feature_runs_date_status', 'feature_runs', ['as_of_date', 'status'])
    op.create_index('ix_feature_runs_exact_lookup', 'feature_runs', ['status', 'universe_hash', 'input_hash', 'published_at'])
    op.create_index('ix_feature_runs_id', 'feature_runs', ['id'])
    op.create_index('ix_feature_runs_status', 'feature_runs', ['status'])

    op.create_table('filter_presets',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('filters', sa.Text(), nullable=False),
        sa.Column('sort_by', sa.String(length=50), nullable=False),
        sa.Column('sort_order', sa.String(length=10), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_filter_presets_id', 'filter_presets', ['id'])
    op.create_index('ix_filter_presets_name', 'filter_presets', ['name'], unique=True)

    op.create_table('ibd_group_ranks',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('industry_group', sa.String(length=100), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('avg_rs_rating', sa.Float(), nullable=False),
        sa.Column('median_rs_rating', sa.Float(), nullable=True),
        sa.Column('weighted_avg_rs_rating', sa.Float(), nullable=True),
        sa.Column('rs_std_dev', sa.Float(), nullable=True),
        sa.Column('num_stocks', sa.Integer(), nullable=True),
        sa.Column('num_stocks_rs_above_80', sa.Integer(), nullable=True),
        sa.Column('top_symbol', sa.String(length=10), nullable=True),
        sa.Column('top_rs_rating', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('industry_group', 'date', name='uix_ibd_group_rank_date'),
    )
    op.create_index('idx_ibd_group_rank_date', 'ibd_group_ranks', ['industry_group', 'date'])
    op.create_index('idx_ibd_rank_by_date', 'ibd_group_ranks', ['date', 'rank'])
    op.create_index('ix_ibd_group_ranks_date', 'ibd_group_ranks', ['date'])
    op.create_index('ix_ibd_group_ranks_id', 'ibd_group_ranks', ['id'])
    op.create_index('ix_ibd_group_ranks_industry_group', 'ibd_group_ranks', ['industry_group'])

    op.create_table('ibd_industry_groups',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('industry_group', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('idx_ibd_industry_group', 'ibd_industry_groups', ['industry_group'])
    op.create_index('ix_ibd_industry_groups_id', 'ibd_industry_groups', ['id'])
    op.create_index('ix_ibd_industry_groups_industry_group', 'ibd_industry_groups', ['industry_group'])
    op.create_index('ix_ibd_industry_groups_symbol', 'ibd_industry_groups', ['symbol'], unique=True)

    op.create_table('industries',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('sector_name', sa.String(length=100), nullable=False),
        sa.Column('industry_group', sa.String(length=100), nullable=True),
        sa.Column('industry', sa.String(length=100), nullable=False),
        sa.Column('sub_industry', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('sector_name', 'industry', name='uix_sector_industry'),
    )
    op.create_index('idx_sector', 'industries', ['sector_name'])
    op.create_index('ix_industries_id', 'industries', ['id'])
    op.create_index('ix_industries_industry', 'industries', ['industry'])
    op.create_index('ix_industries_sector_name', 'industries', ['sector_name'])

    op.create_table('industry_performance',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('industry', sa.String(length=100), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('group_rs', sa.Float(), nullable=True),
        sa.Column('leadership_score', sa.Float(), nullable=True),
        sa.Column('num_stocks', sa.Integer(), nullable=True),
        sa.Column('stage2_count', sa.Integer(), nullable=True),
        sa.Column('stage2_pct', sa.Float(), nullable=True),
        sa.Column('high_rs_count', sa.Integer(), nullable=True),
        sa.Column('high_rs_pct', sa.Float(), nullable=True),
        sa.Column('volume_trend', sa.String(length=20), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('industry', 'date', name='uix_industry_date'),
    )
    op.create_index('idx_industry_date', 'industry_performance', ['industry', 'date'])
    op.create_index('ix_industry_performance_date', 'industry_performance', ['date'])
    op.create_index('ix_industry_performance_id', 'industry_performance', ['id'])
    op.create_index('ix_industry_performance_industry', 'industry_performance', ['industry'])

    op.create_table('institutional_ownership_history',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('institutional_pct', sa.Float(), nullable=True),
        sa.Column('insider_pct', sa.Float(), nullable=True),
        sa.Column('institutional_transactions', sa.Float(), nullable=True),
        sa.Column('valid_from', sa.Date(), nullable=False),
        sa.Column('valid_to', sa.Date(), nullable=True),
        sa.Column('is_current', sa.Boolean(), nullable=True),
        sa.Column('data_source', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=False), nullable=True),
    )
    op.create_index('ix_institutional_ownership_history_symbol', 'institutional_ownership_history', ['symbol'])

    op.create_table('market_breadth',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('stocks_up_4pct', sa.Integer(), nullable=False),
        sa.Column('stocks_down_4pct', sa.Integer(), nullable=False),
        sa.Column('ratio_5day', sa.Float(), nullable=True),
        sa.Column('ratio_10day', sa.Float(), nullable=True),
        sa.Column('stocks_up_25pct_quarter', sa.Integer(), nullable=False),
        sa.Column('stocks_down_25pct_quarter', sa.Integer(), nullable=False),
        sa.Column('stocks_up_25pct_month', sa.Integer(), nullable=False),
        sa.Column('stocks_down_25pct_month', sa.Integer(), nullable=False),
        sa.Column('stocks_up_50pct_month', sa.Integer(), nullable=False),
        sa.Column('stocks_down_50pct_month', sa.Integer(), nullable=False),
        sa.Column('stocks_up_13pct_34days', sa.Integer(), nullable=False),
        sa.Column('stocks_down_13pct_34days', sa.Integer(), nullable=False),
        sa.Column('total_stocks_scanned', sa.Integer(), nullable=False),
        sa.Column('calculation_duration_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('idx_breadth_date', 'market_breadth', ['date'])
    op.create_index('ix_market_breadth_date', 'market_breadth', ['date'], unique=True)
    op.create_index('ix_market_breadth_id', 'market_breadth', ['id'])

    op.create_table('market_status',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('spy_price', sa.Float(), nullable=True),
        sa.Column('spy_ma50', sa.Float(), nullable=True),
        sa.Column('spy_ma200', sa.Float(), nullable=True),
        sa.Column('trend', sa.String(length=20), nullable=True),
        sa.Column('vix', sa.Float(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_market_status_date', 'market_status', ['date'], unique=True)
    op.create_index('ix_market_status_id', 'market_status', ['id'])

    op.create_table('prompt_presets',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_prompt_presets_id', 'prompt_presets', ['id'])
    op.create_index('ix_prompt_presets_name', 'prompt_presets', ['name'], unique=True)

    op.create_table('provider_snapshot_runs',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('snapshot_key', sa.String(length=64), nullable=False),
        sa.Column('run_mode', sa.String(length=16), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('source_revision', sa.String(length=128), nullable=False),
        sa.Column('coverage_stats_json', sa.Text(), nullable=True),
        sa.Column('parity_stats_json', sa.Text(), nullable=True),
        sa.Column('warnings_json', sa.Text(), nullable=True),
        sa.Column('symbols_total', sa.Integer(), nullable=False),
        sa.Column('symbols_published', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('snapshot_key', 'source_revision', name='uq_provider_snapshot_revision'),
    )
    op.create_index('idx_provider_snapshot_runs_key_created', 'provider_snapshot_runs', ['snapshot_key', 'created_at'])
    op.create_index('idx_provider_snapshot_runs_key_status', 'provider_snapshot_runs', ['snapshot_key', 'status'])
    op.create_index('ix_provider_snapshot_runs_id', 'provider_snapshot_runs', ['id'])
    op.create_index('ix_provider_snapshot_runs_snapshot_key', 'provider_snapshot_runs', ['snapshot_key'])
    op.create_index('ix_provider_snapshot_runs_status', 'provider_snapshot_runs', ['status'])

    op.create_table('scan_watchlist',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('list_name', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=50), nullable=False),
        sa.Column('display_name', sa.String(length=100), nullable=True),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('list_name', 'symbol', name='uix_list_symbol'),
    )
    op.create_index('ix_scan_watchlist_id', 'scan_watchlist', ['id'])
    op.create_index('ix_scan_watchlist_list_name', 'scan_watchlist', ['list_name'])

    op.create_table('sector_rotation',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('sector', sa.String(length=100), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('rs_rating', sa.Float(), nullable=True),
        sa.Column('rs_change', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('trend', sa.String(length=20), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('sector', 'date', name='uix_sector_date'),
    )
    op.create_index('idx_sector_date', 'sector_rotation', ['sector', 'date'])
    op.create_index('ix_sector_rotation_date', 'sector_rotation', ['date'])
    op.create_index('ix_sector_rotation_id', 'sector_rotation', ['id'])
    op.create_index('ix_sector_rotation_sector', 'sector_rotation', ['sector'])

    op.create_table('stock_fundamentals',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('market_cap', sa.BigInteger(), nullable=True),
        sa.Column('shares_outstanding', sa.BigInteger(), nullable=True),
        sa.Column('avg_volume', sa.BigInteger(), nullable=True),
        sa.Column('relative_volume', sa.Float(), nullable=True),
        sa.Column('pe_ratio', sa.Float(), nullable=True),
        sa.Column('forward_pe', sa.Float(), nullable=True),
        sa.Column('peg_ratio', sa.Float(), nullable=True),
        sa.Column('price_to_book', sa.Float(), nullable=True),
        sa.Column('price_to_sales', sa.Float(), nullable=True),
        sa.Column('price_to_cash', sa.Float(), nullable=True),
        sa.Column('price_to_fcf', sa.Float(), nullable=True),
        sa.Column('ev_ebitda', sa.Float(), nullable=True),
        sa.Column('ev_sales', sa.Float(), nullable=True),
        sa.Column('target_price', sa.Float(), nullable=True),
        sa.Column('eps_current', sa.Float(), nullable=True),
        sa.Column('eps_next_y', sa.Float(), nullable=True),
        sa.Column('eps_next_5y', sa.Float(), nullable=True),
        sa.Column('eps_next_q', sa.Float(), nullable=True),
        sa.Column('eps_growth_quarterly', sa.Float(), nullable=True),
        sa.Column('eps_growth_annual', sa.Float(), nullable=True),
        sa.Column('eps_growth_yy', sa.Float(), nullable=True),
        sa.Column('revenue_current', sa.BigInteger(), nullable=True),
        sa.Column('revenue_growth', sa.Float(), nullable=True),
        sa.Column('sales_past_5y', sa.Float(), nullable=True),
        sa.Column('sales_growth_yy', sa.Float(), nullable=True),
        sa.Column('sales_growth_qq', sa.Float(), nullable=True),
        sa.Column('recent_quarter_date', sa.String(length=50), nullable=True),
        sa.Column('previous_quarter_date', sa.String(length=50), nullable=True),
        sa.Column('profit_margin', sa.Float(), nullable=True),
        sa.Column('operating_margin', sa.Float(), nullable=True),
        sa.Column('gross_margin', sa.Float(), nullable=True),
        sa.Column('roe', sa.Float(), nullable=True),
        sa.Column('roa', sa.Float(), nullable=True),
        sa.Column('roic', sa.Float(), nullable=True),
        sa.Column('current_ratio', sa.Float(), nullable=True),
        sa.Column('quick_ratio', sa.Float(), nullable=True),
        sa.Column('debt_to_equity', sa.Float(), nullable=True),
        sa.Column('lt_debt_to_equity', sa.Float(), nullable=True),
        sa.Column('insider_ownership', sa.Float(), nullable=True),
        sa.Column('insider_transactions', sa.Float(), nullable=True),
        sa.Column('institutional_ownership', sa.Float(), nullable=True),
        sa.Column('institutional_transactions', sa.Float(), nullable=True),
        sa.Column('institutional_change', sa.Float(), nullable=True),
        sa.Column('short_float', sa.Float(), nullable=True),
        sa.Column('short_ratio', sa.Float(), nullable=True),
        sa.Column('short_interest', sa.BigInteger(), nullable=True),
        sa.Column('beta', sa.Float(), nullable=True),
        sa.Column('rsi_14', sa.Float(), nullable=True),
        sa.Column('atr_14', sa.Float(), nullable=True),
        sa.Column('sma_20', sa.Float(), nullable=True),
        sa.Column('sma_50', sa.Float(), nullable=True),
        sa.Column('sma_200', sa.Float(), nullable=True),
        sa.Column('volatility_week', sa.Float(), nullable=True),
        sa.Column('volatility_month', sa.Float(), nullable=True),
        sa.Column('perf_week', sa.Float(), nullable=True),
        sa.Column('perf_month', sa.Float(), nullable=True),
        sa.Column('perf_quarter', sa.Float(), nullable=True),
        sa.Column('perf_half_year', sa.Float(), nullable=True),
        sa.Column('perf_year', sa.Float(), nullable=True),
        sa.Column('perf_ytd', sa.Float(), nullable=True),
        sa.Column('dividend_ttm', sa.Float(), nullable=True),
        sa.Column('dividend_yield', sa.Float(), nullable=True),
        sa.Column('payout_ratio', sa.Float(), nullable=True),
        sa.Column('week_52_high', sa.Float(), nullable=True),
        sa.Column('week_52_high_distance', sa.Float(), nullable=True),
        sa.Column('week_52_low', sa.Float(), nullable=True),
        sa.Column('week_52_low_distance', sa.Float(), nullable=True),
        sa.Column('sector', sa.String(length=100), nullable=True),
        sa.Column('industry', sa.String(length=100), nullable=True),
        sa.Column('country', sa.String(length=50), nullable=True),
        sa.Column('employees', sa.Integer(), nullable=True),
        sa.Column('ipo_date', sa.Date(), nullable=True),
        sa.Column('description_yfinance', sa.String(), nullable=True),
        sa.Column('description_finviz', sa.String(), nullable=True),
        sa.Column('recommendation', sa.Float(), nullable=True),
        sa.Column('eps_5yr_cagr', sa.Float(), nullable=True),
        sa.Column('eps_q1_yoy', sa.Float(), nullable=True),
        sa.Column('eps_q2_yoy', sa.Float(), nullable=True),
        sa.Column('eps_raw_score', sa.Float(), nullable=True),
        sa.Column('eps_rating', sa.Integer(), nullable=True),
        sa.Column('eps_years_available', sa.Integer(), nullable=True),
        sa.Column('data_source', sa.String(length=20), nullable=True),
        sa.Column('data_source_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finviz_snapshot_revision', sa.String(length=128), nullable=True),
        sa.Column('finviz_snapshot_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('yahoo_profile_refreshed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('yahoo_statements_refreshed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('technicals_refreshed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_stock_fundamentals_finviz_snapshot_revision', 'stock_fundamentals', ['finviz_snapshot_revision'])
    op.create_index('ix_stock_fundamentals_id', 'stock_fundamentals', ['id'])
    op.create_index('ix_stock_fundamentals_symbol', 'stock_fundamentals', ['symbol'], unique=True)

    op.create_table('stock_industry',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('sector', sa.String(length=100), nullable=True),
        sa.Column('industry_group', sa.String(length=100), nullable=True),
        sa.Column('industry', sa.String(length=100), nullable=True),
        sa.Column('sub_industry', sa.String(length=100), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('idx_sector_industry', 'stock_industry', ['sector', 'industry'])
    op.create_index('ix_stock_industry_id', 'stock_industry', ['id'])
    op.create_index('ix_stock_industry_industry', 'stock_industry', ['industry'])
    op.create_index('ix_stock_industry_sector', 'stock_industry', ['sector'])
    op.create_index('ix_stock_industry_symbol', 'stock_industry', ['symbol'], unique=True)

    op.create_table('stock_prices',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('open', sa.Float(), nullable=True),
        sa.Column('high', sa.Float(), nullable=True),
        sa.Column('low', sa.Float(), nullable=True),
        sa.Column('close', sa.Float(), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('adj_close', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('symbol', 'date', name='uix_symbol_date'),
    )
    op.create_index('idx_symbol_date', 'stock_prices', ['symbol', 'date'])
    op.create_index('ix_stock_prices_date', 'stock_prices', ['date'])
    op.create_index('ix_stock_prices_id', 'stock_prices', ['id'])
    op.create_index('ix_stock_prices_symbol', 'stock_prices', ['symbol'])

    op.create_table('stock_technicals',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('current_price', sa.Float(), nullable=True),
        sa.Column('ma_50', sa.Float(), nullable=True),
        sa.Column('ma_150', sa.Float(), nullable=True),
        sa.Column('ma_200', sa.Float(), nullable=True),
        sa.Column('ma_200_month_ago', sa.Float(), nullable=True),
        sa.Column('rs_rating', sa.Float(), nullable=True),
        sa.Column('high_52w', sa.Float(), nullable=True),
        sa.Column('low_52w', sa.Float(), nullable=True),
        sa.Column('avg_volume_50d', sa.BigInteger(), nullable=True),
        sa.Column('current_volume', sa.BigInteger(), nullable=True),
        sa.Column('stage', sa.Integer(), nullable=True),
        sa.Column('vcp_score', sa.Float(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_stock_technicals_id', 'stock_technicals', ['id'])
    op.create_index('ix_stock_technicals_symbol', 'stock_technicals', ['symbol'], unique=True)

    op.create_table('stock_universe',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('exchange', sa.String(length=20), nullable=True),
        sa.Column('sector', sa.String(length=100), nullable=True),
        sa.Column('industry', sa.String(length=100), nullable=True),
        sa.Column('market_cap', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('status_reason', sa.String(length=255), nullable=True),
        sa.Column('is_sp500', sa.Boolean(), nullable=True),
        sa.Column('source', sa.String(length=20), nullable=True),
        sa.Column('added_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('last_seen_in_source_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deactivated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('consecutive_fetch_failures', sa.Integer(), nullable=False),
        sa.Column('last_fetch_success_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_fetch_failure_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('idx_universe_exchange_active', 'stock_universe', ['exchange', 'is_active'])
    op.create_index('idx_universe_exchange_status', 'stock_universe', ['exchange', 'status'])
    op.create_index('idx_universe_sector_active', 'stock_universe', ['sector', 'is_active'])
    op.create_index('idx_universe_status_active', 'stock_universe', ['status', 'is_active'])
    op.create_index('ix_stock_universe_exchange', 'stock_universe', ['exchange'])
    op.create_index('ix_stock_universe_id', 'stock_universe', ['id'])
    op.create_index('ix_stock_universe_is_active', 'stock_universe', ['is_active'])
    op.create_index('ix_stock_universe_is_sp500', 'stock_universe', ['is_sp500'])
    op.create_index('ix_stock_universe_sector', 'stock_universe', ['sector'])
    op.create_index('ix_stock_universe_status', 'stock_universe', ['status'])
    op.create_index('ix_stock_universe_symbol', 'stock_universe', ['symbol'], unique=True)

    op.create_table('stock_universe_status_events',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('old_status', sa.String(length=32), nullable=True),
        sa.Column('new_status', sa.String(length=32), nullable=False),
        sa.Column('trigger_source', sa.String(length=64), nullable=False),
        sa.Column('reason', sa.String(length=255), nullable=True),
        sa.Column('payload_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('idx_universe_status_events_status_created', 'stock_universe_status_events', ['new_status', 'created_at'])
    op.create_index('idx_universe_status_events_symbol_created', 'stock_universe_status_events', ['symbol', 'created_at'])
    op.create_index('ix_stock_universe_status_events_created_at', 'stock_universe_status_events', ['created_at'])
    op.create_index('ix_stock_universe_status_events_id', 'stock_universe_status_events', ['id'])
    op.create_index('ix_stock_universe_status_events_new_status', 'stock_universe_status_events', ['new_status'])
    op.create_index('ix_stock_universe_status_events_symbol', 'stock_universe_status_events', ['symbol'])

    op.create_table('task_execution_history',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('task_name', sa.String(length=100), nullable=False),
        sa.Column('task_function', sa.String(length=200), nullable=True),
        sa.Column('task_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('result_summary', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('triggered_by', sa.String(length=20), nullable=True),
    )
    op.create_index('idx_task_name_started', 'task_execution_history', ['task_name', 'started_at'])
    op.create_index('ix_task_execution_history_id', 'task_execution_history', ['id'])
    op.create_index('ix_task_execution_history_task_name', 'task_execution_history', ['task_name'])

    op.create_table('theme_alerts',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('theme_cluster_id', sa.Integer(), nullable=True),
        sa.Column('alert_type', sa.String(length=30), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('severity', sa.String(length=10), nullable=True),
        sa.Column('related_tickers', sa.JSON(), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('is_read', sa.Boolean(), nullable=True),
        sa.Column('is_dismissed', sa.Boolean(), nullable=True),
        sa.Column('triggered_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('idx_alert_unread', 'theme_alerts', ['is_read', 'triggered_at'])
    op.create_index('ix_theme_alerts_alert_type', 'theme_alerts', ['alert_type'])
    op.create_index('ix_theme_alerts_id', 'theme_alerts', ['id'])
    op.create_index('ix_theme_alerts_theme_cluster_id', 'theme_alerts', ['theme_cluster_id'])
    op.create_index('ix_theme_alerts_triggered_at', 'theme_alerts', ['triggered_at'])

    op.create_table('theme_clusters',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('canonical_key', sa.String(length=96), nullable=False),
        sa.Column('display_name', sa.String(length=200), nullable=False),
        sa.Column('aliases', sa.JSON(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('pipeline', sa.String(length=20), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('is_emerging', sa.Boolean(), nullable=True),
        sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('discovery_source', sa.String(length=20), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_validated', sa.Boolean(), nullable=True),
        sa.Column('lifecycle_state', sa.String(length=20), nullable=False),
        sa.Column('lifecycle_state_updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('lifecycle_state_metadata', sa.JSON(), nullable=True),
        sa.Column('candidate_since_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('dormant_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reactivated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retired_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('parent_cluster_id', sa.Integer(), sa.ForeignKey('theme_clusters.id'), nullable=True),
        sa.Column('is_l1', sa.Boolean(), nullable=False),
        sa.Column('taxonomy_level', sa.Integer(), nullable=False),
        sa.Column('l1_assignment_method', sa.String(length=20), nullable=True),
        sa.Column('l1_assignment_confidence', sa.Float(), nullable=True),
        sa.Column('l1_assigned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('pipeline', 'canonical_key', name='uix_theme_clusters_pipeline_canonical_key'),
        sa.CheckConstraint("lifecycle_state IN ('candidate', 'active', 'dormant', 'reactivated', 'retired')", name='ck_theme_clusters_lifecycle_state'),
    )
    op.create_index('ix_theme_clusters_canonical_key', 'theme_clusters', ['canonical_key'])
    op.create_index('ix_theme_clusters_category', 'theme_clusters', ['category'])
    op.create_index('ix_theme_clusters_id', 'theme_clusters', ['id'])
    op.create_index('ix_theme_clusters_is_active', 'theme_clusters', ['is_active'])
    op.create_index('ix_theme_clusters_lifecycle_state', 'theme_clusters', ['lifecycle_state'])
    op.create_index('ix_theme_clusters_name', 'theme_clusters', ['name'])
    op.create_index('ix_theme_clusters_parent_cluster_id', 'theme_clusters', ['parent_cluster_id'])
    op.create_index('ix_theme_clusters_pipeline', 'theme_clusters', ['pipeline'])

    op.create_table('theme_constituents',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('theme_cluster_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('source', sa.String(length=20), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('mention_count', sa.Integer(), nullable=True),
        sa.Column('first_mentioned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_mentioned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('correlation_to_theme', sa.Float(), nullable=True),
        sa.Column('correlation_updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('theme_cluster_id', 'symbol', name='uix_theme_symbol'),
    )
    op.create_index('idx_theme_constituents', 'theme_constituents', ['theme_cluster_id', 'symbol'])
    op.create_index('ix_theme_constituents_id', 'theme_constituents', ['id'])
    op.create_index('ix_theme_constituents_symbol', 'theme_constituents', ['symbol'])
    op.create_index('ix_theme_constituents_theme_cluster_id', 'theme_constituents', ['theme_cluster_id'])

    op.create_table('theme_embeddings',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('theme_cluster_id', sa.Integer(), nullable=False),
        sa.Column('embedding', sa.Text(), nullable=False),
        sa.Column('embedding_model', sa.String(length=50), nullable=True),
        sa.Column('embedding_text', sa.Text(), nullable=True),
        sa.Column('content_hash', sa.String(length=64), nullable=True),
        sa.Column('model_version', sa.String(length=40), nullable=True),
        sa.Column('is_stale', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_theme_embeddings_content_hash', 'theme_embeddings', ['content_hash'])
    op.create_index('ix_theme_embeddings_id', 'theme_embeddings', ['id'])
    op.create_index('ix_theme_embeddings_is_stale', 'theme_embeddings', ['is_stale'])
    op.create_index('ix_theme_embeddings_model_version', 'theme_embeddings', ['model_version'])
    op.create_index('ix_theme_embeddings_theme_cluster_id', 'theme_embeddings', ['theme_cluster_id'], unique=True)

    op.create_table('theme_mentions',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('content_item_id', sa.Integer(), nullable=True),
        sa.Column('source_type', sa.String(length=20), nullable=False),
        sa.Column('source_name', sa.String(length=100), nullable=True),
        sa.Column('raw_theme', sa.String(length=200), nullable=False),
        sa.Column('canonical_theme', sa.String(length=200), nullable=True),
        sa.Column('theme_cluster_id', sa.Integer(), nullable=True),
        sa.Column('match_method', sa.String(length=40), nullable=True),
        sa.Column('match_score', sa.Float(), nullable=True),
        sa.Column('match_threshold', sa.Float(), nullable=True),
        sa.Column('threshold_version', sa.String(length=40), nullable=True),
        sa.Column('match_score_model', sa.String(length=80), nullable=True),
        sa.Column('match_score_model_version', sa.String(length=40), nullable=True),
        sa.Column('match_fallback_reason', sa.String(length=120), nullable=True),
        sa.Column('best_alternative_cluster_id', sa.Integer(), nullable=True),
        sa.Column('best_alternative_score', sa.Float(), nullable=True),
        sa.Column('match_score_margin', sa.Float(), nullable=True),
        sa.Column('pipeline', sa.String(length=20), nullable=True),
        sa.Column('tickers', sa.JSON(), nullable=True),
        sa.Column('ticker_count', sa.Integer(), nullable=True),
        sa.Column('sentiment', sa.String(length=20), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('excerpt', sa.Text(), nullable=True),
        sa.Column('mentioned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('extracted_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('idx_mention_cluster', 'theme_mentions', ['theme_cluster_id', 'mentioned_at'])
    op.create_index('idx_theme_mention_date', 'theme_mentions', ['canonical_theme', 'mentioned_at'])
    op.create_index('idx_theme_mentions_pipeline', 'theme_mentions', ['pipeline'])
    op.create_index('ix_theme_mentions_best_alternative_cluster_id', 'theme_mentions', ['best_alternative_cluster_id'])
    op.create_index('ix_theme_mentions_canonical_theme', 'theme_mentions', ['canonical_theme'])
    op.create_index('ix_theme_mentions_content_item_id', 'theme_mentions', ['content_item_id'])
    op.create_index('ix_theme_mentions_id', 'theme_mentions', ['id'])
    op.create_index('ix_theme_mentions_match_method', 'theme_mentions', ['match_method'])
    op.create_index('ix_theme_mentions_match_score_model', 'theme_mentions', ['match_score_model'])
    op.create_index('ix_theme_mentions_match_score_model_version', 'theme_mentions', ['match_score_model_version'])
    op.create_index('ix_theme_mentions_mentioned_at', 'theme_mentions', ['mentioned_at'])
    op.create_index('ix_theme_mentions_pipeline', 'theme_mentions', ['pipeline'])
    op.create_index('ix_theme_mentions_source_type', 'theme_mentions', ['source_type'])
    op.create_index('ix_theme_mentions_theme_cluster_id', 'theme_mentions', ['theme_cluster_id'])
    op.create_index('ix_theme_mentions_threshold_version', 'theme_mentions', ['threshold_version'])

    op.create_table('theme_merge_history',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('source_cluster_id', sa.Integer(), nullable=True),
        sa.Column('source_cluster_name', sa.String(length=200), nullable=False),
        sa.Column('target_cluster_id', sa.Integer(), nullable=True),
        sa.Column('target_cluster_name', sa.String(length=200), nullable=False),
        sa.Column('merge_type', sa.String(length=20), nullable=False),
        sa.Column('embedding_similarity', sa.Float(), nullable=True),
        sa.Column('llm_confidence', sa.Float(), nullable=True),
        sa.Column('llm_reasoning', sa.Text(), nullable=True),
        sa.Column('constituents_merged', sa.Integer(), nullable=True),
        sa.Column('mentions_merged', sa.Integer(), nullable=True),
        sa.Column('merged_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('merged_by', sa.String(length=50), nullable=True),
    )
    op.create_index('ix_theme_merge_history_id', 'theme_merge_history', ['id'])
    op.create_index('ix_theme_merge_history_merge_type', 'theme_merge_history', ['merge_type'])
    op.create_index('ix_theme_merge_history_merged_at', 'theme_merge_history', ['merged_at'])

    op.create_table('theme_merge_suggestions',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('source_cluster_id', sa.Integer(), nullable=False),
        sa.Column('target_cluster_id', sa.Integer(), nullable=False),
        sa.Column('pair_min_cluster_id', sa.Integer(), nullable=True),
        sa.Column('pair_max_cluster_id', sa.Integer(), nullable=True),
        sa.Column('embedding_similarity', sa.Float(), nullable=True),
        sa.Column('llm_confidence', sa.Float(), nullable=True),
        sa.Column('llm_reasoning', sa.Text(), nullable=True),
        sa.Column('llm_relationship', sa.String(length=20), nullable=True),
        sa.Column('suggested_canonical_name', sa.String(length=200), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('approval_idempotency_key', sa.String(length=128), nullable=True),
        sa.Column('approval_result_json', sa.Text(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('pair_min_cluster_id', 'pair_max_cluster_id', name='uix_merge_suggestion_pair_canonical'),
        sa.UniqueConstraint('source_cluster_id', 'target_cluster_id', name='uix_merge_suggestion_pair'),
    )
    op.create_index('ix_theme_merge_suggestions_approval_idempotency_key', 'theme_merge_suggestions', ['approval_idempotency_key'])
    op.create_index('ix_theme_merge_suggestions_created_at', 'theme_merge_suggestions', ['created_at'])
    op.create_index('ix_theme_merge_suggestions_id', 'theme_merge_suggestions', ['id'])
    op.create_index('ix_theme_merge_suggestions_pair_max_cluster_id', 'theme_merge_suggestions', ['pair_max_cluster_id'])
    op.create_index('ix_theme_merge_suggestions_pair_min_cluster_id', 'theme_merge_suggestions', ['pair_min_cluster_id'])
    op.create_index('ix_theme_merge_suggestions_source_cluster_id', 'theme_merge_suggestions', ['source_cluster_id'])
    op.create_index('ix_theme_merge_suggestions_status', 'theme_merge_suggestions', ['status'])
    op.create_index('ix_theme_merge_suggestions_target_cluster_id', 'theme_merge_suggestions', ['target_cluster_id'])

    op.create_table('theme_metrics',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('theme_cluster_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('pipeline', sa.String(length=20), nullable=True),
        sa.Column('mentions_1d', sa.Integer(), nullable=True),
        sa.Column('mentions_7d', sa.Integer(), nullable=True),
        sa.Column('mentions_30d', sa.Integer(), nullable=True),
        sa.Column('mention_velocity', sa.Float(), nullable=True),
        sa.Column('sentiment_score', sa.Float(), nullable=True),
        sa.Column('basket_return_1d', sa.Float(), nullable=True),
        sa.Column('basket_return_1w', sa.Float(), nullable=True),
        sa.Column('basket_return_1m', sa.Float(), nullable=True),
        sa.Column('basket_rs_vs_spy', sa.Float(), nullable=True),
        sa.Column('avg_internal_correlation', sa.Float(), nullable=True),
        sa.Column('correlation_tightness', sa.Float(), nullable=True),
        sa.Column('num_constituents', sa.Integer(), nullable=True),
        sa.Column('pct_above_50ma', sa.Float(), nullable=True),
        sa.Column('pct_above_200ma', sa.Float(), nullable=True),
        sa.Column('pct_positive_1w', sa.Float(), nullable=True),
        sa.Column('num_passing_minervini', sa.Integer(), nullable=True),
        sa.Column('num_stage_2', sa.Integer(), nullable=True),
        sa.Column('avg_rs_rating', sa.Float(), nullable=True),
        sa.Column('momentum_score', sa.Float(), nullable=True),
        sa.Column('rank', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('theme_cluster_id', 'date', name='uix_theme_metrics_date'),
    )
    op.create_index('idx_theme_metrics_date', 'theme_metrics', ['theme_cluster_id', 'date'])
    op.create_index('idx_theme_metrics_pipeline_date', 'theme_metrics', ['pipeline', 'date'])
    op.create_index('idx_theme_rank', 'theme_metrics', ['date', 'rank'])
    op.create_index('ix_theme_metrics_date', 'theme_metrics', ['date'])
    op.create_index('ix_theme_metrics_id', 'theme_metrics', ['id'])
    op.create_index('ix_theme_metrics_theme_cluster_id', 'theme_metrics', ['theme_cluster_id'])

    op.create_table('theme_pipeline_runs',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('task_id', sa.String(length=100), nullable=True),
        sa.Column('pipeline', sa.String(length=20), nullable=True),
        sa.Column('total_sources', sa.Integer(), nullable=True),
        sa.Column('items_ingested', sa.Integer(), nullable=True),
        sa.Column('items_processed', sa.Integer(), nullable=True),
        sa.Column('items_reprocessed', sa.Integer(), nullable=True),
        sa.Column('themes_extracted', sa.Integer(), nullable=True),
        sa.Column('themes_updated', sa.Integer(), nullable=True),
        sa.Column('alerts_generated', sa.Integer(), nullable=True),
        sa.Column('current_step', sa.String(length=50), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_theme_pipeline_runs_id', 'theme_pipeline_runs', ['id'])
    op.create_index('ix_theme_pipeline_runs_pipeline', 'theme_pipeline_runs', ['pipeline'])
    op.create_index('ix_theme_pipeline_runs_run_id', 'theme_pipeline_runs', ['run_id'], unique=True)

    op.create_table('ticker_validation_log',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('error_type', sa.String(length=50), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', sa.Text(), nullable=True),
        sa.Column('data_source', sa.String(length=20), nullable=True),
        sa.Column('triggered_by', sa.String(length=50), nullable=False),
        sa.Column('task_id', sa.String(length=100), nullable=True),
        sa.Column('is_resolved', sa.Boolean(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('detected_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('consecutive_failures', sa.Integer(), nullable=True),
    )
    op.create_index('idx_validation_symbol_unresolved', 'ticker_validation_log', ['symbol', 'is_resolved', 'detected_at'])
    op.create_index('idx_validation_unresolved', 'ticker_validation_log', ['is_resolved', 'detected_at'])
    op.create_index('ix_ticker_validation_log_id', 'ticker_validation_log', ['id'])
    op.create_index('ix_ticker_validation_log_is_resolved', 'ticker_validation_log', ['is_resolved'])
    op.create_index('ix_ticker_validation_log_symbol', 'ticker_validation_log', ['symbol'])

    op.create_table('ui_view_snapshots',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False, autoincrement=True),
        sa.Column('view_key', sa.String(length=64), nullable=False),
        sa.Column('variant_key', sa.String(length=128), nullable=False),
        sa.Column('source_revision', sa.String(length=256), nullable=False),
        sa.Column('payload_json', sa.JSON(), nullable=False),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.UniqueConstraint('view_key', 'variant_key', 'source_revision', name='uq_ui_view_snapshots_revision'),
    )
    op.create_index('ix_ui_view_snapshots_source_revision', 'ui_view_snapshots', ['source_revision'])
    op.create_index('ix_ui_view_snapshots_variant_key', 'ui_view_snapshots', ['variant_key'])
    op.create_index('ix_ui_view_snapshots_view_key', 'ui_view_snapshots', ['view_key'])

    op.create_table('user_themes',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('color', sa.String(length=20), nullable=True),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_user_themes_id', 'user_themes', ['id'])
    op.create_index('ix_user_themes_name', 'user_themes', ['name'], unique=True)

    op.create_table('user_watchlists',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('color', sa.String(length=20), nullable=True),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_user_watchlists_id', 'user_watchlists', ['id'])
    op.create_index('ix_user_watchlists_name', 'user_watchlists', ['name'], unique=True)

    op.create_table('watchlist',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('added_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_watchlist_id', 'watchlist', ['id'])
    op.create_index('ix_watchlist_symbol', 'watchlist', ['symbol'], unique=True)

    op.create_table('chatbot_conversations',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('conversation_id', sa.String(length=36), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('message_count', sa.Integer(), nullable=True),
        sa.Column('folder_id', sa.Integer(), sa.ForeignKey('chatbot_folders.id', ondelete='SET NULL'), nullable=True),
    )
    op.create_index('ix_chatbot_conversations_conversation_id', 'chatbot_conversations', ['conversation_id'], unique=True)
    op.create_index('ix_chatbot_conversations_id', 'chatbot_conversations', ['id'])

    op.create_table('content_item_pipeline_state',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('content_item_id', sa.Integer(), sa.ForeignKey('content_items.id', ondelete='CASCADE'), nullable=False),
        sa.Column('pipeline', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=30), nullable=False),
        sa.Column('attempt_count', sa.Integer(), nullable=False),
        sa.Column('error_code', sa.String(length=100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('last_attempt_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.CheckConstraint("status IN ('pending', 'in_progress', 'processed', 'failed_retryable', 'failed_terminal')", name='ck_cips_status_values'),
        sa.CheckConstraint("pipeline IN ('technical', 'fundamental')", name='ck_cips_pipeline_values'),
        sa.UniqueConstraint('content_item_id', 'pipeline', name='uix_cips_content_item_pipeline'),
    )
    op.create_index('idx_cips_content_item_pipeline_status', 'content_item_pipeline_state', ['content_item_id', 'pipeline', 'status'])
    op.create_index('idx_cips_pipeline_status_created', 'content_item_pipeline_state', ['pipeline', 'status', 'created_at'])
    op.create_index('idx_cips_pipeline_status_last_attempt', 'content_item_pipeline_state', ['pipeline', 'status', 'last_attempt_at'])
    op.create_index('idx_cips_updated_at', 'content_item_pipeline_state', ['updated_at'])
    op.create_index('ix_content_item_pipeline_state_id', 'content_item_pipeline_state', ['id'])

    op.create_table('document_chunks',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('document_cache.id'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('section_name', sa.String(length=200), nullable=True),
        sa.Column('chunk_text', sa.Text(), nullable=False),
        sa.Column('chunk_tokens', sa.Integer(), nullable=True),
        sa.Column('embedding', sa.Text(), nullable=True),
        sa.Column('embedding_model', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('idx_document_chunks_document', 'document_chunks', ['document_id'])
    op.create_index('idx_document_chunks_section', 'document_chunks', ['section_name'])
    op.create_index('ix_document_chunks_document_id', 'document_chunks', ['document_id'])
    op.create_index('ix_document_chunks_id', 'document_chunks', ['id'])

    op.create_table('feature_run_pointers',
        sa.Column('key', sa.Text(), primary_key=True, nullable=False),
        sa.Column('run_id', sa.Integer(), sa.ForeignKey('feature_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )

    op.create_table('feature_run_universe_symbols',
        sa.Column('run_id', sa.Integer(), sa.ForeignKey('feature_runs.id', ondelete='CASCADE'), primary_key=True, nullable=False),
        sa.Column('symbol', sa.Text(), primary_key=True, nullable=False),
    )
    op.create_index('ix_feature_run_universe_symbols_run_id', 'feature_run_universe_symbols', ['run_id'])
    op.create_index('ix_feature_run_universe_symbols_symbol', 'feature_run_universe_symbols', ['symbol'])

    op.create_table('provider_snapshot_pointers',
        sa.Column('snapshot_key', sa.String(length=64), primary_key=True, nullable=False),
        sa.Column('run_id', sa.Integer(), sa.ForeignKey('provider_snapshot_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_provider_snapshot_pointers_run_id', 'provider_snapshot_pointers', ['run_id'])

    op.create_table('provider_snapshot_rows',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('run_id', sa.Integer(), sa.ForeignKey('provider_snapshot_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('exchange', sa.String(length=20), nullable=True),
        sa.Column('row_hash', sa.String(length=64), nullable=False),
        sa.Column('normalized_payload_json', sa.Text(), nullable=False),
        sa.Column('raw_payload_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.UniqueConstraint('run_id', 'symbol', name='uq_provider_snapshot_row_run_symbol'),
    )
    op.create_index('idx_provider_snapshot_rows_run_exchange', 'provider_snapshot_rows', ['run_id', 'exchange'])
    op.create_index('ix_provider_snapshot_rows_exchange', 'provider_snapshot_rows', ['exchange'])
    op.create_index('ix_provider_snapshot_rows_id', 'provider_snapshot_rows', ['id'])
    op.create_index('ix_provider_snapshot_rows_run_id', 'provider_snapshot_rows', ['run_id'])
    op.create_index('ix_provider_snapshot_rows_symbol', 'provider_snapshot_rows', ['symbol'])

    op.create_table('scans',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('scan_id', sa.String(length=36), nullable=False),
        sa.Column('criteria', sa.JSON(), nullable=True),
        sa.Column('universe', sa.String(length=50), nullable=True),
        sa.Column('universe_key', sa.String(length=128), nullable=True),
        sa.Column('universe_type', sa.String(length=20), nullable=True),
        sa.Column('universe_exchange', sa.String(length=20), nullable=True),
        sa.Column('universe_index', sa.String(length=20), nullable=True),
        sa.Column('universe_symbols', sa.JSON(), nullable=True),
        sa.Column('screener_types', sa.JSON(), nullable=True),
        sa.Column('composite_method', sa.String(length=50), nullable=True),
        sa.Column('total_stocks', sa.Integer(), nullable=True),
        sa.Column('passed_stocks', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('task_id', sa.String(length=100), nullable=True),
        sa.Column('idempotency_key', sa.String(length=64), nullable=True),
        sa.Column('trigger_source', sa.String(length=20), nullable=False, server_default=sa.text('manual')),
        sa.Column('feature_run_id', sa.Integer(), sa.ForeignKey('feature_runs.id'), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_scans_feature_run_id', 'scans', ['feature_run_id'])
    op.create_index('ix_scans_id', 'scans', ['id'])
    op.create_index('ix_scans_idempotency_key', 'scans', ['idempotency_key'], unique=True)
    op.create_index('ix_scans_scan_id', 'scans', ['scan_id'], unique=True)
    op.create_index('ix_scans_trigger_source', 'scans', ['trigger_source'])
    op.create_index('ix_scans_universe_exchange', 'scans', ['universe_exchange'])
    op.create_index('ix_scans_universe_index', 'scans', ['universe_index'])
    op.create_index('ix_scans_universe_key', 'scans', ['universe_key'])
    op.create_index('ix_scans_universe_type', 'scans', ['universe_type'])

    op.create_table('stock_feature_daily',
        sa.Column('run_id', sa.Integer(), sa.ForeignKey('feature_runs.id', ondelete='CASCADE'), primary_key=True, nullable=False),
        sa.Column('symbol', sa.Text(), primary_key=True, nullable=False),
        sa.Column('as_of_date', sa.Date(), nullable=False),
        sa.Column('composite_score', sa.Float(), nullable=True),
        sa.Column('overall_rating', sa.Integer(), nullable=True),
        sa.Column('passes_count', sa.Integer(), nullable=True),
        sa.Column('details_json', sa.JSON(), nullable=True),
    )
    op.create_index('ix_stock_feature_daily_as_of_date', 'stock_feature_daily', ['as_of_date'])
    op.create_index('ix_stock_feature_daily_composite_score', 'stock_feature_daily', ['composite_score'])
    op.create_index('ix_stock_feature_daily_overall_rating', 'stock_feature_daily', ['overall_rating'])
    op.create_index('ix_stock_feature_daily_run_id', 'stock_feature_daily', ['run_id'])
    op.create_index('ix_stock_feature_daily_run_rating', 'stock_feature_daily', ['run_id', 'overall_rating'])
    op.create_index('ix_stock_feature_daily_run_score', 'stock_feature_daily', ['run_id', 'composite_score'])
    op.create_index('ix_stock_feature_daily_symbol', 'stock_feature_daily', ['symbol'])

    op.create_table('theme_aliases',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('theme_cluster_id', sa.Integer(), sa.ForeignKey('theme_clusters.id', ondelete='CASCADE'), nullable=False),
        sa.Column('pipeline', sa.String(length=20), nullable=False),
        sa.Column('alias_text', sa.String(length=200), nullable=False),
        sa.Column('alias_key', sa.String(length=96), nullable=False),
        sa.Column('source', sa.String(length=30), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('evidence_count', sa.Integer(), nullable=False),
        sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.UniqueConstraint('pipeline', 'alias_key', name='uix_theme_alias_pipeline_alias_key'),
    )
    op.create_index('idx_theme_alias_cluster_active', 'theme_aliases', ['theme_cluster_id', 'is_active'])
    op.create_index('idx_theme_alias_source_confidence', 'theme_aliases', ['source', 'confidence'])
    op.create_index('ix_theme_aliases_alias_key', 'theme_aliases', ['alias_key'])
    op.create_index('ix_theme_aliases_id', 'theme_aliases', ['id'])
    op.create_index('ix_theme_aliases_is_active', 'theme_aliases', ['is_active'])
    op.create_index('ix_theme_aliases_pipeline', 'theme_aliases', ['pipeline'])
    op.create_index('ix_theme_aliases_theme_cluster_id', 'theme_aliases', ['theme_cluster_id'])

    op.create_table('theme_lifecycle_transitions',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('theme_cluster_id', sa.Integer(), sa.ForeignKey('theme_clusters.id', ondelete='CASCADE'), nullable=False),
        sa.Column('from_state', sa.String(length=20), nullable=False),
        sa.Column('to_state', sa.String(length=20), nullable=False),
        sa.Column('actor', sa.String(length=80), nullable=False),
        sa.Column('job_name', sa.String(length=80), nullable=True),
        sa.Column('rule_version', sa.String(length=40), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('transition_metadata', sa.JSON(), nullable=True),
        sa.Column('transitioned_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_theme_lifecycle_transitions_id', 'theme_lifecycle_transitions', ['id'])
    op.create_index('ix_theme_lifecycle_transitions_theme_cluster_id', 'theme_lifecycle_transitions', ['theme_cluster_id'])
    op.create_index('ix_theme_lifecycle_transitions_to_state', 'theme_lifecycle_transitions', ['to_state'])
    op.create_index('ix_theme_lifecycle_transitions_transitioned_at', 'theme_lifecycle_transitions', ['transitioned_at'])

    op.create_table('theme_relationships',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('source_cluster_id', sa.Integer(), sa.ForeignKey('theme_clusters.id', ondelete='CASCADE'), nullable=False),
        sa.Column('target_cluster_id', sa.Integer(), sa.ForeignKey('theme_clusters.id', ondelete='CASCADE'), nullable=False),
        sa.Column('pipeline', sa.String(length=20), nullable=False),
        sa.Column('relationship_type', sa.String(length=20), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('provenance', sa.String(length=40), nullable=False),
        sa.Column('evidence', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.CheckConstraint("relationship_type IN ('subset', 'related', 'distinct')", name='ck_theme_relationship_type'),
        sa.UniqueConstraint('source_cluster_id', 'target_cluster_id', 'relationship_type', 'pipeline', name='uix_theme_relationship_edge'),
        sa.CheckConstraint('source_cluster_id != target_cluster_id', name='ck_theme_relationship_not_self'),
    )
    op.create_index('idx_theme_relationship_source_active', 'theme_relationships', ['source_cluster_id', 'is_active'])
    op.create_index('idx_theme_relationship_target_active', 'theme_relationships', ['target_cluster_id', 'is_active'])
    op.create_index('ix_theme_relationships_id', 'theme_relationships', ['id'])
    op.create_index('ix_theme_relationships_is_active', 'theme_relationships', ['is_active'])
    op.create_index('ix_theme_relationships_pipeline', 'theme_relationships', ['pipeline'])
    op.create_index('ix_theme_relationships_relationship_type', 'theme_relationships', ['relationship_type'])
    op.create_index('ix_theme_relationships_source_cluster_id', 'theme_relationships', ['source_cluster_id'])
    op.create_index('ix_theme_relationships_target_cluster_id', 'theme_relationships', ['target_cluster_id'])

    op.create_table('ui_view_snapshot_pointers',
        sa.Column('view_key', sa.String(length=64), primary_key=True, nullable=False),
        sa.Column('variant_key', sa.String(length=128), primary_key=True, nullable=False),
        sa.Column('snapshot_id', sa.Integer(), sa.ForeignKey('ui_view_snapshots.id', ondelete='CASCADE'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_ui_view_snapshot_pointers_snapshot_id', 'ui_view_snapshot_pointers', ['snapshot_id'])

    op.create_table('user_theme_subgroups',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('theme_id', sa.Integer(), sa.ForeignKey('user_themes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('is_collapsed', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('theme_id', 'name', name='uix_theme_subgroup_name'),
    )
    op.create_index('idx_theme_subgroup', 'user_theme_subgroups', ['theme_id', 'position'])
    op.create_index('ix_user_theme_subgroups_id', 'user_theme_subgroups', ['id'])
    op.create_index('ix_user_theme_subgroups_theme_id', 'user_theme_subgroups', ['theme_id'])

    op.create_table('watchlist_items',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('watchlist_id', sa.Integer(), sa.ForeignKey('user_watchlists.id', ondelete='CASCADE'), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('display_name', sa.String(length=100), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('watchlist_id', 'symbol', name='uix_watchlist_symbol'),
    )
    op.create_index('idx_watchlist_item', 'watchlist_items', ['watchlist_id', 'position'])
    op.create_index('ix_watchlist_items_id', 'watchlist_items', ['id'])
    op.create_index('ix_watchlist_items_watchlist_id', 'watchlist_items', ['watchlist_id'])

    op.create_table('chatbot_messages',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('conversation_id', sa.String(length=36), sa.ForeignKey('chatbot_conversations.conversation_id'), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('agent_type', sa.String(length=50), nullable=True),
        sa.Column('tool_name', sa.String(length=100), nullable=True),
        sa.Column('tool_input', sa.JSON(), nullable=True),
        sa.Column('tool_output', sa.JSON(), nullable=True),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('tool_calls', sa.JSON(), nullable=True),
        sa.Column('thinking_traces', sa.JSON(), nullable=True),
        sa.Column('source_references', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('idx_conversation_messages', 'chatbot_messages', ['conversation_id', 'created_at'])
    op.create_index('ix_chatbot_messages_conversation_id', 'chatbot_messages', ['conversation_id'])
    op.create_index('ix_chatbot_messages_id', 'chatbot_messages', ['id'])

    op.create_table('ibd_group_peer_cache',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('scan_id', sa.String(length=36), sa.ForeignKey('scans.scan_id'), nullable=False),
        sa.Column('industry_group', sa.String(length=100), nullable=False),
        sa.Column('total_stocks', sa.Integer(), nullable=True),
        sa.Column('avg_rs_1m', sa.Float(), nullable=True),
        sa.Column('avg_rs_3m', sa.Float(), nullable=True),
        sa.Column('avg_rs_12m', sa.Float(), nullable=True),
        sa.Column('avg_minervini_score', sa.Float(), nullable=True),
        sa.Column('avg_composite_score', sa.Float(), nullable=True),
        sa.Column('top_symbol', sa.String(length=10), nullable=True),
        sa.Column('top_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('scan_id', 'industry_group', name='uix_scan_group'),
    )
    op.create_index('idx_scan_group', 'ibd_group_peer_cache', ['scan_id', 'industry_group'])
    op.create_index('ix_ibd_group_peer_cache_id', 'ibd_group_peer_cache', ['id'])
    op.create_index('ix_ibd_group_peer_cache_industry_group', 'ibd_group_peer_cache', ['industry_group'])
    op.create_index('ix_ibd_group_peer_cache_scan_id', 'ibd_group_peer_cache', ['scan_id'])

    op.create_table('scan_results',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('scan_id', sa.String(length=36), sa.ForeignKey('scans.scan_id'), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('composite_score', sa.Float(), nullable=True),
        sa.Column('minervini_score', sa.Float(), nullable=True),
        sa.Column('canslim_score', sa.Float(), nullable=True),
        sa.Column('ipo_score', sa.Float(), nullable=True),
        sa.Column('custom_score', sa.Float(), nullable=True),
        sa.Column('volume_breakthrough_score', sa.Float(), nullable=True),
        sa.Column('rating', sa.String(length=20), nullable=True),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('market_cap', sa.BigInteger(), nullable=True),
        sa.Column('stage', sa.Integer(), nullable=True),
        sa.Column('rs_rating', sa.Float(), nullable=True),
        sa.Column('rs_rating_1m', sa.Float(), nullable=True),
        sa.Column('rs_rating_3m', sa.Float(), nullable=True),
        sa.Column('rs_rating_12m', sa.Float(), nullable=True),
        sa.Column('eps_growth_qq', sa.Float(), nullable=True),
        sa.Column('sales_growth_qq', sa.Float(), nullable=True),
        sa.Column('eps_growth_yy', sa.Float(), nullable=True),
        sa.Column('sales_growth_yy', sa.Float(), nullable=True),
        sa.Column('peg_ratio', sa.Float(), nullable=True),
        sa.Column('adr_percent', sa.Float(), nullable=True),
        sa.Column('eps_rating', sa.Integer(), nullable=True),
        sa.Column('ibd_industry_group', sa.String(length=100), nullable=True),
        sa.Column('ibd_group_rank', sa.Integer(), nullable=True),
        sa.Column('gics_sector', sa.String(length=100), nullable=True),
        sa.Column('gics_industry', sa.String(length=100), nullable=True),
        sa.Column('rs_sparkline_data', sa.JSON(), nullable=True),
        sa.Column('rs_trend', sa.Integer(), nullable=True),
        sa.Column('price_sparkline_data', sa.JSON(), nullable=True),
        sa.Column('price_change_1d', sa.Float(), nullable=True),
        sa.Column('price_trend', sa.Integer(), nullable=True),
        sa.Column('perf_week', sa.Float(), nullable=True),
        sa.Column('perf_month', sa.Float(), nullable=True),
        sa.Column('perf_3m', sa.Float(), nullable=True),
        sa.Column('perf_6m', sa.Float(), nullable=True),
        sa.Column('gap_percent', sa.Float(), nullable=True),
        sa.Column('volume_surge', sa.Float(), nullable=True),
        sa.Column('ema_10_distance', sa.Float(), nullable=True),
        sa.Column('ema_20_distance', sa.Float(), nullable=True),
        sa.Column('ema_50_distance', sa.Float(), nullable=True),
        sa.Column('week_52_high_distance', sa.Float(), nullable=True),
        sa.Column('week_52_low_distance', sa.Float(), nullable=True),
        sa.Column('ipo_date', sa.String(length=10), nullable=True),
        sa.Column('beta', sa.Float(), nullable=True),
        sa.Column('beta_adj_rs', sa.Float(), nullable=True),
        sa.Column('beta_adj_rs_1m', sa.Float(), nullable=True),
        sa.Column('beta_adj_rs_3m', sa.Float(), nullable=True),
        sa.Column('beta_adj_rs_12m', sa.Float(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('idx_scan_52w_high', 'scan_results', ['scan_id', 'week_52_high_distance'])
    op.create_index('idx_scan_52w_low', 'scan_results', ['scan_id', 'week_52_low_distance'])
    op.create_index('idx_scan_beta', 'scan_results', ['scan_id', 'beta'])
    op.create_index('idx_scan_beta_adj_rs', 'scan_results', ['scan_id', 'beta_adj_rs'])
    op.create_index('idx_scan_beta_adj_rs_12m', 'scan_results', ['scan_id', 'beta_adj_rs_12m'])
    op.create_index('idx_scan_beta_adj_rs_1m', 'scan_results', ['scan_id', 'beta_adj_rs_1m'])
    op.create_index('idx_scan_beta_adj_rs_3m', 'scan_results', ['scan_id', 'beta_adj_rs_3m'])
    op.create_index('idx_scan_ema_10', 'scan_results', ['scan_id', 'ema_10_distance'])
    op.create_index('idx_scan_ema_20', 'scan_results', ['scan_id', 'ema_20_distance'])
    op.create_index('idx_scan_ema_50', 'scan_results', ['scan_id', 'ema_50_distance'])
    op.create_index('idx_scan_eps_growth', 'scan_results', ['scan_id', 'eps_growth_qq'])
    op.create_index('idx_scan_eps_growth_yy', 'scan_results', ['scan_id', 'eps_growth_yy'])
    op.create_index('idx_scan_eps_rating', 'scan_results', ['scan_id', 'eps_rating'])
    op.create_index('idx_scan_gap_percent', 'scan_results', ['scan_id', 'gap_percent'])
    op.create_index('idx_scan_gics_sector', 'scan_results', ['scan_id', 'gics_sector'])
    op.create_index('idx_scan_ibd_group', 'scan_results', ['scan_id', 'ibd_industry_group'])
    op.create_index('idx_scan_ibd_group_rank', 'scan_results', ['scan_id', 'ibd_group_rank'])
    op.create_index('idx_scan_ipo_date', 'scan_results', ['scan_id', 'ipo_date'])
    op.create_index('idx_scan_market_cap', 'scan_results', ['scan_id', 'market_cap'])
    op.create_index('idx_scan_peg', 'scan_results', ['scan_id', 'peg_ratio'])
    op.create_index('idx_scan_perf_3m', 'scan_results', ['scan_id', 'perf_3m'])
    op.create_index('idx_scan_perf_6m', 'scan_results', ['scan_id', 'perf_6m'])
    op.create_index('idx_scan_perf_month', 'scan_results', ['scan_id', 'perf_month'])
    op.create_index('idx_scan_perf_week', 'scan_results', ['scan_id', 'perf_week'])
    op.create_index('idx_scan_price_change_1d', 'scan_results', ['scan_id', 'price_change_1d'])
    op.create_index('idx_scan_price_trend', 'scan_results', ['scan_id', 'price_trend'])
    op.create_index('idx_scan_result_score', 'scan_results', ['scan_id', 'composite_score'])
    op.create_index('idx_scan_rs_12m', 'scan_results', ['scan_id', 'rs_rating_12m'])
    op.create_index('idx_scan_rs_1m', 'scan_results', ['scan_id', 'rs_rating_1m'])
    op.create_index('idx_scan_rs_3m', 'scan_results', ['scan_id', 'rs_rating_3m'])
    op.create_index('idx_scan_rs_rating', 'scan_results', ['scan_id', 'rs_rating'])
    op.create_index('idx_scan_rs_trend', 'scan_results', ['scan_id', 'rs_trend'])
    op.create_index('idx_scan_sales_growth', 'scan_results', ['scan_id', 'sales_growth_qq'])
    op.create_index('idx_scan_sales_growth_yy', 'scan_results', ['scan_id', 'sales_growth_yy'])
    op.create_index('idx_scan_stage', 'scan_results', ['scan_id', 'stage'])
    op.create_index('idx_scan_volume', 'scan_results', ['scan_id', 'volume'])
    op.create_index('idx_scan_volume_surge', 'scan_results', ['scan_id', 'volume_surge'])
    op.create_index('idx_symbol_scan', 'scan_results', ['symbol', 'scan_id'])
    op.create_index('ix_scan_results_adr_percent', 'scan_results', ['adr_percent'])
    op.create_index('ix_scan_results_beta', 'scan_results', ['beta'])
    op.create_index('ix_scan_results_beta_adj_rs', 'scan_results', ['beta_adj_rs'])
    op.create_index('ix_scan_results_beta_adj_rs_12m', 'scan_results', ['beta_adj_rs_12m'])
    op.create_index('ix_scan_results_beta_adj_rs_1m', 'scan_results', ['beta_adj_rs_1m'])
    op.create_index('ix_scan_results_beta_adj_rs_3m', 'scan_results', ['beta_adj_rs_3m'])
    op.create_index('ix_scan_results_composite_score', 'scan_results', ['composite_score'])
    op.create_index('ix_scan_results_ema_10_distance', 'scan_results', ['ema_10_distance'])
    op.create_index('ix_scan_results_ema_20_distance', 'scan_results', ['ema_20_distance'])
    op.create_index('ix_scan_results_ema_50_distance', 'scan_results', ['ema_50_distance'])
    op.create_index('ix_scan_results_eps_growth_qq', 'scan_results', ['eps_growth_qq'])
    op.create_index('ix_scan_results_eps_growth_yy', 'scan_results', ['eps_growth_yy'])
    op.create_index('ix_scan_results_eps_rating', 'scan_results', ['eps_rating'])
    op.create_index('ix_scan_results_gap_percent', 'scan_results', ['gap_percent'])
    op.create_index('ix_scan_results_gics_sector', 'scan_results', ['gics_sector'])
    op.create_index('ix_scan_results_ibd_group_rank', 'scan_results', ['ibd_group_rank'])
    op.create_index('ix_scan_results_ibd_industry_group', 'scan_results', ['ibd_industry_group'])
    op.create_index('ix_scan_results_id', 'scan_results', ['id'])
    op.create_index('ix_scan_results_ipo_date', 'scan_results', ['ipo_date'])
    op.create_index('ix_scan_results_market_cap', 'scan_results', ['market_cap'])
    op.create_index('ix_scan_results_peg_ratio', 'scan_results', ['peg_ratio'])
    op.create_index('ix_scan_results_perf_3m', 'scan_results', ['perf_3m'])
    op.create_index('ix_scan_results_perf_6m', 'scan_results', ['perf_6m'])
    op.create_index('ix_scan_results_perf_month', 'scan_results', ['perf_month'])
    op.create_index('ix_scan_results_perf_week', 'scan_results', ['perf_week'])
    op.create_index('ix_scan_results_price_change_1d', 'scan_results', ['price_change_1d'])
    op.create_index('ix_scan_results_price_trend', 'scan_results', ['price_trend'])
    op.create_index('ix_scan_results_rs_rating', 'scan_results', ['rs_rating'])
    op.create_index('ix_scan_results_rs_rating_12m', 'scan_results', ['rs_rating_12m'])
    op.create_index('ix_scan_results_rs_rating_1m', 'scan_results', ['rs_rating_1m'])
    op.create_index('ix_scan_results_rs_rating_3m', 'scan_results', ['rs_rating_3m'])
    op.create_index('ix_scan_results_rs_trend', 'scan_results', ['rs_trend'])
    op.create_index('ix_scan_results_sales_growth_qq', 'scan_results', ['sales_growth_qq'])
    op.create_index('ix_scan_results_sales_growth_yy', 'scan_results', ['sales_growth_yy'])
    op.create_index('ix_scan_results_scan_id', 'scan_results', ['scan_id'])
    op.create_index('ix_scan_results_stage', 'scan_results', ['stage'])
    op.create_index('ix_scan_results_symbol', 'scan_results', ['symbol'])
    op.create_index('ix_scan_results_volume_surge', 'scan_results', ['volume_surge'])
    op.create_index('ix_scan_results_week_52_high_distance', 'scan_results', ['week_52_high_distance'])
    op.create_index('ix_scan_results_week_52_low_distance', 'scan_results', ['week_52_low_distance'])

    op.create_table('user_theme_stocks',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('subgroup_id', sa.Integer(), sa.ForeignKey('user_theme_subgroups.id', ondelete='CASCADE'), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('display_name', sa.String(length=100), nullable=True),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.UniqueConstraint('subgroup_id', 'symbol', name='uix_subgroup_symbol'),
    )
    op.create_index('idx_subgroup_stock', 'user_theme_stocks', ['subgroup_id', 'position'])
    op.create_index('ix_user_theme_stocks_id', 'user_theme_stocks', ['id'])
    op.create_index('ix_user_theme_stocks_subgroup_id', 'user_theme_stocks', ['subgroup_id'])

    op.create_table('chatbot_agent_executions',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('message_id', sa.Integer(), sa.ForeignKey('chatbot_messages.id'), nullable=False),
        sa.Column('agent_type', sa.String(length=50), nullable=False),
        sa.Column('step_number', sa.Integer(), nullable=True),
        sa.Column('input_prompt', sa.Text(), nullable=True),
        sa.Column('raw_output', sa.Text(), nullable=True),
        sa.Column('parsed_output', sa.JSON(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('model_used', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    )
    op.create_index('ix_chatbot_agent_executions_id', 'chatbot_agent_executions', ['id'])
    op.create_index('ix_chatbot_agent_executions_message_id', 'chatbot_agent_executions', ['message_id'])


def downgrade() -> None:
    op.drop_table('chatbot_agent_executions')
    op.drop_table('user_theme_stocks')
    op.drop_table('scan_results')
    op.drop_table('ibd_group_peer_cache')
    op.drop_table('chatbot_messages')
    op.drop_table('watchlist_items')
    op.drop_table('user_theme_subgroups')
    op.drop_table('ui_view_snapshot_pointers')
    op.drop_table('theme_relationships')
    op.drop_table('theme_lifecycle_transitions')
    op.drop_table('theme_aliases')
    op.drop_table('stock_feature_daily')
    op.drop_table('scans')
    op.drop_table('provider_snapshot_rows')
    op.drop_table('provider_snapshot_pointers')
    op.drop_table('feature_run_universe_symbols')
    op.drop_table('feature_run_pointers')
    op.drop_table('document_chunks')
    op.drop_table('content_item_pipeline_state')
    op.drop_table('chatbot_conversations')
    op.drop_table('watchlist')
    op.drop_table('user_watchlists')
    op.drop_table('user_themes')
    op.drop_table('ui_view_snapshots')
    op.drop_table('ticker_validation_log')
    op.drop_table('theme_pipeline_runs')
    op.drop_table('theme_metrics')
    op.drop_table('theme_merge_suggestions')
    op.drop_table('theme_merge_history')
    op.drop_table('theme_mentions')
    op.drop_table('theme_embeddings')
    op.drop_table('theme_constituents')
    op.drop_table('theme_clusters')
    op.drop_table('theme_alerts')
    op.drop_table('task_execution_history')
    op.drop_table('stock_universe_status_events')
    op.drop_table('stock_universe')
    op.drop_table('stock_technicals')
    op.drop_table('stock_prices')
    op.drop_table('stock_industry')
    op.drop_table('stock_fundamentals')
    op.drop_table('sector_rotation')
    op.drop_table('scan_watchlist')
    op.drop_table('provider_snapshot_runs')
    op.drop_table('prompt_presets')
    op.drop_table('market_status')
    op.drop_table('market_breadth')
    op.drop_table('institutional_ownership_history')
    op.drop_table('industry_performance')
    op.drop_table('industries')
    op.drop_table('ibd_industry_groups')
    op.drop_table('ibd_group_ranks')
    op.drop_table('filter_presets')
    op.drop_table('feature_runs')
    op.drop_table('document_cache')
    op.drop_table('content_sources')
    op.drop_table('content_items')
    op.drop_table('chatbot_folders')
    op.drop_table('app_settings')
