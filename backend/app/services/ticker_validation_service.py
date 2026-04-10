"""
Ticker Validation Service for logging and reporting invalid tickers.

Provides centralized functionality for:
- Logging ticker validation failures during data fetching
- Classifying errors into standard categories
- Querying and reporting on validation failures
- Managing resolution status
"""
import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_

from ..models.ticker_validation import TickerValidationLog

logger = logging.getLogger(__name__)


class TickerValidationService:
    """Service for logging and reporting ticker validation failures."""

    # Error type constants
    ERROR_NO_DATA = 'no_data'
    ERROR_DELISTED = 'delisted'
    ERROR_API_ERROR = 'api_error'
    ERROR_INVALID_RESPONSE = 'invalid_response'
    ERROR_EMPTY_INFO = 'empty_info'

    # Data source constants
    SOURCE_YFINANCE = 'yfinance'
    SOURCE_FINVIZ = 'finviz'
    SOURCE_BOTH = 'both'

    # Trigger constants
    TRIGGER_FUNDAMENTALS_REFRESH = 'fundamentals_refresh'
    TRIGGER_CACHE_WARMUP = 'cache_warmup'
    TRIGGER_UNIVERSE_REFRESH = 'universe_refresh'
    TRIGGER_SCAN = 'scan'

    def __init__(self):
        """Initialize ticker validation service."""
        pass

    def classify_error(
        self,
        exception: Optional[Exception] = None,
        info_response: Optional[Dict] = None,
    ) -> Tuple[str, str]:
        """
        Classify an error into a standard error type and message.

        Args:
            exception: Optional exception that was raised
            info_response: Optional response dict from yfinance/finviz

        Returns:
            Tuple of (error_type, error_message)
        """
        # Case 1: Exception was thrown
        if exception:
            error_str = str(exception).lower()

            # Rate limiting
            if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                return (self.ERROR_API_ERROR, f'Rate limited: {exception}')

            # Network errors
            if 'timeout' in error_str or 'connection' in error_str or 'network' in error_str:
                return (self.ERROR_API_ERROR, f'Network error: {exception}')

            # Delisted detection
            if 'delisted' in error_str or 'no longer traded' in error_str or 'not found' in error_str:
                return (self.ERROR_DELISTED, f'Ticker appears delisted: {exception}')

            return (self.ERROR_API_ERROR, str(exception))

        # Case 2: No response at all
        if info_response is None:
            return (self.ERROR_NO_DATA, 'No data returned from API')

        # Case 3: Empty or minimal response
        if isinstance(info_response, dict) and len(info_response) < 3:
            return (self.ERROR_EMPTY_INFO, f'Response has only {len(info_response)} keys')

        # Case 4: Missing essential fields
        if isinstance(info_response, dict):
            if not info_response.get('longName') and not info_response.get('shortName') and not info_response.get('symbol'):
                return (self.ERROR_INVALID_RESPONSE, 'Missing essential fields (name, symbol)')

        return (self.ERROR_NO_DATA, 'Unclassified validation failure')

    def log_validation_failure(
        self,
        db: Session,
        symbol: str,
        error_type: str,
        error_message: str,
        data_source: str,
        triggered_by: str,
        task_id: Optional[str] = None,
        error_details: Optional[Dict] = None,
    ) -> int:
        """
        Log a ticker validation failure.

        If the same symbol failed recently (within 24 hours) with the same error type,
        increments the consecutive_failures counter instead of creating a new record.

        Args:
            db: Database session
            symbol: Stock ticker symbol
            error_type: Error category (use class constants)
            error_message: Human-readable error message
            data_source: Data source that failed (yfinance, finviz, both)
            triggered_by: What triggered this validation (fundamentals_refresh, etc.)
            task_id: Optional Celery task ID for correlation
            error_details: Optional dict with additional context (will be JSON serialized)

        Returns:
            ID of the log entry (new or updated)
        """
        try:
            symbol = symbol.upper()
            now = datetime.utcnow()
            cutoff = now - timedelta(hours=24)

            # Check for recent failure of same symbol with same error type
            recent_failure = db.query(TickerValidationLog).filter(
                and_(
                    TickerValidationLog.symbol == symbol,
                    TickerValidationLog.error_type == error_type,
                    TickerValidationLog.is_resolved == False,
                    TickerValidationLog.detected_at >= cutoff
                )
            ).order_by(desc(TickerValidationLog.detected_at)).first()

            if recent_failure:
                # Increment consecutive failures
                recent_failure.consecutive_failures += 1
                recent_failure.error_message = error_message
                recent_failure.task_id = task_id
                if error_details:
                    recent_failure.error_details = json.dumps(error_details)
                db.commit()
                logger.debug(f"Incremented failure count for {symbol}: {recent_failure.consecutive_failures}")
                return recent_failure.id
            else:
                # Create new record
                new_log = TickerValidationLog(
                    symbol=symbol,
                    error_type=error_type,
                    error_message=error_message,
                    error_details=json.dumps(error_details) if error_details else None,
                    data_source=data_source,
                    triggered_by=triggered_by,
                    task_id=task_id,
                    is_resolved=False,
                    consecutive_failures=1,
                )
                db.add(new_log)
                db.commit()
                db.refresh(new_log)
                logger.debug(f"Created new validation log for {symbol}: {error_type}")
                return new_log.id

        except Exception as e:
            logger.error(f"Error logging validation failure for {symbol}: {e}", exc_info=True)
            db.rollback()
            return -1

    def get_unresolved_failures(
        self,
        db: Session,
        limit: int = 100,
        offset: int = 0,
        error_type: Optional[str] = None,
        triggered_by: Optional[str] = None,
        min_consecutive_failures: int = 1,
        days_back: int = 30,
    ) -> List[Dict]:
        """
        Get list of unresolved ticker validation failures.

        Args:
            db: Database session
            limit: Maximum number of results
            offset: Offset for pagination
            error_type: Optional filter by error type
            triggered_by: Optional filter by trigger source
            min_consecutive_failures: Minimum consecutive failures to include
            days_back: Number of days to look back

        Returns:
            List of failure dicts
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days_back)

            query = db.query(TickerValidationLog).filter(
                and_(
                    TickerValidationLog.is_resolved == False,
                    TickerValidationLog.detected_at >= cutoff,
                    TickerValidationLog.consecutive_failures >= min_consecutive_failures
                )
            )

            if error_type:
                query = query.filter(TickerValidationLog.error_type == error_type)

            if triggered_by:
                query = query.filter(TickerValidationLog.triggered_by == triggered_by)

            # Order by consecutive failures (most failures first), then by date
            query = query.order_by(
                desc(TickerValidationLog.consecutive_failures),
                desc(TickerValidationLog.detected_at)
            )

            results = query.offset(offset).limit(limit).all()

            return [
                {
                    'id': r.id,
                    'symbol': r.symbol,
                    'error_type': r.error_type,
                    'error_message': r.error_message,
                    'error_details': json.loads(r.error_details) if r.error_details else None,
                    'data_source': r.data_source,
                    'triggered_by': r.triggered_by,
                    'task_id': r.task_id,
                    'consecutive_failures': r.consecutive_failures,
                    'detected_at': r.detected_at.isoformat() if r.detected_at else None,
                    'is_resolved': r.is_resolved,
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Error getting unresolved failures: {e}", exc_info=True)
            return []

    def get_failure_summary(self, db: Session, days_back: int = 7) -> Dict:
        """
        Get summary statistics of validation failures.

        Args:
            db: Database session
            days_back: Number of days to include in summary

        Returns:
            Dict with aggregate statistics
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days_back)

            # Get base query for unresolved failures in time window
            base_filter = and_(
                TickerValidationLog.is_resolved == False,
                TickerValidationLog.detected_at >= cutoff
            )

            # Total failures
            total_failures = db.query(TickerValidationLog).filter(base_filter).count()

            # Unique symbols
            unique_symbols = db.query(
                func.count(func.distinct(TickerValidationLog.symbol))
            ).filter(base_filter).scalar() or 0

            # By error type
            by_error_type = {}
            error_type_counts = db.query(
                TickerValidationLog.error_type,
                func.count(TickerValidationLog.id)
            ).filter(base_filter).group_by(TickerValidationLog.error_type).all()

            for error_type, count in error_type_counts:
                by_error_type[error_type] = count

            # By data source
            by_data_source = {}
            source_counts = db.query(
                TickerValidationLog.data_source,
                func.count(TickerValidationLog.id)
            ).filter(base_filter).group_by(TickerValidationLog.data_source).all()

            for source, count in source_counts:
                if source:
                    by_data_source[source] = count

            # By trigger
            by_trigger = {}
            trigger_counts = db.query(
                TickerValidationLog.triggered_by,
                func.count(TickerValidationLog.id)
            ).filter(base_filter).group_by(TickerValidationLog.triggered_by).all()

            for trigger, count in trigger_counts:
                by_trigger[trigger] = count

            # Top failing symbols
            top_failing = db.query(
                TickerValidationLog.symbol,
                func.sum(TickerValidationLog.consecutive_failures).label('total_failures')
            ).filter(base_filter).group_by(
                TickerValidationLog.symbol
            ).order_by(
                desc('total_failures')
            ).limit(10).all()

            top_failing_symbols = [
                {'symbol': symbol, 'failures': int(failures)}
                for symbol, failures in top_failing
            ]

            return {
                'total_failures': total_failures,
                'unique_symbols': unique_symbols,
                'by_error_type': by_error_type,
                'by_data_source': by_data_source,
                'by_trigger': by_trigger,
                'top_failing_symbols': top_failing_symbols,
                'period_start': cutoff.isoformat(),
                'period_end': datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting failure summary: {e}", exc_info=True)
            return {
                'total_failures': 0,
                'unique_symbols': 0,
                'by_error_type': {},
                'by_data_source': {},
                'by_trigger': {},
                'top_failing_symbols': [],
                'error': str(e),
            }

    def get_symbol_history(
        self,
        db: Session,
        symbol: str,
        limit: int = 20,
    ) -> List[Dict]:
        """
        Get validation failure history for a specific symbol.

        Args:
            db: Database session
            symbol: Stock ticker symbol
            limit: Maximum number of results

        Returns:
            List of failure dicts for the symbol
        """
        try:
            symbol = symbol.upper()

            results = db.query(TickerValidationLog).filter(
                TickerValidationLog.symbol == symbol
            ).order_by(
                desc(TickerValidationLog.detected_at)
            ).limit(limit).all()

            return [
                {
                    'id': r.id,
                    'symbol': r.symbol,
                    'error_type': r.error_type,
                    'error_message': r.error_message,
                    'error_details': json.loads(r.error_details) if r.error_details else None,
                    'data_source': r.data_source,
                    'triggered_by': r.triggered_by,
                    'task_id': r.task_id,
                    'consecutive_failures': r.consecutive_failures,
                    'detected_at': r.detected_at.isoformat() if r.detected_at else None,
                    'is_resolved': r.is_resolved,
                    'resolved_at': r.resolved_at.isoformat() if r.resolved_at else None,
                    'resolution_notes': r.resolution_notes,
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Error getting symbol history for {symbol}: {e}", exc_info=True)
            return []

    def resolve_failure(
        self,
        db: Session,
        log_id: int,
        resolution_notes: str,
    ) -> bool:
        """
        Mark a validation failure as resolved.

        Args:
            db: Database session
            log_id: ID of the log entry to resolve
            resolution_notes: Notes about the resolution

        Returns:
            True if successful, False otherwise
        """
        try:
            log_entry = db.query(TickerValidationLog).filter(
                TickerValidationLog.id == log_id
            ).first()

            if not log_entry:
                logger.warning(f"Log entry {log_id} not found")
                return False

            log_entry.is_resolved = True
            log_entry.resolved_at = datetime.utcnow()
            log_entry.resolution_notes = resolution_notes
            db.commit()

            logger.info(f"Resolved validation failure {log_id} for {log_entry.symbol}")
            return True

        except Exception as e:
            logger.error(f"Error resolving failure {log_id}: {e}", exc_info=True)
            db.rollback()
            return False

    def bulk_resolve_by_symbol(
        self,
        db: Session,
        symbol: str,
        resolution_notes: str,
    ) -> int:
        """
        Resolve all unresolved failures for a symbol.

        Args:
            db: Database session
            symbol: Stock ticker symbol
            resolution_notes: Notes about the resolution

        Returns:
            Number of entries resolved
        """
        try:
            symbol = symbol.upper()
            now = datetime.utcnow()

            count = db.query(TickerValidationLog).filter(
                and_(
                    TickerValidationLog.symbol == symbol,
                    TickerValidationLog.is_resolved == False
                )
            ).update(
                {
                    'is_resolved': True,
                    'resolved_at': now,
                    'resolution_notes': resolution_notes,
                },
                synchronize_session=False
            )

            db.commit()
            logger.info(f"Bulk resolved {count} failures for {symbol}")
            return count

        except Exception as e:
            logger.error(f"Error bulk resolving for {symbol}: {e}", exc_info=True)
            db.rollback()
            return 0
