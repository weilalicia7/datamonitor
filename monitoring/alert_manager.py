"""
Alert Manager
=============

Manages alerts and notifications based on events and system state.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OperatingMode,
    DATA_CACHE_DIR,
    get_logger
)

from .event_aggregator import Event, EventAggregator

logger = get_logger(__name__)


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    priority: AlertPriority
    title: str
    message: str
    created_at: datetime
    expires_at: Optional[datetime]
    status: AlertStatus = AlertStatus.ACTIVE
    source_event_id: Optional[str] = None
    actions: List[Dict] = field(default_factory=list)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Check if alert is still active"""
        if self.status != AlertStatus.ACTIVE:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'priority': self.priority.name,
            'title': self.title,
            'message': self.message,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'status': self.status.value,
            'source_event_id': self.source_event_id,
            'actions': self.actions,
            'is_active': self.is_active
        }


class AlertManager:
    """
    Manages system alerts and notifications.

    Creates alerts based on events, manages alert lifecycle,
    and provides notification functionality.
    """

    def __init__(self, event_aggregator: EventAggregator = None):
        """
        Initialize alert manager.

        Args:
            event_aggregator: EventAggregator instance to monitor
        """
        self.event_aggregator = event_aggregator or EventAggregator()
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self._alert_counter = 0
        self._callbacks: List[Callable] = []
        self._last_mode: OperatingMode = OperatingMode.NORMAL
        self._cache_file = DATA_CACHE_DIR / "alerts_cache.json"
        self._load_alerts()

    def _load_alerts(self) -> None:
        """Load alerts from cache"""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    data = json.load(f)
                    self._alert_counter = data.get('counter', 0)
                    # Could restore alerts here if needed
            except Exception as e:
                logger.warning(f"Failed to load alerts cache: {e}")

    def _save_alerts(self) -> None:
        """Save alerts to cache"""
        try:
            data = {
                'counter': self._alert_counter,
                'active_alerts': [a.to_dict() for a in self.alerts.values() if a.is_active]
            }
            with open(self._cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save alerts cache: {e}")

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        self._alert_counter += 1
        return f"ALT-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}"

    def register_callback(self, callback: Callable) -> None:
        """
        Register callback for new alerts.

        Args:
            callback: Function to call when new alert is created
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self, alert: Alert) -> None:
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def create_alert(self, priority: AlertPriority, title: str,
                     message: str, expires_hours: float = 24,
                     source_event_id: str = None,
                     actions: List[Dict] = None) -> Alert:
        """
        Create a new alert.

        Args:
            priority: Alert priority level
            title: Alert title
            message: Alert message
            expires_hours: Hours until alert expires
            source_event_id: Related event ID
            actions: List of action buttons

        Returns:
            Created Alert object
        """
        alert = Alert(
            alert_id=self._generate_alert_id(),
            priority=priority,
            title=title,
            message=message,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=expires_hours),
            source_event_id=source_event_id,
            actions=actions or []
        )

        self.alerts[alert.alert_id] = alert
        self._save_alerts()
        self._notify_callbacks(alert)

        logger.info(f"Created alert: {alert.alert_id} - {title}")
        return alert

    def create_event_alert(self, event: Event) -> Optional[Alert]:
        """
        Create alert from an event.

        Args:
            event: Event to create alert from

        Returns:
            Created Alert or None if not warranted
        """
        # Only create alerts for significant events
        if event.severity < 0.3:
            return None

        # Determine priority
        if event.severity >= 0.8:
            priority = AlertPriority.CRITICAL
        elif event.severity >= 0.5:
            priority = AlertPriority.HIGH
        elif event.severity >= 0.3:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW

        # Create appropriate actions
        actions = [
            {'label': 'View Details', 'action': 'view_event', 'event_id': event.event_id},
            {'label': 'Dismiss', 'action': 'dismiss'}
        ]

        if event.severity >= 0.5:
            actions.insert(1, {
                'label': 'Adjust Schedule',
                'action': 'adjust_schedule',
                'event_id': event.event_id
            })

        # Create message with impact info
        message = f"{event.description}\n\n"
        message += f"Severity: {event.severity_level.upper()}\n"
        message += f"Affected areas: {', '.join(event.affected_postcodes[:5])}"
        if event.noshow_adjustment > 0:
            message += f"\nExpected no-show increase: +{event.noshow_adjustment:.0%}"
        if event.duration_adjustment > 0:
            message += f"\nExpected delays: +{event.duration_adjustment} min"

        return self.create_alert(
            priority=priority,
            title=event.title,
            message=message,
            expires_hours=12 if event.expires_at is None else
                         (event.expires_at - datetime.now()).total_seconds() / 3600,
            source_event_id=event.event_id,
            actions=actions
        )

    def create_mode_change_alert(self, old_mode: OperatingMode,
                                  new_mode: OperatingMode) -> Alert:
        """
        Create alert for operating mode change.

        Args:
            old_mode: Previous operating mode
            new_mode: New operating mode

        Returns:
            Created Alert
        """
        mode_messages = {
            OperatingMode.NORMAL: "System operating normally. Standard scheduling rules apply.",
            OperatingMode.ELEVATED: "Elevated conditions detected. Adding schedule buffers and sending patient notifications.",
            OperatingMode.CRISIS: "Crisis mode activated. Prioritizing P1-P3 patients. P4 patients may be rescheduled.",
            OperatingMode.EMERGENCY: "EMERGENCY MODE. Critical patients only. Community sites may be activated."
        }

        mode_priorities = {
            OperatingMode.NORMAL: AlertPriority.LOW,
            OperatingMode.ELEVATED: AlertPriority.MEDIUM,
            OperatingMode.CRISIS: AlertPriority.HIGH,
            OperatingMode.EMERGENCY: AlertPriority.CRITICAL
        }

        actions = [
            {'label': 'View Events', 'action': 'view_events'},
            {'label': 'View Schedule', 'action': 'view_schedule'}
        ]

        if new_mode in [OperatingMode.CRISIS, OperatingMode.EMERGENCY]:
            actions.append({'label': 'Re-optimize Schedule', 'action': 're_optimize'})

        return self.create_alert(
            priority=mode_priorities[new_mode],
            title=f"Operating Mode Changed: {new_mode.value.upper()}",
            message=f"System mode changed from {old_mode.value} to {new_mode.value}.\n\n"
                    f"{mode_messages[new_mode]}",
            expires_hours=4,
            actions=actions
        )

    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge
            user: User acknowledging the alert

        Returns:
            True if acknowledged, False if not found
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = user
            self._save_alerts()
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        return False

    def dismiss_alert(self, alert_id: str) -> bool:
        """
        Dismiss an alert.

        Args:
            alert_id: Alert ID to dismiss

        Returns:
            True if dismissed, False if not found
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.DISMISSED
            self.alert_history.append(alert)
            del self.alerts[alert_id]
            self._save_alerts()
            logger.info(f"Alert {alert_id} dismissed")
            return True
        return False

    def get_active_alerts(self, priority: AlertPriority = None) -> List[Alert]:
        """
        Get active alerts.

        Args:
            priority: Filter by priority (optional)

        Returns:
            List of active alerts
        """
        alerts = [a for a in self.alerts.values() if a.is_active]

        if priority:
            alerts = [a for a in alerts if a.priority == priority]

        # Sort by priority (highest first) then by time (newest first)
        alerts.sort(key=lambda x: (-x.priority.value, x.created_at), reverse=True)

        return alerts

    def get_critical_alerts(self) -> List[Alert]:
        """Get critical alerts"""
        return self.get_active_alerts(AlertPriority.CRITICAL)

    def cleanup_expired(self) -> int:
        """
        Clean up expired alerts.

        Returns:
            Number of alerts cleaned up
        """
        expired = []
        for alert_id, alert in self.alerts.items():
            if not alert.is_active and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.EXPIRED
                expired.append(alert_id)

        for alert_id in expired:
            self.alert_history.append(self.alerts[alert_id])
            del self.alerts[alert_id]

        if expired:
            self._save_alerts()
            logger.info(f"Cleaned up {len(expired)} expired alerts")

        return len(expired)

    def check_events_and_update(self) -> List[Alert]:
        """
        Check events and create necessary alerts.

        Returns:
            List of newly created alerts
        """
        new_alerts = []

        # Get current events
        events = self.event_aggregator.get_active_events()

        # Create alerts for new significant events
        existing_event_ids = {
            a.source_event_id for a in self.alerts.values()
            if a.source_event_id
        }

        for event in events:
            if event.event_id not in existing_event_ids:
                alert = self.create_event_alert(event)
                if alert:
                    new_alerts.append(alert)

        # Check for mode changes
        current_mode = self.event_aggregator.determine_operating_mode()
        if current_mode != self._last_mode:
            alert = self.create_mode_change_alert(self._last_mode, current_mode)
            new_alerts.append(alert)
            self._last_mode = current_mode

        # Cleanup expired
        self.cleanup_expired()

        return new_alerts

    def get_summary(self) -> Dict:
        """Get alert summary"""
        active = self.get_active_alerts()
        return {
            'total_active': len(active),
            'critical': len([a for a in active if a.priority == AlertPriority.CRITICAL]),
            'high': len([a for a in active if a.priority == AlertPriority.HIGH]),
            'medium': len([a for a in active if a.priority == AlertPriority.MEDIUM]),
            'low': len([a for a in active if a.priority == AlertPriority.LOW]),
            'alerts': [a.to_dict() for a in active[:10]]
        }


# Example usage
if __name__ == "__main__":
    manager = AlertManager()

    # Check events and create alerts
    print("Checking events...")
    new_alerts = manager.check_events_and_update()
    print(f"Created {len(new_alerts)} new alerts")

    # Get active alerts
    active = manager.get_active_alerts()
    print(f"\nActive alerts: {len(active)}")
    for alert in active:
        print(f"  [{alert.priority.name}] {alert.title}")

    # Get summary
    summary = manager.get_summary()
    print(f"\nSummary: {summary['total_active']} active "
          f"({summary['critical']} critical)")
