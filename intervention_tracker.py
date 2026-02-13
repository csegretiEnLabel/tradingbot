"""
Intervention Tracker
--------------------
Tracks when and why signals are rejected or modified by the AI agent or risk manager.
Provides visibility into decision-making process and helps identify patterns in rejections.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class Intervention:
    """Record of an AI or risk manager intervention on a signal."""
    
    signal_id: str
    intervener: Literal["AI_AGENT", "RISK_MANAGER", "USER"]
    action: Literal["REJECTED", "MODIFIED", "APPROVED"]
    reasoning: str
    timestamp: str
    original_action: str  # What signal recommended (BUY, SELL, HOLD)
    final_action: str  # What actually happened
    symbol: str
    strategy: Optional[str] = None  # Which quant strategy generated the signal
    metadata: Optional[dict] = None  # Additional context


class InterventionTracker:
    """
    Tracks interventions on trading signals.
    
    Saves to data/interventions.jsonl (append-only log).
    Provides analytics on rejection patterns.
    """
    
    INTERVENTION_FILE = os.path.join("data", "interventions.jsonl")
    
    def __init__(self):
        """Initialize intervention tracker."""
        os.makedirs("data", exist_ok=True)
        logger.info("InterventionTracker initialized")
    
    def record_intervention(
        self,
        signal_id: str,
        symbol: str,
        intervener: Literal["AI_AGENT", "RISK_MANAGER", "USER"],
        action: Literal["REJECTED", "MODIFIED", "APPROVED"],
        reasoning: str,
        original_action: str,
        final_action: str,
        strategy: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Record an intervention.
        
        Args:
            signal_id: ID of the signal being intervened on
            symbol: Stock symbol
            intervener: Who made the decision
            action: What they did (REJECTED, MODIFIED, APPROVED)
            reasoning: Why they did it
            original_action: What signal recommended
            final_action: What actually happened
            strategy: Which strategy generated the signal (optional)
            metadata: Additional context (optional)
        """
        intervention = Intervention(
            signal_id=signal_id,
            symbol=symbol,
            intervener=intervener,
            action=action,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            original_action=original_action,
            final_action=final_action,
            strategy=strategy,
            metadata=metadata or {},
        )
        
        # Append to JSONL file
        try:
            with open(self.INTERVENTION_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(intervention)) + "\n")
            
            logger.info(
                f"Intervention recorded: {intervener} {action} {symbol} "
                f"({original_action} → {final_action})"
            )
        except Exception as e:
            logger.error(f"Failed to record intervention: {e}")
    
    def get_intervention_history(
        self,
        symbol: Optional[str] = None,
        intervener: Optional[str] = None,
        limit: int = 100,
    ) -> list[Intervention]:
        """
        Retrieve intervention history.
        
        Args:
            symbol: Filter by symbol (optional)
            intervener: Filter by intervener (optional)
            limit: Max number of interventions to return
            
        Returns:
            List of Intervention objects (most recent first)
        """
        if not os.path.exists(self.INTERVENTION_FILE):
            return []
        
        interventions = []
        try:
            with open(self.INTERVENTION_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        intervention = Intervention(**data)
                        
                        # Apply filters
                        if symbol and intervention.symbol != symbol:
                            continue
                        if intervener and intervention.intervener != intervener:
                            continue
                        
                        interventions.append(intervention)
        except Exception as e:
            logger.error(f"Failed to load intervention history: {e}")
        
        # Return most recent first
        interventions.reverse()
        return interventions[:limit]
    
    def get_rejection_stats(self, days: int = 7) -> dict:
        """
        Get rejection statistics for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with rejection statistics:
            {
                "total_interventions": 45,
                "by_intervener": {"AI_AGENT": 30, "RISK_MANAGER": 15},
                "by_action": {"REJECTED": 25, "APPROVED": 15, "MODIFIED": 5},
                "rejection_rate": 0.56,
                "top_reasons": [
                    {"reason": "Insufficient R:R ratio", "count": 12},
                    ...
                ],
                "by_symbol": {"PLTR": 5, "SOFI": 3, ...}
            }
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        stats = {
            "total_interventions": 0,
            "by_intervener": {},
            "by_action": {},
            "rejection_rate": 0.0,
            "top_reasons": [],
            "by_symbol": {},
            "by_strategy": {},
        }
        
        if not os.path.exists(self.INTERVENTION_FILE):
            return stats
        
        interventions = []
        reason_counter = {}
        
        try:
            with open(self.INTERVENTION_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        intervention = Intervention(**data)
                        
                        # Filter by date
                        if datetime.fromisoformat(intervention.timestamp) < cutoff:
                            continue
                        
                        interventions.append(intervention)
                        
                        # Count by intervener
                        stats["by_intervener"][intervention.intervener] = (
                            stats["by_intervener"].get(intervention.intervener, 0) + 1
                        )
                        
                        # Count by action
                        stats["by_action"][intervention.action] = (
                            stats["by_action"].get(intervention.action, 0) + 1
                        )
                        
                        # Count by symbol
                        stats["by_symbol"][intervention.symbol] = (
                            stats["by_symbol"].get(intervention.symbol, 0) + 1
                        )
                        
                        # Count by strategy
                        if intervention.strategy:
                            stats["by_strategy"][intervention.strategy] = (
                                stats["by_strategy"].get(intervention.strategy, 0) + 1
                            )
                        
                        # Count reasons
                        reason = intervention.reasoning[:100]  # Truncate for grouping
                        reason_counter[reason] = reason_counter.get(reason, 0) + 1
            
            stats["total_interventions"] = len(interventions)
            
            # Calculate rejection rate
            rejected = stats["by_action"].get("REJECTED", 0)
            if stats["total_interventions"] > 0:
                stats["rejection_rate"] = rejected / stats["total_interventions"]
            
            # Top reasons
            stats["top_reasons"] = [
                {"reason": reason, "count": count}
                for reason, count in sorted(
                    reason_counter.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]
            
        except Exception as e:
            logger.error(f"Failed to calculate rejection stats: {e}")
        
        return stats
    
    def get_strategy_approval_rates(self) -> dict:
        """
        Get approval rates by strategy.
        
        Returns:
            {
                "kama": {"approved": 12, "rejected": 8, "approval_rate": 0.60},
                "trend_follow": {"approved": 15, "rejected": 5, "approval_rate": 0.75},
                "momentum": {"approved": 10, "rejected": 10, "approval_rate": 0.50}
            }
        """
        if not os.path.exists(self.INTERVENTION_FILE):
            return {}
        
        strategy_stats = {}
        
        try:
            with open(self.INTERVENTION_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        intervention = Intervention(**data)
                        
                        if not intervention.strategy:
                            continue
                        
                        if intervention.strategy not in strategy_stats:
                            strategy_stats[intervention.strategy] = {
                                "approved": 0,
                                "rejected": 0,
                                "modified": 0,
                            }
                        
                        if intervention.action == "APPROVED":
                            strategy_stats[intervention.strategy]["approved"] += 1
                        elif intervention.action == "REJECTED":
                            strategy_stats[intervention.strategy]["rejected"] += 1
                        elif intervention.action == "MODIFIED":
                            strategy_stats[intervention.strategy]["modified"] += 1
            
            # Calculate approval rates
            for strategy, counts in strategy_stats.items():
                total = counts["approved"] + counts["rejected"] + counts["modified"]
                if total > 0:
                    counts["approval_rate"] = counts["approved"] / total
                else:
                    counts["approval_rate"] = 0.0
        
        except Exception as e:
            logger.error(f"Failed to calculate strategy approval rates: {e}")
        
        return strategy_stats
