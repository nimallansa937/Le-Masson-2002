"""
Layer 7: Failure Cache + Exhaustion Tracking (Memory Layer)

Stores all architecture attempts, their results, and metadata
for DreamCoder analysis and Balloon Expansion decisions.

Analogous to the Memory layer in DESCARTES-Collatz that caches
proof attempts and Z3 results for reuse.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import json
import time
from pathlib import Path


@dataclass
class MemoryEntry:
    """One architecture attempt in memory."""
    template_id: str
    family: str
    timestamp: float
    status: str              # "verify_failed", "train_failed", "completed"
    spike_correlation: float
    n_recovered: int
    verify_passed: bool
    verify_time_seconds: float
    train_time_hours: float
    failure_reason: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    gap_at_time: Optional[float] = None


class MemoryLayer:
    """
    Persistent memory for the DESCARTES search.

    Tracks:
    - All architecture attempts and their results
    - Family exhaustion status
    - Template exhaustion status
    - Best results per family
    - Timeline of gap reduction
    """

    def __init__(self):
        self.entries: List[MemoryEntry] = []
        self.exhausted_templates: Set[str] = set()
        self.exhausted_families: Set[str] = set()
        self.best_by_family: Dict[str, MemoryEntry] = {}
        self.gap_timeline: List[float] = []

    def record(self, entry: MemoryEntry):
        """Record an architecture attempt."""
        self.entries.append(entry)

        # Update best per family
        family = entry.family
        if (family not in self.best_by_family or
                entry.n_recovered > self.best_by_family[family].n_recovered):
            self.best_by_family[family] = entry

        # Track gap timeline
        if entry.gap_at_time is not None:
            self.gap_timeline.append(entry.gap_at_time)

    def mark_template_exhausted(self, template_id: str):
        """Mark a template as exhausted (all C2 params tried)."""
        self.exhausted_templates.add(template_id)

    def mark_family_exhausted(self, family: str):
        """Mark a family as exhausted (all templates in family tried)."""
        self.exhausted_families.add(family)

    def is_template_exhausted(self, template_id: str) -> bool:
        return template_id in self.exhausted_templates

    def is_family_exhausted(self, family: str) -> bool:
        return family in self.exhausted_families

    def get_family_attempts(self, family: str) -> List[MemoryEntry]:
        """Get all attempts for a specific family."""
        return [e for e in self.entries if e.family == family]

    def get_best_overall(self) -> Optional[MemoryEntry]:
        """Get the best attempt across all families."""
        completed = [e for e in self.entries if e.status == "completed"]
        if not completed:
            return None
        return max(completed, key=lambda e: e.n_recovered)

    def get_statistics(self) -> Dict:
        """Get overall search statistics."""
        return {
            'total_attempts': len(self.entries),
            'completed': sum(1 for e in self.entries if e.status == "completed"),
            'verify_failed': sum(1 for e in self.entries if e.status == "verify_failed"),
            'train_failed': sum(1 for e in self.entries if e.status == "train_failed"),
            'exhausted_templates': len(self.exhausted_templates),
            'exhausted_families': len(self.exhausted_families),
            'best_recovery': max((e.n_recovered for e in self.entries), default=0),
            'best_spike_corr': max((e.spike_correlation for e in self.entries), default=0.0),
            'total_compute_hours': sum(e.train_time_hours for e in self.entries),
            'families_tried': len(set(e.family for e in self.entries)),
        }

    def save(self, path: str):
        """Save memory to JSON file."""
        data = {
            'entries': [
                {
                    'template_id': e.template_id,
                    'family': e.family,
                    'timestamp': e.timestamp,
                    'status': e.status,
                    'spike_correlation': e.spike_correlation,
                    'n_recovered': e.n_recovered,
                    'verify_passed': e.verify_passed,
                    'verify_time_seconds': e.verify_time_seconds,
                    'train_time_hours': e.train_time_hours,
                    'failure_reason': e.failure_reason,
                    'gap_at_time': e.gap_at_time,
                }
                for e in self.entries
            ],
            'exhausted_templates': list(self.exhausted_templates),
            'exhausted_families': list(self.exhausted_families),
            'statistics': self.get_statistics(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load memory from JSON file."""
        filepath = Path(path)
        if not filepath.exists():
            return

        with open(path, 'r') as f:
            data = json.load(f)

        for entry_data in data.get('entries', []):
            self.entries.append(MemoryEntry(
                template_id=entry_data['template_id'],
                family=entry_data['family'],
                timestamp=entry_data['timestamp'],
                status=entry_data['status'],
                spike_correlation=entry_data['spike_correlation'],
                n_recovered=entry_data['n_recovered'],
                verify_passed=entry_data['verify_passed'],
                verify_time_seconds=entry_data['verify_time_seconds'],
                train_time_hours=entry_data['train_time_hours'],
                failure_reason=entry_data.get('failure_reason'),
                gap_at_time=entry_data.get('gap_at_time'),
            ))

        self.exhausted_templates = set(data.get('exhausted_templates', []))
        self.exhausted_families = set(data.get('exhausted_families', []))
