"""Shared configuration/constants for ResearchMemory."""
from __future__ import annotations
from pathlib import Path

DB_DIR = Path(__file__).parent
DB_PATH = DB_DIR / "research.db"
BWK_ROOT = Path(__file__).resolve().parents[2]

# Provenance priority (path, doc_type, provenance, priority)
SOURCE_PRIORITY = [
    (BWK_ROOT / "foreman-staff/researcher/approved", "research", "validated", 10),
    (BWK_ROOT / "common/docs", "reference", "reference", 12),
    (BWK_ROOT / "common/claude", "reference", "reference", 13),
    (BWK_ROOT / "foreman-staff/researcher/archive", "research", "research", 20),
    (BWK_ROOT / "foreman-staff/researcher/cache", "research", "research", 25),
    (BWK_ROOT / "foreman-staff/researcher/inbox", "research", "research", 30),
    (BWK_ROOT / "main/docs", "research", "research", 40),
    (BWK_ROOT / "gemm/docs", "research", "research", 40),
    (BWK_ROOT / "fused-mlp/docs", "research", "research", 40),
    (BWK_ROOT / "attention/docs", "research", "research", 40),
    (BWK_ROOT / "dotproduct/docs", "research", "research", 40),
    (BWK_ROOT / "linalg/docs", "research", "research", 40),
    (BWK_ROOT / "lu/docs", "research", "research", 40),
    (BWK_ROOT / "qr/docs", "research", "research", 40),
    (BWK_ROOT / "rmsnorm/docs", "research", "research", 40),
    (BWK_ROOT / "spmv/docs", "research", "research", 40),
    (BWK_ROOT / "numerical/docs", "research", "research", 40),
    (BWK_ROOT / "cuquantum/docs", "research", "research", 40),
    (BWK_ROOT / "chess-training/docs", "research", "research", 40),
    (BWK_ROOT / "octave-gpu/docs", "research", "research", 40),
    (BWK_ROOT / "ui/docs", "research", "research", 40),
    (BWK_ROOT / "main/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "gemm/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "fused-mlp/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "attention/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "dotproduct/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "linalg/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "lu/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "qr/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "rmsnorm/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "spmv/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "numerical/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "cuquantum/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "chess-training/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "octave-gpu/.claude", "agent_state", "validated", 45),
]

CODE_SOURCES = []

PROVENANCE_TIERS = {
    "validated": {"boost": 1.5, "description": "Foreman-approved, empirically backed"},
    "reference": {"boost": 1.3, "description": "Shared reference docs, manuals"},
    "research":  {"boost": 1.0, "description": "Active research briefs"},
    "archive":   {"boost": 0.7, "description": "Historical snapshots, may be stale"},
}

JOB_PHASES = {
    'ideation':    ['wishlist', 'planning'],
    'development': ['not_started', 'algo_building', 'algo_optimizing', 'hw_optimizing',
                    'stuck_needs_research', 'research_available'],
    'validation':  ['compiles_ok', 'tests_writing', 'testing', 'testing_pass', 'testing_fail',
                    'edge_testing', 'edge_pass', 'edge_fail'],
    'rework':      ['rework', 'rework_complete', 'retesting', 'retest_pass', 'retest_fail'],
    'quality':     ['linting', 'lint_pass', 'lint_fail'],
    'shipping':    ['ready_to_ship', 'shipping', 'shipped'],
    'terminal':    ['converged', 'parked', 'abandoned'],
}

PHASE_ORDER = ['ideation', 'development', 'validation', 'rework', 'quality', 'shipping', 'terminal']
STATE_TO_PHASE = {s: phase for phase, states in JOB_PHASES.items() for s in states}
ALL_JOB_STATES = set(STATE_TO_PHASE.keys())
JOB_TYPES = {'kernel', 'algorithm', 'infrastructure', 'research'}
JOB_PRIORITIES = {'1', '2', '3', '4', '5'}
FACTORY_MODES = {
    'fixed_shape_kernel', 'general_shape_library', 'numerical_method',
    'alternative_arithmetic', 'research_exploration',
}
OPTIMIZATION_SCOPES = {'algorithmic', 'hardware_tuned', 'hybrid'}
EXECUTION_LANES = {'active', 'hopper', 'incubating', 'parked'}
MESSAGE_TYPES = {'halt', 'blocker', 'question', 'feedback', 'info', 'directive'}
MESSAGE_STATUSES = {'open', 'acknowledged', 'resolved'}
MESSAGE_PRIORITIES = {'urgent', 'normal', 'low'}

__all__ = [
    'DB_DIR', 'DB_PATH', 'BWK_ROOT', 'SOURCE_PRIORITY', 'CODE_SOURCES', 'PROVENANCE_TIERS',
    'JOB_PHASES', 'PHASE_ORDER', 'STATE_TO_PHASE', 'ALL_JOB_STATES', 'JOB_TYPES', 'JOB_PRIORITIES',
    'FACTORY_MODES', 'OPTIMIZATION_SCOPES', 'EXECUTION_LANES', 'MESSAGE_TYPES',
    'MESSAGE_STATUSES', 'MESSAGE_PRIORITIES'
]
