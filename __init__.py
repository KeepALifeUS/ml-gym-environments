"""
ML Gym Environments for Crypto Trading Bot v5.0
Context7 Enterprise Patterns для production-ready trading environments

Advanced OpenAI Gym trading environments с comprehensive features:
- Multi-asset crypto trading simulation
- Sentiment-based trading signals integration
- Real-time market microstructure simulation
- Advanced risk management
- Enterprise-grade logging и monitoring
- Scalable parallel execution

This package implements ML-Framework-1345 - Sentiment-Based Trading Signals
through sophisticated ML environments for agent training.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Crypto Trading Bot v5.0 Team"
__license__ = "MIT"

# Core imports
from .src.environments.base_trading_env import (
    BaseTradingEnvironment,
    BaseTradingConfig,
    AsyncBaseTradingEnvironment
)

from .src.environments.crypto_trading_env import (
    CryptoTradingEnvironment,
    CryptoTradingConfig,
    MarketRegime,
    create_crypto_env,
    create_sentiment_crypto_env
)

# Space definitions
from .src.spaces.observations import (
    CryptoObservationSpace,
    ObservationConfig,
    ObservationMode
)

from .src.spaces.actions import (
    CryptoActionSpace,
    ActionConfig,
    ActionMode,
    OrderType
)

# Reward functions
from .src.rewards.profit_reward import (
    ProfitReward,
    ProfitRewardConfig,
    RiskAdjustedProfitReward,
    create_simple_profit_reward,
    create_risk_adjusted_profit_reward
)

from .src.rewards.sharpe_reward import (
    SharpeReward,
    SharpeRewardConfig,
    SortinoReward,
    AdaptiveSharpeReward,
    create_sharpe_reward,
    create_sortino_reward
)

# Simulation components
from .src.simulation.market_simulator import (
    MarketSimulator,
    MarketSimulatorConfig,
    OrderExecution,
    MarketImpactModel
)

# Utilities
from .src.utils.logger import (
    StructuredLogger,
    TradingEvent,
    EventType,
    create_environment_logger
)

from .src.utils.risk_metrics import (
    RiskCalculator,
    RiskMetrics,
    PositionSizer,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)

# Package metadata
__all__ = [
    # Core environments
    "BaseTradingEnvironment",
    "BaseTradingConfig", 
    "AsyncBaseTradingEnvironment",
    "CryptoTradingEnvironment",
    "CryptoTradingConfig",
    "MarketRegime",
    
    # Factory functions
    "create_crypto_env",
    "create_sentiment_crypto_env",
    
    # Spaces
    "CryptoObservationSpace",
    "ObservationConfig",
    "ObservationMode",
    "CryptoActionSpace",
    "ActionConfig",
    "ActionMode",
    "OrderType",
    
    # Rewards
    "ProfitReward",
    "ProfitRewardConfig",
    "RiskAdjustedProfitReward", 
    "SharpeReward",
    "SharpeRewardConfig",
    "SortinoReward",
    "AdaptiveSharpeReward",
    "create_simple_profit_reward",
    "create_risk_adjusted_profit_reward",
    "create_sharpe_reward",
    "create_sortino_reward",
    
    # Simulation
    "MarketSimulator",
    "MarketSimulatorConfig",
    "OrderExecution",
    "MarketImpactModel",
    
    # Utils
    "StructuredLogger",
    "TradingEvent",
    "EventType", 
    "create_environment_logger",
    "RiskCalculator",
    "RiskMetrics",
    "PositionSizer",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
]

# Package configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Enterprise patterns compatibility check
try:
    import gymnasium as gym
    import numpy as np
    import pandas as pd
    GYM_AVAILABLE = True
except ImportError as e:
    GYM_AVAILABLE = False
    import warnings
    warnings.warn(f"Required dependencies not available: {e}")

# Context7 integration check
try:
    from ..common.src.patterns.context7 import Context7Pattern
    CONTEXT7_AVAILABLE = True
except ImportError:
    CONTEXT7_AVAILABLE = False

# Package health check
def health_check() -> dict:
    """Perform package health check"""
    
    health = {
        "package_version": __version__,
        "gym_available": GYM_AVAILABLE,
        "context7_available": CONTEXT7_AVAILABLE,
        "dependencies": {
            "gymnasium": GYM_AVAILABLE,
            "numpy": True,  # Should always be available
            "pandas": True   # Should always be available
        }
    }
    
    return health


# Quick start example
def quick_start_example():
    """Quick start example для package usage"""
    
    if not GYM_AVAILABLE:
        print("Please install required dependencies: pip install gymnasium numpy pandas")
        return None
    
    # Create simple crypto environment
    config = CryptoTradingConfig(
        assets=["BTC", "ETH"],
        enable_sentiment_signals=True,
        initial_balance=10000.0
    )
    
    env = CryptoTradingEnvironment(config)
    
    print("Crypto Trading Environment created successfully!")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Assets: {config.assets}")
    
    return env


if __name__ == "__main__":
    # Run health check when imported directly
    health = health_check()
    print("ML Gym Environments Package Health Check:")
    print(f"Version: {health['package_version']}")
    print(f"Gymnasium Available: {health['gym_available']}")
    print(f"Context7 Available: {health['context7_available']}")
    
    # Run quick start if possible
    if health['gym_available']:
        quick_start_example()