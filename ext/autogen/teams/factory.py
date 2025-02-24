from enum import Enum
from typing import Type, TypeVar, Dict, Generic, cast
from .base import BaseTeamBuilder
from .reflect import ReflectionTeamBuilder

# 如果未来添加新的 team 类型,只需要:
# 1. 在 TeamType 中添加新的类型
# 2. 创建新的 Builder 类
# 3. 在 TeamBuilderFactory._builders 中注册
# 4. 使用时直接通过 factory 创建即可

class TeamType(Enum):
    REFLECTION = "reflection"
    # 未来可以添加更多类型
    # CODE_EXECUTION = "code_execution" 
    # DEBATE = "debate"

B = TypeVar('B', bound=BaseTeamBuilder)

class TeamBuilderFactory(Generic[B]):
    _builders: Dict[TeamType, Type[BaseTeamBuilder]] = {
        TeamType.REFLECTION: ReflectionTeamBuilder,
        # 未来可以添加更多 builder
        # TeamType.CODE_EXECUTION: CodeExecutionTeamBuilder,
        # TeamType.DEBATE: DebateTeamBuilder,
    }
    
    @classmethod
    def register_builder(cls, team_type: TeamType, builder_class: Type[B]) -> None:
        """注册新的 builder"""
        cls._builders[team_type] = builder_class
    
    @classmethod
    def create_builder(cls, team_type: TeamType) -> B:
        """创建指定类型的 builder"""
        builder_class = cls._builders.get(team_type)
        if not builder_class:
            raise ValueError(f"Unknown team type: {team_type}")
        return cast(B, builder_class())  # 使用 cast 明确返回类型