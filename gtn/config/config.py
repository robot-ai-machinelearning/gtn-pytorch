from dataclasses import dataclass

@dataclass
class LearnerConfig:
    learner_conv1_filters: int
    learner_conv2_filters: int

    learner_fc1_filters: int
    learner_learning_rate: float = 1e-2

@dataclass
class GeneratorConfig:
    generator_fc1_filters: int = 1024
    generator_fc2_filters: int = 128

    generator_conv1_filters: int = 64
    generator_conv2_filters: int = 1